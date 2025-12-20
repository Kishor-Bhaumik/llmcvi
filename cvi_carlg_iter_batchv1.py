
from transformers import get_linear_schedule_with_warmup, AutoModel
from torch.optim import AdamW
import torch,os,time,hydra,wandb
import torch.nn as nn
#from dataloader import DataModule
from few_shot_dataloader import DataModule 
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utils.fast_slt_1 import _compute_silhouette_loss_carlg, StepwiseLR, _compute_silhouette_loss_spectral,_compute_silhouette_loss_kmnsplus
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, TaskType
import warnings
from collections import deque
# remove all warnings
warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision('medium')  # Set precision for matmul operations

'''
we are running k means and batches over each batch we are calculating silhouette score
problem in this code : the projection head is not learning 
the validation accuracy might be garbage

'''


class BertClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

# Add clustering components for silhouette
        self.use_silhouette = self.hparams.USE_SILHOUETTE
        # Use num_labels as fallback
        self.num_clusters = self.hparams.NUM_CLUSTERS or self.hparams.num_labels
        self.goal = self.hparams.SLT_SCORE_GOAL
        if self.hparams.CLUSTER_TYPE == 'kmeans':
            self.slt_score = _compute_silhouette_loss_carlg
        elif self.hparams.CLUSTER_TYPE == 'spectral':
            self.slt_score = _compute_silhouette_loss_spectral
        elif self.hparams.CLUSTER_TYPE == 'kmnsplus':
            self.slt_score = _compute_silhouette_loss_kmnsplus
        else:
            raise ValueError(f"Unknown CLUSTER_TYPE: {self.hparams.CLUSTER_TYPE}")
    
        self.epoch_slt_loss = 0.0
        self.automatic_optimization = False 

        self.bert = AutoModel.from_pretrained(self.hparams.MODEL_NAME)
        self.warmup_epochs = 1 
        self.cluster_centroids = [None] 

        self.beta_scheduler = StepwiseLR(
            optimizer=None,  # No optimizer, just for beta scheduling
            init_lr=0.0,
            gamma=0.95,
            decay_rate=0.75
        )
        self.old_batch , self.old_label = None, None

        if self.hparams.USE_LORA:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=self.hparams.get('LORA_R', 16),  # rank
                lora_alpha=self.hparams.get('LORA_ALPHA', 32),
                lora_dropout=self.hparams.get('LORA_DROPOUT', 0.1),
                target_modules=["query", "key", "value", "dense"]  # use it when task_type is TaskType.FEATURE_EXTRACTION
                ) # BERT attention layers)

            self.bert = get_peft_model(self.bert, lora_config)

        self.bert.print_trainable_parameters() 
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.hparams.num_labels)
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        self.validation_embeddings = []
        self.validation_labels = []
        self.batch_feat, self.epch_feat = [], []

        self.temps= []
        

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        dropped_output = self.dropout(cls_embedding)
        logits = self.classifier(dropped_output)
        return logits, cls_embedding

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch

        # if batch idx 0 , store the inputs and labels in the self.temps
        # if batch_idx == 0:
        #     self.temps.append((input_ids, attention_mask, labels))
        
        # if batch_idx == 4:
        #     # use the stored batch for training
        #     input_ids, attention_mask, labels = self.temps[0][0], self.temps[0][1], self.temps[0][2]
        
        # Get number of iterations per batch from config (default to 1 if not specified)
        iterations_per_batch = self.hparams.get('ITERATIONS_PER_BATCH', 1)
        
        # Initialize accumulators for averaging
        total_loss_sum = 0.0
        total_acc_sum = 0.0
        slt_score_sum = 0.0
        
        # Store iteration-wise silhouette scores for this batch
        iteration_scores = []
        
        init_centrs = [None]
        
        # Run N iterations on the same batch
        for iter_idx in range(iterations_per_batch):
            logits, embeddings = self(input_ids, attention_mask)
            
            # Calculate main loss
            if self.hparams['COMPELETELY_UNSUPERVISED']:
                main_loss = 0.0
            else:
                main_loss = self.hparams['MAIN_LOSS_WEIGHT'] * self.loss_fn(logits, labels)
            
            # Calculate silhouette loss
            slt_loss = 0.0
            slt_score = 0.0
            
            if self.hparams['USE_SILHOUETTE']:
                # Calculate silhouette on every iteration
                slt_score, slt_loss, cntrs = self.slt_score(self.num_clusters, embeddings, init_centrs, self.goal)
                init_centrs = cntrs
                
                # Store score for this iteration
                slt_score_value = slt_score.item() if isinstance(slt_score, torch.Tensor) else slt_score
                iteration_scores.append(slt_score_value)
                
                beta = self.hparams['SLT_LOSS_WEIGHT'] if not self.hparams['COMPELETELY_UNSUPERVISED'] else 1.0
                slt_loss = beta * slt_loss
            
            # Total loss for this iteration
            iter_total_loss = main_loss + slt_loss
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == labels).float().mean()
            
            # Accumulate metrics
            total_loss_sum += iter_total_loss.item()
            total_acc_sum += accuracy.item()
            slt_score_sum += slt_score_value if self.hparams['USE_SILHOUETTE'] else 0
            
            # Backward and optimize
            self.manual_backward(iter_total_loss)
            
            # Get optimizer and step
            opt = self.optimizers()
            opt.step()
            opt.zero_grad()
        
        # Average metrics across iterations
        avg_total_loss = total_loss_sum / iterations_per_batch
        avg_accuracy = total_acc_sum / iterations_per_batch
        avg_slt_score = slt_score_sum / iterations_per_batch
        
        # Log averaged metrics
        self.log('train_loss', avg_total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', avg_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        if self.hparams['USE_SILHOUETTE']:
            self.log('slt_score', avg_slt_score, on_step=False, on_epoch=True, prog_bar=False)
            

        
        # Return the averaged loss (for logging purposes)
        # if batch_idx == 5: exit(0)
        return torch.tensor(avg_total_loss, device=self.device)




    def on_train_epoch_end(self):
        
        if self.hparams['USE_SILHOUETTE'] and self.hparams['CALCULATE_SLT_AFTER'] == "EACH_EPOCH":
            all_embeddings = torch.cat([emb for emb, _ in self.epch_feat])
            all_labels = torch.cat([lbl for _, lbl in self.epch_feat])
            slt_score,slt_loss,cntrs= self.slt_score(self.num_clusters, all_embeddings,self.cluster_centroids, goal=self.goal)
            self.epoch_slt_loss = slt_loss
            self.log( "slt_score", slt_score, on_step=False, on_epoch=True, prog_bar=False)
            # Clear for next epoch
            self.epch_feat = []
            self.batch_feat = []
            self.cluster_centroids = cntrs

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits, embeddings = self(input_ids, attention_mask)
        # if self.hparams['COMPELETELY_UNSUPERVISED']:
        #     score,loss = self.slt_score(self.num_clusters, embeddings, goal=self.goal)
        #     a_key = 'val_slt_score'
        #     a_value = score

        # else:
        # if batch_idx > 0:
        #     input_ids,  labels = self.old_batch, self.old_label
        loss = self.loss_fn(logits, labels)
        a_key = 'val_loss'
        a_value = loss
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_epoch=True, prog_bar=True)

        return {a_key: a_value, 'val_acc': accuracy, 'preds': preds, 'labels': labels}


    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits, _ = self(input_ids, attention_mask)
        # if self.hparams['COMPELETELY_UNSUPERVISED']:
        #     score,loss = self.slt_score(self.num_clusters, logits, goal=self.goal)
        # else:
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', accuracy, on_epoch=True)
        
        return {'test_loss': loss, 'test_acc': accuracy, 'preds': preds, 'labels': labels}

    # def configure_optimizers(self):
    #     optimizer = AdamW(
    #         self.parameters(), 
    #         lr=self.hparams.LEARNING_RATE, 
    #         eps=1e-8)
    #     total_steps = self.trainer.estimated_stepping_batches
    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=0,
    #         num_training_steps=total_steps)
        
    #     return {
    #         'optimizer': optimizer,
    #         'lr_scheduler': {
    #             'scheduler': scheduler,
    #             'interval': 'step'}}
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), 
            lr=self.hparams.LEARNING_RATE, 
            eps=1e-8)
        
        return optimizer

PROJECT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_DIR, "configs")

@hydra.main(config_path=CONFIG_DIR, config_name="defaults", version_base='1.2')
def main(cfg: DictConfig):
    # Configuration
    config = OmegaConf.to_container(cfg, resolve=True)
    
    pl.seed_everything(123)
    if config['USE_LOGGER']:
        wandb_logger = WandbLogger(
        project=config['PROJECT_NAME'],
        name=config['RUN_NAME'],
        notes=config['NOTES_w_cvi'] if config['USE_SILHOUETTE'] else config['NOTES_wo_cvi'],
        config=config ,
        log_model=False,
        )

    # Create data module
    data_module = DataModule(config)
    data_module.setup()  # Setup to get num_labels
    config['num_labels'] = data_module.config['num_labels']
    config['label_names'] = data_module.config['label_names']
    
    # Create model
    model = BertClassifier(config)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        filename='best-checkpoint-{epoch:02d}-{val_acc:.3f}' )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )
    
    # Create trainer
    trainer = pl.Trainer(
        
        logger=wandb_logger if config['USE_LOGGER'] else None,
        #callbacks=[early_stopping], #[checkpoint_callback, early_stopping],
        accelerator='gpu',
        devices=[config['DEVICE'] ],
        #devices=[config['DEVICE'], 1 ],strategy='deepspeed', 
        #gradient_clip_val=1.0,
        log_every_n_steps=2,
        enable_progress_bar=True,
        #val_check_interval=1,
        max_epochs=config['EPOCHS'],
        limit_train_batches= 5, #config.get('LIMIT_TRAIN_BATCHES', 1),
        limit_val_batches=1, #config.get('LIMIT_VAL_BATCHES', 1),
        limit_test_batches=1, #config.get('LIMIT_TEST_BATCHES', 1)
    )
    
    # Train model
    print(OmegaConf.to_yaml(config))
    start_time= time.time()
    trainer.fit(model, data_module)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    #log training time
    if config['USE_LOGGER']:
        wandb_logger.experiment.log({"training_time_minutes": (end_time - start_time) / 60})
    
    # Test model
    trainer.test(model, data_module ) #, ckpt_path='best')
    
    # Finish wandb run
    if config['USE_LOGGER']: wandb.finish()


if __name__ == '__main__':
    main()



'''
okay this is bert-base-uncased.

so I want to change the model in such wy that it calculates the scores after 3 , 6 , 9 and 12 layers of the bert model.

the score from each layer will be logged. but for which layers siloutee loss will be calculated it will be mentioned
 in config file as
"CALCULATE_SLT_AFTER_LAYERS": 3 # or 6 or 9 or 12

'''

#. cd /home/kbhau001/llm/loracvi/llmcvi && conda activate cvi


            # Save iteration-wise silhouette scores to text file
            # if len(iteration_scores) > 0:
            #     # Create directory if it doesn't exist
            #     save_dir = os.path.join(PROJECT_DIR, 'batch_slt_'+str(self.current_epoch))
            #     os.makedirs(save_dir, exist_ok=True)
                
            #     # Save to file: batch_<idx>.txt
            #     file_path = os.path.join(save_dir, f'batch_{batch_idx}.txt')
            #     with open(file_path, 'w') as f:
            #         f.write('iteration,slt_score\n')
            #         for iter_idx, score in enumerate(iteration_scores):
            #             f.write(f'{iter_idx},{score}\n')