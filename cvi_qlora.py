
from transformers import BertModel, get_linear_schedule_with_warmup, AutoModel
from torch.optim import AdamW
import torch,os,time,hydra,wandb
import torch.nn as nn
#from dataloader import DataModule
from few_shot_dataloader import DataModule 
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils.fast_slt import _compute_silhouette_loss
from utils.fast_slt_wo_kmeans import wo_compute_silhouette_loss
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, TaskType
import warnings
from collections import deque
# remove all warnings
warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision('medium')  # Set precision for matmul operations

class BertClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

# Add clustering components for silhouette
        self.use_silhouette = self.hparams.get('USE_SILHOUETTE', False)
        self.num_clusters = self.hparams.get('NUM_CLUSTERS', self.hparams.num_labels)
        self.goal = self.hparams.get('SLT_score_GOAL', 0.8)
        self.slt_score = _compute_silhouette_loss #wo_compute_silhouette_loss #
        # Model components
        self.bert = AutoModel.from_pretrained(self.hparams.MODEL_NAME)
        if self.hparams.get('USE_LORA', False):
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=self.hparams.get('LORA_R', 16),  # rank
                lora_alpha=self.hparams.get('LORA_ALPHA', 32),
                lora_dropout=self.hparams.get('LORA_DROPOUT', 0.1),
                target_modules=["query", "key", "value", "dense"] ) # BERT attention layers)

            self.bert = get_peft_model(self.bert, lora_config)

        self.bert.print_trainable_parameters()
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.hparams.num_labels)
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        self.validation_embeddings = []
        self.validation_labels = []
        self.batch_feat, self.epch_feat = [], []
        self.slt_loss = 0

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        pooled_output = outputs.pooler_output
        dropped_output = self.dropout(pooled_output)
        logits = self.classifier(dropped_output)
        return logits, cls_embedding
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits, embeddings = self(input_ids, attention_mask)
        total_loss = self.hparams['MAIN_LOSS_WEIGHT'] + self.loss_fn(logits, labels)
        if self.hparams['USE_SILHOUETTE']:
            
            if self.hparams['CALCULATE_SLT_AFTER']==  "EACH_BATCH":
                slt_score,slt_loss=self.slt_score(self.num_clusters, embeddings, goal=self.goal)
                self.slt_loss = slt_loss
                self.log( "slt_score", slt_score, on_step=False, on_epoch=True, prog_bar=False)

            elif self.hparams['CALCULATE_SLT_AFTER']==  "EACH_EPOCH":
                self.epch_feat.append((embeddings.detach().cpu(), labels.detach().cpu()))

            else:
                calc_interval = int(self.hparams['CALCULATE_SLT_AFTER'])
                self.batch_feat.append((embeddings.detach().cpu(), labels.detach().cpu()))

                if (batch_idx + 1) % calc_interval == 0:
                    all_embeddings = torch.cat([emb for emb, _ in self.batch_feat])
                    all_labels = torch.cat([lbl for _, lbl in self.batch_feat])
                    slt_score,slt_loss = self.slt_score(self.num_clusters, all_embeddings, goal=self.goal)
                    self.slt_loss = slt_loss
                    self.batch_feat = []
                    self.log( "slt_score", slt_score, on_step=False, on_epoch=True, prog_bar=False)

            #total_loss += self.hparams['SILHOUETTE_WEIGHT']*self.slt_score(self.num_clusters, embeddings, goal=self.goal)
            total_loss += self.hparams['SILHOUETTE_WEIGHT'] * self.slt_loss

        self.log('train_loss', total_loss,on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def on_train_epoch_end(self):
        if self.hparams['USE_SILHOUETTE'] and self.hparams['CALCULATE_SLT_AFTER'] == "EACH_EPOCH":
            if self.epch_feat:
                all_embeddings = torch.cat([emb for emb, _ in self.epch_feat])
                all_labels = torch.cat([lbl for _, lbl in self.epch_feat])
                slt_score,slt_loss= self.slt_score(self.num_clusters, all_embeddings, goal=self.goal)
                self.slt_loss = slt_loss
                self.log( "slt_score", slt_score, on_step=False, on_epoch=True, prog_bar=False)
                # Clear for next epoch
                self.epch_feat = []

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits, embeddings = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': accuracy, 'preds': preds, 'labels': labels}
    

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits, _ = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', accuracy, on_epoch=True)
        
        return {'test_loss': loss, 'test_acc': accuracy, 'preds': preds, 'labels': labels}

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), 
            lr=self.hparams.LEARNING_RATE, 
            eps=1e-8)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'}}


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
        callbacks=[early_stopping], #[checkpoint_callback, early_stopping],
        accelerator='gpu',
        devices=[config['DEVICE']], 
        gradient_clip_val=1.0,
        log_every_n_steps=2,
        enable_progress_bar=True,
        max_epochs=config['EPOCHS'],
        # limit_train_batches=config.get('LIMIT_TRAIN_BATCHES', 1),
        # limit_val_batches=config.get('LIMIT_VAL_BATCHES', 1),
        # limit_test_batches=config.get('LIMIT_TEST_BATCHES', 1)
    )
    
    # Train model
    start_time= time.time()
    trainer.fit(model, data_module)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    #log training time
    if config['USE_LOGGER']:
        wandb.log({"training_time_minutes": (end_time - start_time) / 60})
    
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