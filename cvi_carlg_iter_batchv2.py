from transformers import get_linear_schedule_with_warmup, AutoModel, AutoConfig
from torch.optim import AdamW
import torch,os,time,hydra,wandb
import torch.nn as nn
from few_shot_dataloader import DataModule 
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

from utils.fast_slt_1 import _compute_silhouette_loss_carlg, StepwiseLR
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, TaskType
import warnings

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')


class BertClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Determine mode
        self.mode = self.hparams.MODE  # 'pretrain' or 'finetune'
        
        if self.hparams.get('USE_PRETRAINED_WEIGHTS', True):
            self.bert = AutoModel.from_pretrained(self.hparams.MODEL_NAME)
        else:
            config = AutoConfig.from_pretrained(self.hparams.MODEL_NAME)
            self.bert = AutoModel.from_config(config)
        
        # Configure based on mode
        if self.mode == 'pretrain':
            self._setup_pretrain()
        elif self.mode == 'finetune':
            self._setup_finetune()
        else:
            raise ValueError(f"Unknown MODE: {self.mode}. Must be 'pretrain' or 'finetune'")
        
        # Add after validation storage
        self.train_embeddings = []
        self.train_labels = []
        self.train_sample_rate = 0.4  # 40% sampling
    
    def _setup_pretrain(self):
        """Setup for unsupervised pretraining phase"""
        print("="*50)
        print("PRETRAINING MODE: Unsupervised learning with silhouette loss")
        print("="*50)
        
        # Clustering components for silhouette
        self.use_silhouette = self.hparams.USE_SILHOUETTE
        self.num_clusters = self.hparams.NUM_CLUSTERS or self.hparams.num_labels
        self.goal = self.hparams.SLT_SCORE_GOAL
        self.slt_score = _compute_silhouette_loss_carlg

        self.automatic_optimization = False
        self.cluster_centroids = [None]
        
        # Apply LoRA if specified
        if self.hparams.USE_LORA:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=self.hparams.get('LORA_R', 16),
                lora_alpha=self.hparams.get('LORA_ALPHA', 32),
                lora_dropout=self.hparams.get('LORA_DROPOUT', 0.1),
                target_modules=["query", "key", "value", "dense"]
            )
            self.bert = get_peft_model(self.bert, lora_config)
            self.bert.print_trainable_parameters()
        
        # NO CLASSIFIER HEAD in pretrain mode
        self.classifier = None
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        
        # Storage for validation kNN
        self.val_embeddings = []
        self.val_labels = []
        
    def _setup_finetune(self):
        """Setup for supervised fine-tuning phase"""
        print("="*50)
        print("FINE-TUNING MODE: Supervised learning with frozen BERT")
        print("="*50)
        
        # Load pretrained checkpoint if provided
        if self.hparams.PRETRAINED_CHECKPOINT:
            print(f"Loading pretrained BERT from: {self.hparams.PRETRAINED_CHECKPOINT}")
            checkpoint = torch.load(self.hparams.PRETRAINED_CHECKPOINT)
            # Load only BERT weights
            bert_state_dict = {k.replace('bert.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('bert.')}
            self.bert.load_state_dict(bert_state_dict, strict=False)
        
        # Freeze BERT encoder
        for param in self.bert.parameters():
            param.requires_grad = False
        print("BERT encoder frozen!")
        
        # Add classifier head (trainable)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.hparams.num_labels)
        print(f"Classifier head added: {self.bert.config.hidden_size} -> {self.hparams.num_labels}")
        
        # Standard supervised training
        self.automatic_optimization = True
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask):
        """Forward pass - returns embeddings (and logits if classifier exists)"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        dropped_output = self.dropout(cls_embedding)
        
        if self.classifier is not None:
            logits = self.classifier(dropped_output)
            return logits, cls_embedding
        else:
            return cls_embedding
    
    def training_step(self, batch, batch_idx):
        if self.mode == 'pretrain':
            return self._pretrain_step(batch, batch_idx)
        else:
            return self._finetune_step(batch, batch_idx)
    
    def _pretrain_step(self, batch, batch_idx):
        """Unsupervised pretraining with silhouette loss"""
        input_ids, attention_mask, labels = batch
        
        iterations_per_batch = self.hparams.get('ITERATIONS_PER_BATCH', 1)
        
        total_slt_loss_sum = 0.0
        total_slt_score_sum = 0.0
        
        init_centrs = [None]
        
        for iter_idx in range(iterations_per_batch):
            embeddings = self(input_ids, attention_mask)
            
            # Calculate silhouette loss
            slt_score, slt_loss, cntrs = self.slt_score(
                self.num_clusters, 
                embeddings, 
                init_centrs, 
                self.goal
            )
            init_centrs = cntrs
            
            # Accumulate metrics
            slt_score_value = slt_score.item() if isinstance(slt_score, torch.Tensor) else slt_score
            total_slt_loss_sum += slt_loss.item()
            total_slt_score_sum += slt_score_value
            
            # Backward and optimize
            self.manual_backward(slt_loss)
            
            opt = self.optimizers()
            opt.step()
            opt.zero_grad()
        
        # Average metrics
        avg_slt_loss = total_slt_loss_sum / iterations_per_batch
        avg_slt_score = total_slt_score_sum / iterations_per_batch
        
        # Log metrics
        self.log('train_slt_loss', avg_slt_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_slt_score', avg_slt_score, on_step=False, on_epoch=True, prog_bar=True)

        # After self.log() calls, before return
        if torch.rand(1).item() < self.train_sample_rate:
            self.train_embeddings.append(embeddings.detach().cpu())
            self.train_labels.append(labels.detach().cpu())
        
        return torch.tensor(avg_slt_loss, device=self.device)
    
    def _finetune_step(self, batch, batch_idx):
        """Supervised fine-tuning with classification loss"""
        input_ids, attention_mask, labels = batch
        logits, _ = self(input_ids, attention_mask)
        
        loss = self.loss_fn(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.mode == 'pretrain':
            return self._pretrain_val_step(batch, batch_idx)
        else:
            return self._finetune_val_step(batch, batch_idx)
    
    def _pretrain_val_step(self, batch, batch_idx):
        """Validation with kNN for pretraining"""
        input_ids, attention_mask, labels = batch
        embeddings = self(input_ids, attention_mask)
        
        # Store embeddings and labels for kNN
        self.val_embeddings.append(embeddings.detach().cpu())
        self.val_labels.append(labels.detach().cpu())
        
        return {'embeddings': embeddings, 'labels': labels}
    
    def _finetune_val_step(self, batch, batch_idx):
        """Validation with classification accuracy for fine-tuning"""
        input_ids, attention_mask, labels = batch
        logits, _ = self(input_ids, attention_mask)
        
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': accuracy}
    
    def on_train_epoch_end(self):

        # Add at the beginning of method, before existing code
        if self.mode == 'pretrain' and len(self.train_embeddings) > 0:
            all_train_embeddings = torch.cat(self.train_embeddings, dim=0).numpy()
            all_train_labels = torch.cat(self.train_labels, dim=0).numpy()
            
            knn = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
            knn.fit(all_train_embeddings, all_train_labels)
            predictions = knn.predict(all_train_embeddings)
            train_knn_acc = accuracy_score(all_train_labels, predictions)
            
            self.log('train_knn_acc', train_knn_acc, on_epoch=True, prog_bar=True)
            
            # Clear for next epoch
            self.train_embeddings = []
            self.train_labels = []


    def on_validation_epoch_end(self):
        if self.mode == 'pretrain':
            self._compute_knn_accuracy()
    
    def _compute_knn_accuracy(self):
        """Compute kNN accuracy on validation set"""
        if len(self.val_embeddings) == 0:
            return
        
        # Concatenate all embeddings and labels
        all_embeddings = torch.cat(self.val_embeddings, dim=0).numpy()
        all_labels = torch.cat(self.val_labels, dim=0).numpy()
        
        # Train kNN classifier (k=10)
        knn = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
        knn.fit(all_embeddings, all_labels)
        
        # Predict on same data (since we're evaluating embedding quality)
        predictions = knn.predict(all_embeddings)
        knn_acc = accuracy_score(all_labels, predictions)
        
        # Log kNN accuracy
        self.log('val_knn_acc', knn_acc, on_epoch=True, prog_bar=True)
        
        # Clear storage for next epoch
        self.val_embeddings = []
        self.val_labels = []
    
    def test_step(self, batch, batch_idx):
        if self.mode == 'pretrain':
            return self._pretrain_test_step(batch, batch_idx)
        else:
            return self._finetune_test_step(batch, batch_idx)
    
    def _pretrain_test_step(self, batch, batch_idx):
        """Test with kNN for pretraining"""
        input_ids, attention_mask, labels = batch
        embeddings = self(input_ids, attention_mask)
        
        # Store for final kNN computation
        if not hasattr(self, 'test_embeddings'):
            self.test_embeddings = []
            self.test_labels = []
        
        self.test_embeddings.append(embeddings.detach().cpu())
        self.test_labels.append(labels.detach().cpu())
        
        return {'embeddings': embeddings, 'labels': labels}
    
    def _finetune_test_step(self, batch, batch_idx):
        """Test with classification accuracy for fine-tuning"""
        input_ids, attention_mask, labels = batch
        logits, _ = self(input_ids, attention_mask)
        
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', accuracy, on_epoch=True)
        
        return {'test_loss': loss, 'test_acc': accuracy}
    
    def on_test_epoch_end(self):
        if self.mode == 'pretrain' and hasattr(self, 'test_embeddings'):
            # Compute final kNN accuracy on test set
            all_embeddings = torch.cat(self.test_embeddings, dim=0).numpy()
            all_labels = torch.cat(self.test_labels, dim=0).numpy()
            
            knn = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
            knn.fit(all_embeddings, all_labels)
            predictions = knn.predict(all_embeddings)
            knn_acc = accuracy_score(all_labels, predictions)
            
            self.log('test_knn_acc', knn_acc, on_epoch=True)
            print(f"\nTest kNN Accuracy (k=10): {knn_acc:.4f}")
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), 
            lr=self.hparams.LEARNING_RATE, 
            eps=1e-8
        )
        return optimizer


PROJECT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_DIR, "configs")

@hydra.main(config_path=CONFIG_DIR, config_name="defaults", version_base='1.2')
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    
    pl.seed_everything(123)
    
    # Setup logger
    if config['USE_LOGGER']:
        run_name = f"{config['RUN_NAME']}_{config['MODE']}"
        wandb_logger = WandbLogger(
            project=config['PROJECT_NAME'],
            name=run_name,
            notes=config.get('NOTES', ''),
            config=config,
            log_model=False,
        )
    
    # Create data module
    data_module = DataModule(config)
    data_module.setup()
    config['num_labels'] = data_module.config['num_labels']
    config['label_names'] = data_module.config['label_names']
    
    # Create model
    model = BertClassifier(config)
    
    # Callbacks based on mode
    if config['MODE'] == 'pretrain':
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(PROJECT_DIR, 'saved_models/pretrain'),
            monitor='train_slt_score',
            mode='max',
            save_top_k=1,
            filename='best-pretraink50-{epoch:02d}-{train_slt_score:.3f}'
        )
    else:  # finetune
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(PROJECT_DIR, 'saved_models/finetune'),
            monitor='val_acc',
            mode='max',
            save_top_k=1,
            filename='best-finetune-{epoch:02d}-{val_acc:.3f}'
        )
    
    early_stopping = EarlyStopping(
        #monitor='val_knn_acc' if config['MODE'] == 'pretrain' else 'val_acc',
        #monitor train slt score for pretrain
        monitor='train_slt_score' if config['MODE'] == 'pretrain' else 'val_acc',
        min_delta=0.00,
        patience=5,
        mode='max'
    )
    
    # Create trainer
    trainer = pl.Trainer(
        logger=wandb_logger if config['USE_LOGGER'] else None,
        callbacks=[checkpoint_callback], #, early_stopping],
        accelerator='gpu',
        devices=[config['DEVICE']],
        log_every_n_steps=2,
        enable_progress_bar=True,
        max_epochs= config['EPOCHS'],
        # max_epochs=1,
        # limit_train_batches=1,
        # limit_val_batches=1,
        # limit_test_batches=1
    )
    
    # Train model
    print("\n" + "="*70)
    print(f"Starting {config['MODE'].upper()} phase")
    print("="*70)
    print(OmegaConf.to_yaml(config))
    
    start_time = time.time()
    trainer.fit(model, data_module)
    end_time = time.time()
    
    training_time = (end_time - start_time) / 60
    print(f"\n{config['MODE'].capitalize()} completed in {training_time:.2f} minutes")
    
    if config['USE_LOGGER']:
        wandb_logger.experiment.log({f"{config['MODE']}_time_minutes": training_time})
    
    # Test model
    trainer.test(model, data_module)
    
    # Finish wandb run
    if config['USE_LOGGER']:
        wandb.finish()


if __name__ == '__main__':
    main()
