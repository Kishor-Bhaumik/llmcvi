from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from tqdm import tqdm
from dataloader import DataModule  # Importing the DataModule from dataloader.py
import torch.nn.functional as F
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils.fast_slt import _compute_silhouette_loss
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
import hydra
from omegaconf import DictConfig, OmegaConf



torch.set_float32_matmul_precision('medium')  # Set precision for matmul operations
class BertClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

# Add clustering components for silhouette
        self.use_silhouette = self.hparams.get('USE_SILHOUETTE', False)
        self.silhouette_weight = self.hparams.get('SILHOUETTE_WEIGHT', 0.1)
        self.num_clusters = self.hparams.get('NUM_CLUSTERS', self.hparams.num_labels)
        self.slt_score = _compute_silhouette_loss 
        # Model components
        self.bert = BertModel.from_pretrained(self.hparams.MODEL_NAME)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.hparams.num_labels)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # For collecting embeddings during validation
        self.validation_embeddings = []
        self.validation_labels = []

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
        
        # Main classification loss
        classification_loss = self.loss_fn(logits, labels)
        total_loss = classification_loss
        
        # Add silhouette loss if enabled and after warmup
        if self.use_silhouette and self.current_epoch >= self.hparams.get('SILHOUETTE_START_EPOCH', 0):
            silhouette_loss = self.slt_score(self.num_clusters, embeddings)
            if not torch.isnan(silhouette_loss):
                total_loss = classification_loss + self.silhouette_weight * silhouette_loss
                self.log('silhouette_loss', silhouette_loss, on_step=True, on_epoch=True)
        
        # Calculate accuracy (same as before)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('classification_loss', classification_loss, on_step=True, on_epoch=True)
        self.log('train_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits, embeddings = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        
        # Store embeddings and labels for silhouette calculation
        self.validation_embeddings.append(embeddings.detach().cpu())
        self.validation_labels.append(labels.detach().cpu())
        
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
    
    if config['USE_LOGGER']:
        wandb_logger = WandbLogger(
        project=config['PROJECT_NAME'],
        config=config )
    
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
        filename='best-checkpoint-{epoch:02d}-{val_acc:.3f}'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['EPOCHS'],
        logger=wandb_logger if config['USE_LOGGER'] else None,
        callbacks=[early_stopping], #[checkpoint_callback, early_stopping],
        accelerator='gpu',
        devices=[config['DEVICE']], 
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        enable_progress_bar=True,
        limit_train_batches=config.get('LIMIT_TRAIN_BATCHES', 20),
        limit_val_batches=config.get('LIMIT_VAL_BATCHES', 1),
        limit_test_batches=config.get('LIMIT_TEST_BATCHES', 1)
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    trainer.test(model, data_module, ckpt_path='best')
    
    # Finish wandb run
    wandb.finish() if config['USE_LOGGER'] else None


if __name__ == '__main__':
    main()