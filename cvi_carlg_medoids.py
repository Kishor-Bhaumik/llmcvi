from transformers import get_linear_schedule_with_warmup, AutoModel, AutoConfig
from torch.optim import AdamW
import torch,os,time,hydra,wandb
import torch.nn as nn
from few_shot_dataloader import DataModule 
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, TaskType
import warnings

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')


def get_fast_silhouette_score_with_precomputed_centroids(
    feats: torch.Tensor, 
    cluster_labels: torch.Tensor,
    global_centroids: torch.Tensor
):
    """
    Compute fast silhouette score using pre-computed global centroids.
    
    Args:
        feats: [batch_size, embedding_dim] - current batch embeddings
        cluster_labels: [batch_size] - pre-assigned cluster labels for batch
        global_centroids: [k, embedding_dim] - pre-computed KMeans centroids
    
    Returns:
        silhouette_score: scalar tensor
    """
    device = feats.device
    k = len(global_centroids)
    
    if k <= 1 or len(feats) == 0:
        return torch.tensor(0.0, device=device)
    
    # Compute distances from batch features to GLOBAL centroids
    dists_to_centroids = torch.cdist(feats, global_centroids)  # [batch_size, k]
    
    scores = []
    unique_labels = torch.unique(cluster_labels)
    
    for label in unique_labels:
        label_val = label.item()
        mask = cluster_labels == label
        curr_feats_dists = dists_to_centroids[mask]  # Features in this cluster
        
        if len(curr_feats_dists) == 0:
            continue
        
        # a(i) = distance to own cluster's GLOBAL centroid
        a = curr_feats_dists[:, label_val]
        
        # b(i) = min distance to other GLOBAL centroids
        other_mask = torch.ones(k, dtype=torch.bool, device=device)
        other_mask[label_val] = False
        
        if torch.sum(other_mask) == 0:
            continue
            
        b = torch.min(curr_feats_dists[:, other_mask], dim=1).values
        
        sils = (b - a) / (torch.max(a, b) + 1e-9)
        scores.append(sils)
    
    if not scores:
        return torch.tensor(0.0, device=device)
    
    return torch.mean(torch.cat(scores))


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
    
    def _setup_pretrain(self):
        """Setup for unsupervised pretraining phase"""
        print("="*50)
        print("PRETRAINING MODE: Unsupervised learning with silhouette loss")
        print("="*50)
        
        # Clustering components
        self.num_clusters = self.hparams.NUM_CLUSTERS or self.hparams.num_labels
        self.global_centroids = None
        self.cluster_labels_dict = {}  # Maps sample indices to cluster labels
        self.medoid_indices = {}  # Maps cluster_id -> global_sample_index
        self.medoid_input_ids = None
        self.medoid_attention_mask = None
        
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
        
        # For KNN evaluation
        self.knn_train_indices = None
        self.knn_k = self.hparams.get('KNN_K', 10)
        self.train_sample_rate = self.hparams.get('TRAIN_SAMPLE_RATE', 0.5)
        
        print(f"  Number of clusters: {self.num_clusters}")
        print(f"  KNN k: {self.knn_k}")
        print(f"  Train sample rate for KNN: {self.train_sample_rate}")
        
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
    
    def compute_kmeans_and_assignments(self, dataloader):
        """
        Run MiniBatch KMeans on entire dataset and store cluster assignments.
        """
        print("Computing KMeans centroids...")
        self.eval()
        
        # Initialize MiniBatch KMeans
        kmeans = MiniBatchKMeans(
            n_clusters=self.num_clusters,
            batch_size=512,
            max_iter=100,
            random_state=42
        )
        
        # First pass: Fit KMeans incrementally
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                embeddings = self(input_ids, attention_mask)
                
                # Fit KMeans incrementally
                kmeans.partial_fit(embeddings.cpu().numpy())
        
        # Get final centroids
        self.global_centroids = torch.tensor(
            kmeans.cluster_centers_, 
            dtype=torch.float32
        ).to(self.device)
        
        # Second pass: Assign labels AND find medoids simultaneously
        print("Assigning cluster labels and finding medoids...")
        self.cluster_labels_dict = {}
        self.medoid_indices = {}
        min_distances = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                embeddings = self(input_ids, attention_mask)
                
                # Predict labels for this batch
                batch_labels = kmeans.predict(embeddings.cpu().numpy())
                
                # Compute distances from batch to all centroids
                dists = torch.cdist(embeddings, self.global_centroids)
                
                # Store labels
                start_idx = batch_idx * self.hparams.BATCH_SIZE
                for i, label in enumerate(batch_labels):
                    global_idx = start_idx + i
                    self.cluster_labels_dict[global_idx] = label
                    
                    cluster_id = label
                    if cluster_id not in min_distances:
                        min_distances[cluster_id] = float('inf')
                    
                    # Check if this sample is closer to its centroid
                    dist_to_centroid = dists[i, cluster_id].item()
                    if dist_to_centroid < min_distances[cluster_id]:
                        min_distances[cluster_id] = dist_to_centroid
                        self.medoid_indices[cluster_id] = global_idx
        
        print(f"KMeans complete. Centroids shape: {self.global_centroids.shape}")
        print(f"Found {len(self.medoid_indices)} non-empty clusters out of {self.num_clusters}")
        print(f"Medoids found: {self.medoid_indices}")
        
        # Update self.num_clusters to actual number of non-empty clusters
        self.num_clusters = len(self.medoid_indices)
        
        # Load medoid inputs once and store them
        print("Loading medoid inputs...")
        medoid_input_ids_list = []
        medoid_attention_mask_list = []
        
        for cluster_id in sorted(self.medoid_indices.keys()):
            medoid_idx = self.medoid_indices[cluster_id]
            # Get from dataset directly
            sample = dataloader.dataset[medoid_idx]
            medoid_input_ids_list.append(sample[0])  # input_ids
            medoid_attention_mask_list.append(sample[1])  # attention_mask
        
        self.medoid_input_ids = torch.stack(medoid_input_ids_list).to(self.device)
        self.medoid_attention_mask = torch.stack(medoid_attention_mask_list).to(self.device)
        print(f"Medoid inputs loaded: {self.medoid_input_ids.shape}")
        
        # Sample indices for KNN evaluation (50% of training data)
        if self.knn_train_indices is None:
            total_samples = len(self.cluster_labels_dict)
            sample_size = int(self.train_sample_rate * total_samples)
            self.knn_train_indices = np.random.choice(
                total_samples,
                size=sample_size,
                replace=False
            )
            print(f"Sampled {len(self.knn_train_indices)} training samples for KNN evaluation")
        
        self.train()
    
    def compute_knn_accuracy(self, train_dataloader, val_dataloader):
        """
        Compute KNN accuracy on training subset and full validation set.
        """
        print("Computing KNN accuracy...")
        self.eval()
        
        # Extract embeddings and labels for training subset
        train_embeddings = []
        train_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(train_dataloader):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                embeddings = self(input_ids, attention_mask)
                
                # Track indices
                start_idx = batch_idx * self.hparams.BATCH_SIZE
                batch_indices = range(start_idx, start_idx + len(input_ids))
                
                # Only keep samples in KNN training subset
                for i, global_idx in enumerate(batch_indices):
                    if global_idx in self.knn_train_indices:
                        train_embeddings.append(embeddings[i].cpu().numpy())
                        train_labels.append(labels[i].item())
        
        train_embeddings = np.array(train_embeddings)
        train_labels = np.array(train_labels)
        
        # Fit KNN classifier
        knn = KNeighborsClassifier(n_neighbors=self.knn_k)
        knn.fit(train_embeddings, train_labels)
        
        # Compute training accuracy on the same subset
        train_preds = knn.predict(train_embeddings)
        train_acc = (train_preds == train_labels).mean()
        
        # Extract embeddings for full validation set
        val_embeddings = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                embeddings = self(input_ids, attention_mask)
                val_embeddings.append(embeddings.cpu().numpy())
                val_labels.append(labels.cpu().numpy())
        
        val_embeddings = np.concatenate(val_embeddings, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        
        # Compute validation accuracy
        val_preds = knn.predict(val_embeddings)
        val_acc = (val_preds == val_labels).mean()
        
        self.train()
        return train_acc, val_acc
    
    def training_step(self, batch, batch_idx):
        if self.mode == 'pretrain':
            return self._pretrain_step(batch, batch_idx)
        else:
            return self._finetune_step(batch, batch_idx)
    
    def _pretrain_step(self, batch, batch_idx):
        """Unsupervised pretraining with silhouette loss and medoids"""
        input_ids, attention_mask, labels = batch
        
        # Concatenate batch inputs + medoid inputs
        all_input_ids = torch.cat([input_ids, self.medoid_input_ids], dim=0)
        all_attention_mask = torch.cat([attention_mask, self.medoid_attention_mask], dim=0)
        
        # Forward pass both batch and medoids together
        all_embeddings = self(all_input_ids, all_attention_mask)
        batch_embeddings = all_embeddings[:len(input_ids)]
        medoid_embeddings = all_embeddings[len(input_ids):]
        
        # Get pre-assigned cluster labels for this batch
        start_idx = batch_idx * self.hparams.BATCH_SIZE
        batch_indices = range(start_idx, start_idx + len(input_ids))
        batch_cluster_labels = torch.tensor(
            [self.cluster_labels_dict[idx] for idx in batch_indices],
            dtype=torch.long,
            device=self.device
        )
        
        # Compute silhouette with UPDATED centroids (medoid embeddings)
        slt_score = get_fast_silhouette_score_with_precomputed_centroids(
            batch_embeddings,
            batch_cluster_labels,
            medoid_embeddings
        )
        
        loss = -slt_score
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_slt_score', slt_score, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
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
    
    def on_train_epoch_start(self):
        """
        Before each epoch, recompute KMeans with frozen model.
        """
        if self.mode == 'pretrain':
            if self.current_epoch == 0 or self.global_centroids is None:
                print(f"\n=== Epoch {self.current_epoch}: Computing KMeans ===")
                self.compute_kmeans_and_assignments(self.trainer.datamodule.train_dataloader())
            else:
                print(f"\n=== Epoch {self.current_epoch}: Recomputing KMeans ===")
                self.compute_kmeans_and_assignments(self.trainer.datamodule.train_dataloader())
    
    def on_train_epoch_end(self):
        """
        After each epoch, compute KNN accuracy.
        """
        if self.mode == 'pretrain':
            train_knn_acc, val_knn_acc = self.compute_knn_accuracy(
                self.trainer.datamodule.train_dataloader(),
                self.trainer.datamodule.val_dataloader()
            )
            
            self.log('train_knn_acc', train_knn_acc, prog_bar=True)
            self.log('val_knn_acc', val_knn_acc, prog_bar=True)
            
            print(f"Epoch {self.current_epoch} - Train KNN Acc: {train_knn_acc:.4f}, Val KNN Acc: {val_knn_acc:.4f}")
    
    def validation_step(self, batch, batch_idx):
        if self.mode == 'pretrain':
            # No validation step needed for pretrain (KNN computed in on_train_epoch_end)
            return None
        else:
            return self._finetune_val_step(batch, batch_idx)
    
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
            
            knn = KNeighborsClassifier(n_neighbors=self.knn_k, metric='euclidean')
            knn.fit(all_embeddings, all_labels)
            predictions = knn.predict(all_embeddings)
            knn_acc = accuracy_score(all_labels, predictions)
            
            self.log('test_knn_acc', knn_acc, on_epoch=True)
            print(f"\nTest kNN Accuracy (k={self.knn_k}): {knn_acc:.4f}")
    
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
            # add NUM_CLUSTERS to filename
            filename=f'best-pretrain-epoch={{epoch:02d}}-slt={{slt:.3f}}-k={model.num_clusters}'
            #filename='best-pretrain-{epoch:02d}-{val_knn_acc:.3f}'
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
        max_epochs=config['EPOCHS'],
        check_val_every_n_epoch=999 if config['MODE'] == 'pretrain' else 1,
        # limit_train_batches=1,
        # limit_val_batches=1,
        # limit_test_batches=1,

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
