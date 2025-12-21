import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import os
import time
import hydra
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
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


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config['BATCH_SIZE']
        self.num_workers = config.get('NUM_WORKERS', 4)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = datasets.MNIST(
                root=self.config.get('DATA_DIR', './data'),
                train=True,
                download=True,
                transform=self.transform
            )
            
            self.val_dataset = datasets.MNIST(
                root=self.config.get('DATA_DIR', './data'),
                train=False,
                download=True,
                transform=self.transform
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.MNIST(
                root=self.config.get('DATA_DIR', './data'),
                train=False,
                download=True,
                transform=self.transform
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )


class UnsupervisedCNN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        self.num_clusters = config['NUM_CLUSTERS']
        self.embedding_dim = config.get('EMBEDDING_DIM', 128)
        
        # CNN Architecture (Option A: Simpler embedding)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, self.embedding_dim)  # Embedding layer
        
        # For pretraining phase
        self.global_centroids = None
        self.cluster_labels_dict = {}  # Maps sample indices to cluster labels
        self.medoid_indices = {}  # Maps cluster_id -> global_sample_index
        
        # For NMI calculation
        self.nmi_sample_indices = None
        self.nmi_sample_size = config.get('NMI_SAMPLE_SIZE', 10000)
        
        print(f"Unsupervised CNN Architecture:")
        print(f"  Conv1: 1 -> 10 (kernel_size=5)")
        print(f"  Conv2: 10 -> 20 (kernel_size=5)")
        print(f"  FC1 (Embedding): 320 -> {self.embedding_dim}")
        print(f"  Number of clusters: {self.num_clusters}")
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        embedding = F.relu(self.fc1(x))
        return embedding
    
    def compute_kmeans_and_assignments(self, dataloader):
        """
        Run MiniBatch KMeans on entire dataset and store cluster assignments.
        """
        print("Computing KMeans centroids...")
        self.eval()
        
        # Initialize MiniBatch KMeans
        kmeans = MiniBatchKMeans(
            n_clusters=self.num_clusters,
            batch_size=1024,
            max_iter=100,
            random_state=42
        )
        
        # First pass: Fit KMeans incrementally
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(dataloader):
                images = images.to(self.device)
                embeddings = self(images)
                
                # Fit KMeans incrementally
                kmeans.partial_fit(embeddings.cpu().numpy())
        
        # Get final centroids
        self.global_centroids = torch.tensor(
            kmeans.cluster_centers_, 
            dtype=torch.float32
        ).to(self.device)
        
        # Second pass: Assign labels batch by batch
        # Second pass: Assign labels AND find medoids simultaneously
        print("Assigning cluster labels and finding medoids...")
        self.cluster_labels_dict = {}
        self.medoid_indices = {}  # Only store clusters that have samples
        min_distances = {}  # Track minimum distance per cluster

        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(dataloader):
                images = images.to(self.device)
                embeddings = self(images)
                
                # Predict labels for this batch
                batch_labels = kmeans.predict(embeddings.cpu().numpy())
                
                # Compute distances from batch to all centroids
                dists = torch.cdist(embeddings, self.global_centroids)  # [batch_size, k]
                
                # Store labels
                start_idx = batch_idx * dataloader.batch_size
                for i, label in enumerate(batch_labels):
                    global_idx = start_idx + i
                    self.cluster_labels_dict[global_idx] = label
                    
                    # Initialize min_distance for this cluster if first time seeing it
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

        # Load medoid images once and store them
        print("Loading medoid images...")
        medoid_images = []
        for cluster_id in sorted(self.medoid_indices.keys()):  # Sort for consistency
            medoid_idx = self.medoid_indices[cluster_id]
            medoid_img, _ = dataloader.dataset[medoid_idx]
            medoid_images.append(medoid_img)

        self.medoid_images = torch.stack(medoid_images).to(self.device)
        print(f"Medoid images loaded: {self.medoid_images.shape}")

        # Sample indices for NMI calculation
        if self.nmi_sample_indices is None:
            total_samples = len(self.cluster_labels_dict)
            self.nmi_sample_indices = np.random.choice(
                total_samples, 
                size=min(self.nmi_sample_size, total_samples),
                replace=False
            )
        
        self.train()


    def compute_nmi(self, dataloader):
        """
        Compute NMI on a subset of training data.
        """
        self.eval()
        
        predicted_labels = []
        true_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)
                embeddings = self(images)
                
                # Get distances to centroids
                dists = torch.cdist(embeddings, self.global_centroids)
                pred_clusters = torch.argmin(dists, dim=1).cpu().numpy()
                
                # Track indices
                start_idx = batch_idx * dataloader.batch_size
                batch_indices = range(start_idx, start_idx + len(images))
                
                # Only keep samples in NMI subset
                for i, global_idx in enumerate(batch_indices):
                    if global_idx in self.nmi_sample_indices:
                        predicted_labels.append(pred_clusters[i])
                        true_labels.append(labels[i].item())
        
        if len(predicted_labels) > 0:
            nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        else:
            nmi = 0.0
        
        self.train()
        return nmi
    
    def training_step(self, batch, batch_idx):
        images, _ = batch
        
        # Forward pass batch + medoids together
        all_images = torch.cat([images, self.medoid_images], dim=0)
        all_embeddings = self(all_images)
        batch_embeddings = all_embeddings[:len(images)]
        medoid_embeddings = all_embeddings[len(images):]
        
        # Get pre-assigned cluster labels for this batch
        start_idx = batch_idx * self.trainer.datamodule.batch_size
        batch_indices = range(start_idx, start_idx + len(images))
        batch_cluster_labels = torch.tensor(
            [self.cluster_labels_dict[idx] for idx in batch_indices],
            dtype=torch.long,
            device=self.device
        )
        
        # Compute silhouette with UPDATED centroids (medoid embeddings)
        slt_score = get_fast_silhouette_score_with_precomputed_centroids(
            batch_embeddings,
            batch_cluster_labels,
            medoid_embeddings  # ← Changed from self.global_centroids
        )
        
        loss = -slt_score
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_slt_score', slt_score, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_train_epoch_start(self):
        """
        Before each epoch, recompute KMeans with frozen model.
        """
        if self.current_epoch == 0 or self.global_centroids is None:
            print(f"\n=== Epoch {self.current_epoch}: Computing KMeans ===")
            self.compute_kmeans_and_assignments(self.trainer.datamodule.train_dataloader())
        else:
            print(f"\n=== Epoch {self.current_epoch}: Recomputing KMeans ===")
            self.compute_kmeans_and_assignments(self.trainer.datamodule.train_dataloader())
    
    def on_train_epoch_end(self):
        """
        After each epoch, compute NMI.
        """
        nmi = self.compute_nmi(self.trainer.datamodule.train_dataloader())
        self.log('train_nmi', nmi, prog_bar=True)
        print(f"Epoch {self.current_epoch} NMI: {nmi:.4f}")
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.LEARNING_RATE,
            momentum=self.hparams.MOMENTUM
        )
        return optimizer


class FinetuneCNN(pl.LightningModule):
    def __init__(self, pretrained_model, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Copy pretrained layers and freeze them
        self.conv1 = pretrained_model.conv1
        self.conv2 = pretrained_model.conv2
        self.conv2_drop = pretrained_model.conv2_drop
        self.fc1 = pretrained_model.fc1
        
        # Freeze pretrained layers
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False
        for param in self.conv2_drop.parameters():
            param.requires_grad = False
        for param in self.fc1.parameters():
            param.requires_grad = False
        
        # Add new classification head
        self.classifier = nn.Linear(pretrained_model.embedding_dim, 10)
        
        self.loss_fn = nn.CrossEntropyLoss()
        
        print(f"Finetuning CNN:")
        print(f"  Frozen: conv1, conv2, fc1")
        print(f"  Trainable: classifier ({pretrained_model.embedding_dim} -> 10)")
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        embedding = F.relu(self.fc1(x))
        logits = self.classifier(embedding)
        return logits
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': accuracy}
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', accuracy, on_epoch=True)
        
        return {'test_loss': loss, 'test_acc': accuracy}
    
    def configure_optimizers(self):
        # Only optimize the classifier
        optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams.FINETUNE_LR
        )
        return optimizer


PROJECT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_DIR, "configs")

@hydra.main(config_path=CONFIG_DIR, config_name="mnist_unsupervised", version_base='1.2')
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    
    pl.seed_everything(config['SEED'])
    
    if config['USE_LOGGER']:
        wandb_logger = WandbLogger(
            project=config['PROJECT_NAME'],
            name=config['RUN_NAME'],
            config=config,
            log_model=False,
        )
    
    # Create data module
    data_module = MNISTDataModule(config)
    data_module.setup()
    
    # ============================================
    # PHASE 1: UNSUPERVISED PRETRAINING
    # ============================================
    print("\n" + "="*60)
    print("PHASE 1: UNSUPERVISED PRETRAINING")
    print("="*60)
    
    pretrain_model = UnsupervisedCNN(config)
    
    pretrain_checkpoint = ModelCheckpoint(
        monitor='train_nmi',
        mode='max',
        save_top_k=1,
        filename='pretrain-best-{epoch:02d}-{train_nmi:.3f}'
    )
    
    pretrain_trainer = pl.Trainer(
        logger=wandb_logger if config['USE_LOGGER'] else None,
        callbacks=[pretrain_checkpoint],
        accelerator='gpu',
        devices=[config['DEVICE']],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_progress_bar=True,
        max_epochs=config['PRETRAIN_EPOCHS'],
        check_val_every_n_epoch=999,  # Skip validation during pretraining
    )
    
    start_time = time.time()
    pretrain_trainer.fit(pretrain_model, data_module)
    pretrain_time = time.time() - start_time
    print(f"Pretraining completed in {pretrain_time:.2f} seconds")
    
    # ============================================
    # PHASE 2: SUPERVISED FINETUNING
    # ============================================
    print("\n" + "="*60)
    print("PHASE 2: SUPERVISED FINETUNING")
    print("="*60)
    #pretrain_model = UnsupervisedCNN(config)
    
    finetune_model = FinetuneCNN(pretrain_model, config)
    
    finetune_checkpoint = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        filename='finetune-best-{epoch:02d}-{val_acc:.3f}'
    )
    
    finetune_trainer = pl.Trainer(
        logger=wandb_logger if config['USE_LOGGER'] else None,
        callbacks=[finetune_checkpoint],
        accelerator='gpu',
        devices=[config['DEVICE']],
        log_every_n_steps=10,
        enable_progress_bar=True,
        max_epochs=config['FINETUNE_EPOCHS'],
    )
    
    start_time = time.time()
    finetune_trainer.fit(finetune_model, data_module)
    finetune_time = time.time() - start_time
    print(f"Finetuning completed in {finetune_time:.2f} seconds")
    
    # Test final model
    finetune_trainer.test(finetune_model, data_module)
    
    # Log total training time
    if config['USE_LOGGER']:
        wandb_logger.experiment.log({
            "pretrain_time_minutes": pretrain_time / 60,
            "finetune_time_minutes": finetune_time / 60,
            "total_time_minutes": (pretrain_time + finetune_time) / 60
        })
        wandb.finish()


if __name__ == '__main__':
    main()




''' 

Pretraining Phase (Unsupervised):

Before each epoch: Runs MiniBatch KMeans on entire dataset (incrementally, no memory issues)

Extracts embeddings with frozen model
Computes global centroids
Assigns cluster labels to all samples


During training:

Uses pre-computed centroids and pre-assigned labels
Computes fast silhouette score for each batch
Loss = abs(1 - silhouette_score)
Updates model parameters


After each epoch:

Computes NMI on a 10k sample subset (efficient)
NMI shows how well learned clusters align with true digits



Finetuning Phase (Supervised):

Freezes all pretrained layers (conv1, conv2, fc1)
Adds new linear classifier (128 → 10)
Trains for 10 epochs with true labels
Standard cross-entropy loss

'''
