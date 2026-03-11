"""
DeepCluster-inspired unsupervised training on MNIST.

- Randomly initialized MobileNetV3-Small + Linear projection head (1024 -> 64)
- MiniBatchKMeans clustering every epoch on fresh 64-dim features
- UnifLabelSampler (from DeepCluster) for balanced cluster sampling
- Silhouette loss with pseudo-labels (unsupervised)
- Evaluation: KNN, NMI, Hungarian accuracy with true labels (end only)
- UMAP visualization with true labels, white background
- WandB logging

Install deps:
    pip install torch torchvision scikit-learn tqdm wandb umap-learn matplotlib
"""

import numpy as np
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import umap

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# ─── Config ───────────────────────────────────────────────────────────────────
BATCH_SIZE   = 256
NUM_WORKERS  = 4
EPOCHS       = 10
LR           = 1e-3
PROJ_DIM     = 64
KNN_K        = 5
NMB_CLUSTERS = 10
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = True
print(f"Using device: {DEVICE}")

# ─── WandB ────────────────────────────────────────────────────────────────────
if logger:
    wandb.init(
        project="deepcluster-mnist",
        name=f"silhouette_deepclsuter",
        config={
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "proj_dim": PROJ_DIM,
            "knn_k": KNN_K,
            "nmb_clusters": NMB_CLUSTERS,
            "loss": "silhouette",
            "backbone": "mobilenet_v3_small_random_init",
        }
    )

# ─── Transforms ───────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── Datasets ─────────────────────────────────────────────────────────────────
print("Loading MNIST...")
train_dataset_base = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_dataset       = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Base loader (no sampler) used for feature extraction and clustering
base_loader = DataLoader(train_dataset_base, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train: {len(train_dataset_base)} | Test: {len(test_dataset)}")

# ─── Model ────────────────────────────────────────────────────────────────────
class MobileNetWithProjection(nn.Module):
    def __init__(self, proj_dim=64):
        super().__init__()
        # Random init — no pretrained weights
        base = models.mobilenet_v3_small(weights=None)

        # Backbone: everything up to 1024-dim
        self.backbone = nn.Sequential(
            base.features,
            base.avgpool,
            nn.Flatten(),
            base.classifier[0],  # Linear(576, 1024)
            base.classifier[1],  # BatchNorm
            base.classifier[2],  # Hardswish
        )

        # Projection head: 1024 -> proj_dim
        self.projector = nn.Linear(1024, proj_dim)

        # Gaussian init — std=0.01 prevents exploding activations through deep layers
        # std=1 causes all features to collapse to one cluster after L2 norm
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0, std=0.01)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0, std=0.01)

    def forward(self, x):
        features = self.backbone(x)       # (B, 1024)
        return self.projector(features)   # (B, 64)


model = MobileNetWithProjection(proj_dim=PROJ_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,}  (full model)")

# ─── UnifLabelSampler (from DeepCluster) ──────────────────────────────────────
class UnifLabelSampler(Sampler):
    """Samples elements uniformly across pseudolabels.
    Args:
        N (int): size of returned iterator.
        images_lists: list of lists — for each cluster, list of image indexes
    """
    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        nmb_non_empty_clusters = sum(
            1 for i in range(len(self.images_lists)) if len(self.images_lists[i]) != 0
        )
        size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])
        for i in range(len(self.images_lists)):
            if len(self.images_lists[i]) == 0:
                continue
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res = np.concatenate((res, indexes))
        np.random.shuffle(res)
        res = list(res.astype('int'))
        if len(res) >= self.N:
            return res[:self.N]
        res += res[:(self.N - len(res))]
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)


# ─── Pseudo-label Dataset ─────────────────────────────────────────────────────
class PseudoLabelDataset(Dataset):
    """Wraps the base MNIST dataset with pseudo-labels from clustering."""
    def __init__(self, base_dataset, pseudo_labels):
        self.base_dataset = base_dataset
        self.pseudo_labels = pseudo_labels  # np.array of length N

    def __getitem__(self, index):
        img, _ = self.base_dataset[index]   # discard true label
        return img, int(self.pseudo_labels[index])

    def __len__(self):
        return len(self.base_dataset)


# ─── Differentiable Silhouette Loss ───────────────────────────────────────────
def silhouette_loss(embeddings, labels):
    """
    Differentiable (1 - silhouette) loss.
    Uses mean intra-class distance (a) and min mean inter-class distance (b).
    """
    device = embeddings.device
    n = embeddings.size(0)
    classes = labels.unique()

    diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)   # (N, N, D)
    dist = (diff ** 2).sum(dim=-1).clamp(min=1e-8).sqrt()      # (N, N)

    a = torch.zeros(n, device=device)
    b = torch.full((n,), float("inf"), device=device)

    for cls in classes:
        mask = (labels == cls)
        idx  = mask.nonzero(as_tuple=True)[0]

        if idx.numel() < 2:
            continue

        intra     = dist[idx][:, idx]
        intra_sum = intra.sum(dim=1) - intra.diag()
        a[idx]    = intra_sum / (idx.numel() - 1)

        other_idx = (~mask).nonzero(as_tuple=True)[0]
        inter     = dist[other_idx][:, idx].mean(dim=1)
        b[other_idx] = torch.minimum(b[other_idx], inter)

    b    = torch.where(torch.isinf(b), a, b)
    sil  = (b - a) / torch.maximum(a, b).clamp(min=1e-8)
    loss = (1 - sil).mean()
    return loss


# ─── MiniBatchKMeans Clustering ───────────────────────────────────────────────
def run_minibatch_kmeans(loader, model, n_clusters):
    """
    Extract features batch-by-batch and fit MiniBatchKMeans iteratively.
    Fresh init each epoch (like DeepCluster's random seed per epoch).
    Returns:
        pseudo_labels (np.array): cluster assignment per sample
        images_lists (list of list): for each cluster, list of image indexes
        all_features (np.array): L2-normalized embeddings for all samples
    """
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=3,
        batch_size=BATCH_SIZE,
        random_state=np.random.randint(1234),  # fresh random seed each epoch
        max_iter=100,
    )

    model.eval()
    # Pass 1: partial_fit batch by batch
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="  MiniBatchKMeans fitting", leave=False):
            emb = model(images.to(DEVICE)).cpu().numpy().astype('float32')
            emb = normalize(emb)
            kmeans.partial_fit(emb)

    # Pass 2: assign labels + collect features iteratively batch-by-batch
    n_total       = len(loader.dataset)
    pseudo_labels = np.zeros(n_total, dtype=np.int32)
    all_features  = np.zeros((n_total, PROJ_DIM), dtype='float32')
    images_lists  = [[] for _ in range(n_clusters)]
    sample_idx    = 0

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="  Assigning labels", leave=False):
            emb          = model(images.to(DEVICE)).cpu().numpy().astype('float32')
            emb          = normalize(emb)
            batch_labels = kmeans.predict(emb)
            batch_size   = len(batch_labels)

            pseudo_labels[sample_idx: sample_idx + batch_size] = batch_labels
            all_features[sample_idx: sample_idx + batch_size]  = emb
            for offset, cluster in enumerate(batch_labels):
                images_lists[cluster].append(sample_idx + offset)

            sample_idx += batch_size

    return pseudo_labels, images_lists, all_features


def hungarian_accuracy(true_labels, cluster_labels, n_classes=10):
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, c in zip(true_labels, cluster_labels):
        confusion[t, c] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    correct = confusion[row_ind, col_ind].sum()
    return correct / len(true_labels)


# ─── Training Loop ────────────────────────────────────────────────────────────
print(f"\nTraining for {EPOCHS} epochs (DeepCluster-style unsupervised)...\n")

for epoch in range(1, EPOCHS + 1):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch}/{EPOCHS}")

    # ── Step 1: Cluster fresh features with MiniBatchKMeans ──
    print("  Clustering features...")
    pseudo_labels, images_lists, train_emb_n = run_minibatch_kmeans(base_loader, model, NMB_CLUSTERS)

    cluster_sizes = [len(c) for c in images_lists]
    print(f"  Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
          f"empty={sum(1 for s in cluster_sizes if s == 0)}")

    # ── Step 2: Build pseudo-label dataset ──
    pseudo_dataset = PseudoLabelDataset(train_dataset_base, pseudo_labels)

    # ── Step 3: UnifLabelSampler for balanced cluster sampling ──
    sampler = UnifLabelSampler(len(pseudo_dataset), images_lists)

    train_loader = DataLoader(
        pseudo_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # ── Step 4: Train one epoch with silhouette loss ──
    model.train()
    epoch_loss = 0.0

    for images, pseudo_lbl in tqdm(train_loader, desc=f"  Training", unit="batch"):
        images     = images.to(DEVICE)
        pseudo_lbl = pseudo_lbl.to(DEVICE)

        optimizer.zero_grad()
        embeddings = model(images)                        # (B, 64)
        loss       = silhouette_loss(embeddings, pseudo_lbl)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    # ── Step 5: Silhouette score — reuse features already extracted in clustering ──
    # Guard against degenerate clustering (all samples collapsed to 1 cluster)
    n_unique = len(np.unique(pseudo_labels))
    if n_unique >= 2:
        sil_train = silhouette_score(train_emb_n, pseudo_labels, metric="euclidean")
    else:
        sil_train = float('nan')
        print(f"  Warning: only {n_unique} cluster(s) found, skipping silhouette")

    print(f"  Loss: {avg_loss:.4f} | Silhouette (pseudo): {sil_train:.4f}")

    # ── WandB logging (training metrics only) ──
    if logger:
        wandb.log({
            "epoch":              epoch,
            "train/loss":         avg_loss,
            "train/silhouette":   sil_train,
        })

# ─── Final Evaluation (true labels) ──────────────────────────────────────────
print(f"\n{'='*60}")
print("Running final evaluation with true labels...")

# train_emb_n already available from last epoch's clustering — no re-extraction needed
# Collect true train labels inline (they were discarded during training)
train_lbl = np.array([train_dataset_base[i][1] for i in range(len(train_dataset_base))])

# Extract test embeddings + true labels batch-by-batch (stacking unavoidable for eval metrics)
model.eval()
test_emb_list, test_lbl_list = [], []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Extracting test embeddings", leave=False):
        emb = model(images.to(DEVICE)).cpu().numpy().astype('float32')
        test_emb_list.append(normalize(emb))
        test_lbl_list.append(labels.numpy())
test_emb_n = np.concatenate(test_emb_list)
test_lbl   = np.concatenate(test_lbl_list)

# KNN Accuracy
knn = KNeighborsClassifier(n_neighbors=KNN_K, metric="euclidean", n_jobs=-1)
knn.fit(train_emb_n, train_lbl)
knn_acc = knn.score(test_emb_n, test_lbl) * 100

# Silhouette Score (true labels)
sil_test = silhouette_score(test_emb_n, test_lbl, metric="euclidean")

# NMI & Hungarian via KMeans on test embeddings
from sklearn.cluster import KMeans
kmeans_eval = KMeans(n_clusters=NMB_CLUSTERS, random_state=42, n_init=10)
kmeans_labels = kmeans_eval.fit_predict(test_emb_n)

nmi      = normalized_mutual_info_score(test_lbl, kmeans_labels)
hung_acc = hungarian_accuracy(test_lbl, kmeans_labels) * 100

print(f"\n{'='*55}")
print(f"  Final Results after {EPOCHS} epochs")
print(f"{'='*55}")
print(f"  KNN Accuracy     (k={KNN_K})  : {knn_acc:.2f}%")
print(f"  Hungarian Accuracy       : {hung_acc:.2f}%")
print(f"  NMI                      : {nmi:.4f}")
print(f"  Silhouette Score (64-dim): {sil_test:.4f}")
print(f"{'='*55}")

# Log final eval metrics to wandb
if logger:
    wandb.log({
        "eval/knn_accuracy":        knn_acc,
        "eval/hungarian_accuracy":  hung_acc,
        "eval/nmi":                 nmi,
        "eval/silhouette":          sil_test,
    })

# ─── UMAP Visualization (true labels, white background) ──────────────────────
print("\nFitting UMAP on 64-dim test embeddings...")
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="euclidean",
    random_state=42,
    verbose=True,
)
embedding_2d = reducer.fit_transform(test_emb_n)
print("UMAP done.")

sil_umap = silhouette_score(embedding_2d, test_lbl, metric="euclidean")
print(f"  Silhouette Score (2D UMAP): {sil_umap:.4f}")

if logger:
    wandb.log({"eval/silhouette_umap_2d": sil_umap})

NUM_CLASSES = 10
colors = cm.get_cmap("tab10", NUM_CLASSES)

fig, ax = plt.subplots(figsize=(12, 10))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for cls in range(NUM_CLASSES):
    mask = test_lbl == cls
    ax.scatter(
        embedding_2d[mask, 0],
        embedding_2d[mask, 1],
        c=[colors(cls)],
        label=str(cls),
        s=4,
        alpha=0.6,
        linewidths=0,
    )

legend = ax.legend(
    title="Digit",
    title_fontsize=11,
    fontsize=10,
    markerscale=3,
    loc="upper right",
    framealpha=0.3,
    facecolor="white",
    edgecolor="#444",
)

ax.set_title(
    f"UMAP of 64-dim Learned Embeddings on MNIST Test Set\n"
    f"KNN Acc: {knn_acc:.2f}%  |  Sil (64-dim): {sil_test:.4f}  |  Sil (2D UMAP): {sil_umap:.4f}",
    fontsize=13,
    color="black",
    pad=14,
)
ax.set_xlabel("UMAP-1", color="black", fontsize=11)
ax.set_ylabel("UMAP-2", color="black", fontsize=11)
ax.tick_params(colors="black")
for spine in ax.spines.values():
    spine.set_edgecolor("#cccccc")

plt.tight_layout()
umap_path = "mnist_umap_deepcluster.png"
umap_pdf  = "mnist_umap_deepcluster.pdf"
plt.savefig(umap_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.savefig(umap_pdf,  bbox_inches="tight", facecolor="white")
print(f"Saved → {umap_path}")
print(f"Saved → {umap_pdf}")

if logger:
    wandb.finish()
