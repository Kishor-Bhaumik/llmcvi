import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import umap
import wandb

# ──────────────────────────────────────────────────────────────────────────────
# Argument Parser
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="DKM-style BERT Fine-tuning on 20 Newsgroups")
parser.add_argument("--model_name",  type=str, default="bert-base-uncased")
parser.add_argument("--batch_size",  type=int, default=512)
parser.add_argument("--max_length",  type=int, default=128)
parser.add_argument("--n_clusters",  type=int, default=40)
parser.add_argument("--epochs",      type=int, default=10)
parser.add_argument("--lr",          type=float, default=0.00001)
parser.add_argument("--device",      type=int, default=1)
parser.add_argument("--no_wandb",    action="store_true")
parser.add_argument("--use_ramp_up", action="store_true")
#add seed
parser.add_argument("--seed",       type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()

#set pytorch seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
USE_WANDB       = not args.no_wandb
BATCH_SIZE      = args.batch_size
MAX_LENGTH      = args.max_length
N_CLUSTERS      = args.n_clusters
EPOCHS          = args.epochs
LR              = args.lr
DEVICE          = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
SOFT_ASSIGNMENT = True
ALPHA           = 50
LAMBDA          = 0.64
GOAL            = 0.5

WANDB_PROJECT   = "dkmpSlt-bert-20ng-finetune"
WANDB_RUN_NAME  = f"BERT-Sil-{args.model_name.split('/')[-1]}-LR{LR}"

def ramp_up(current_epoch, max_epochs):
    p = float(current_epoch) / max_epochs
    return 2. / (1. + np.exp(-10 * p)) - 1

# ──────────────────────────────────────────────────────────────────────────────
# WandB Init
# ──────────────────────────────────────────────────────────────────────────────
if USE_WANDB:
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config=vars(args))

# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────
print("Loading 20 Newsgroups...")
newsgroups_train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"), random_state=42)
newsgroups_test  = fetch_20newsgroups(subset="test",  remove=("headers", "footers", "quotes"), random_state=42)

class NewsgroupsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts, self.labels, self.tokenizer, self.max_length = texts, labels, tokenizer, max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids": encoding["input_ids"].squeeze(0), "attention_mask": encoding["attention_mask"].squeeze(0), "label": torch.tensor(self.labels[idx], dtype=torch.long)}

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
def data_loader(groups, shuffle=False):
    return DataLoader(NewsgroupsDataset(groups.data, groups.target, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=4)
train_loader = data_loader(newsgroups_train, shuffle=True)
test_loader  = data_loader(newsgroups_test)

# ──────────────────────────────────────────────────────────────────────────────
# Model & Centroid Init
# ──────────────────────────────────────────────────────────────────────────────
bert_model = AutoModel.from_pretrained(args.model_name).to(DEVICE)

def extract_embeddings(loader, model, device):
    model.eval()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting"):
            outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            all_emb.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
            all_lbl.append(batch["label"].numpy())
    return np.concatenate(all_emb), np.concatenate(all_lbl)

print("Initializing centroids via KMeans on pretrained BERT...")
train_emb_init, _ = extract_embeddings(train_loader, bert_model, DEVICE)
kmeans = KMeans(n_clusters=N_CLUSTERS, init="k-means++", n_init=10, random_state=42).fit(train_emb_init)
centroids = nn.Parameter(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=DEVICE))

# ──────────────────────────────────────────────────────────────────────────────
# Loss & Metrics
# ──────────────────────────────────────────────────────────────────────────────
def simplified_silhouette_loss(embeddings, centroids, goal=0.3, soft=True, alpha=1.0):
    B = len(embeddings)
    arange = torch.arange(B, device=embeddings.device)
    dist = torch.cdist(embeddings, centroids, p=2)
    with torch.no_grad(): labels = dist.argmin(dim=1)

    if soft:
        min_dist = dist.min(dim=1, keepdim=True).values
        exp = torch.exp(-alpha * (dist - min_dist))
        G_k = exp / exp.sum(dim=1, keepdim=True)
        a = (G_k * dist).sum(dim=1)
    else:
        a = dist[arange, labels]

    dist_masked = dist.clone()
    dist_masked[arange, labels] = float('inf')
    b = dist_masked.min(dim=1).values
    sil = (b - a) / torch.max(a, b).clamp(min=1e-8)
    return goal - sil.mean(), sil.mean().item()

def hungarian_accuracy(true_labels, cluster_labels, n_classes, n_clusters):
    size = max(n_classes, n_clusters)
    confusion = np.zeros((size, size), dtype=np.int64)
    for t, c in zip(true_labels, cluster_labels): confusion[t % size, c % size] += 1
    row, col = linear_sum_assignment(-confusion)
    return confusion[row, col].sum() / len(true_labels)

# ──────────────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(list(bert_model.parameters()) + [centroids], lr=LR)

for epoch in range(1, EPOCHS + 1):
    bert_model.train()
    total_loss, total_sil = 0, 0
    curr_lambda = ramp_up(epoch, EPOCHS) if args.use_ramp_up else LAMBDA

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        optimizer.zero_grad()
        emb = bert_model(input_ids=batch["input_ids"].to(DEVICE), attention_mask=batch["attention_mask"].to(DEVICE)).last_hidden_state[:, 0, :]
        loss, s_score = simplified_silhouette_loss(emb, centroids, goal=GOAL, soft=SOFT_ASSIGNMENT, alpha=ALPHA)
        (curr_lambda * loss).backward()
        optimizer.step()
        total_loss += loss.item(); total_sil += s_score

    # Evaluation
    test_emb, test_lbl = extract_embeddings(test_loader, bert_model, DEVICE)
    test_n = normalize(test_emb)
    km_test = KMeans(n_clusters=N_CLUSTERS, n_init=10).fit(test_n)
    
    nmi = normalized_mutual_info_score(test_lbl, km_test.labels_)
    h_acc = hungarian_accuracy(test_lbl, km_test.labels_, len(np.unique(test_lbl)), N_CLUSTERS) * 100
    sil = silhouette_score(test_n, test_lbl)

    print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | NMI: {nmi:.4f} | Hung: {h_acc:.2f}% | Sil: {sil:.4f}")
    if USE_WANDB:
        wandb.log({"epoch": epoch, "loss": total_loss/len(train_loader), "nmi": nmi, "hung_acc": h_acc, "silhouette": sil})

if USE_WANDB: wandb.finish()
