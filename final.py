"""
CARL-G Pretraining on 20 Newsgroups (all) + Linear Probe on TREC6
==================================================================
Phase 1 - Pretraining  : BERT-base (fine-tuned) backbone + SimCLR MLP projector on 20NG (subset="all")
                         Silhouette loss on projector output. Logs silhouette + erank.
Phase 2 - Cleanup      : MLP projector removed.
Phase 3 - Linear Probe : Frozen backbone + 6-way linear classifier.
                         Train on full TREC6 train set.
                         Evaluate on full TREC6 test set after every epoch.
                         Logs val loss, val acc, test acc.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import MiniBatchKMeans
from transformers import BertTokenizer, BertModel
import wandb

# ─────────────────────────────────────────────
# 1. Argument Parser
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="CARL-G BERT-base + TREC6 Linear Probe")

    # Data
    parser.add_argument("--max_len",            type=int,   default=128)
    parser.add_argument("--trec_train",         type=str,
                        default="/home/kbhau001/llm/loracvi/llmcvi/dkm_text/trec_train.label",
                        help="Path to TREC train label file")
    parser.add_argument("--trec_test",          type=str,
                        default="/home/kbhau001/llm/loracvi/llmcvi/dkm_text/trec_test.label",
                        help="Path to TREC test label file")

    # SimCLR projector
    parser.add_argument("--proj_hidden_dim",    type=int,   default=256)
    parser.add_argument("--proj_out_dim",       type=int,   default=128)

    # CARL-G Pretraining
    parser.add_argument("--pretrain_epochs",    type=int,   default=50)
    parser.add_argument("--batch_size",         type=int,   default=512)
    parser.add_argument("--lr",                 type=float, default=1e-5)
    parser.add_argument("--weight_decay",       type=float, default=1e-4)
    parser.add_argument("--sil_goal",           type=float, default=0.5)
    parser.add_argument("--k",                  type=int,   default=20,
                        help="Number of clusters for CARL-G")

    # Linear Probe
    parser.add_argument("--probe_epochs",       type=int,   default=15)
    parser.add_argument("--probe_lr",           type=float, default=1e-2)
    parser.add_argument("--probe_weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_classes",        type=int,   default=6)

    # Misc
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--use_wandb",          action="store_false")
    parser.add_argument("--wandb_project",      type=str,   default="carlg_bert")
    parser.add_argument("--wandb_run_name",     type=str,   default="ngToTrec_bert")

    return parser.parse_args()

# ─────────────────────────────────────────────
# 2. TREC6 Loader
# ─────────────────────────────────────────────

COARSE_MAP     = {"ABBR": 0, "ENTY": 1, "DESC": 2, "HUM": 3, "LOC": 4, "NUM": 5}
CATEGORY_NAMES = ["Abbreviation", "Entity", "Description", "Human", "Location", "Numeric"]


def parse_trec_file(path):
    texts, labels = [], []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            coarse = line.split(":")[0]
            text   = " ".join(line.split(" ")[1:])
            texts.append(text)
            labels.append(COARSE_MAP[coarse])
    return texts, labels

# ─────────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len, labels=None):
        self.texts     = texts
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.labels    = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ─────────────────────────────────────────────
# 4. Backbone: BERT-base (fine-tuned), CLS pooling
# ─────────────────────────────────────────────

BERT_HIDDEN_DIM = 768  # bert-base-uncased hidden size (hardcoded)

class BERTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids      = input_ids,
            attention_mask = attention_mask,
        )
        # CLS token representation: outputs.last_hidden_state[:, 0, :]
        return outputs.last_hidden_state[:, 0, :]   # (B, 768)

# ─────────────────────────────────────────────
# 5. SimCLR-style MLP Projector
# ─────────────────────────────────────────────

class MLPProjector(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,     hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

# ─────────────────────────────────────────────
# 6. CARL-G Metrics
# ─────────────────────────────────────────────

def compute_effective_rank(embeddings_np):
    X = embeddings_np - embeddings_np.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    s = s[s > 1e-10]
    if len(s) == 0:
        return 1.0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def silhouette_loss(embeddings, centroids, goal):
    K = centroids.shape[0]
    if K < 2:
        return embeddings.sum() * 0, torch.tensor(0.0)

    emb_n  = F.normalize(embeddings, p=2, dim=1)
    cent_n = F.normalize(centroids,  p=2, dim=1).detach()
    dist   = torch.cdist(emb_n, cent_n, p=2)

    with torch.no_grad():
        labels = dist.argmin(dim=1)

    a = dist[torch.arange(len(embeddings)), labels]
    dist_other = dist.clone()
    dist_other[torch.arange(len(embeddings)), labels] = float('inf')
    b = dist_other.min(dim=1).values

    s         = (b - a) / torch.max(a, b).clamp(min=1e-8)
    sil_score = s.mean()
    return torch.abs(goal - sil_score), sil_score.detach()

# ─────────────────────────────────────────────
# 7. LR Scheduler (Warmup + Cosine)
# ─────────────────────────────────────────────

def get_lr_scheduler(optimizer, total_epochs):
    warmup_steps = total_epochs // 2

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_epochs - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ─────────────────────────────────────────────
# 8. Embedding Extraction
# ─────────────────────────────────────────────

@torch.no_grad()
def get_all_embeddings(backbone, projector, loader, device):
    backbone.eval()
    #if projector is not None:
    projector.eval()

    all_backbone, all_proj = [], []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        emb  = backbone(ids, mask)
        #all_backbone.append(emb.cpu().numpy())
        #if projector is not None:
        all_proj.append(projector(emb).cpu().numpy())

    #backbone_np = np.concatenate(all_backbone, axis=0)
    proj_np     = np.concatenate(all_proj, axis=0) #if projector is not None else None
    return  proj_np

# ─────────────────────────────────────────────
# 9. Phase 1 – Pretraining
# ─────────────────────────────────────────────

def pretrain(backbone, projector, loader, args, device):
    params    = list(backbone.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_lr_scheduler(optimizer, args.pretrain_epochs)

    current_centroids = 'k-means++'

    for epoch in range(args.pretrain_epochs):

        # Clustering on projector output
        proj_embs = get_all_embeddings(backbone, projector, loader, device)

        # Erank on backbone embeddings (CLS)
        # backbone_embs, _ = get_all_embeddings(backbone, None, loader, device)
        erank = compute_effective_rank(proj_embs)

        km = MiniBatchKMeans(
            n_clusters   = args.k,
            init         = current_centroids,
            n_init       = 1 if not isinstance(current_centroids, str) else 10,
            batch_size   = args.batch_size,
            random_state = args.seed,
        )
        km.fit(proj_embs)
        current_centroids = km.cluster_centers_
        centroids_tensor  = torch.tensor(current_centroids, dtype=torch.float32, device=device)

        # Full-batch gradient accumulation
        backbone.train()
        projector.train()
        optimizer.zero_grad()
        n_batches  = len(loader)
        epoch_loss = 0.0
        epoch_sil  = 0.0

        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            emb  = backbone(ids, mask)
            proj = projector(emb)
            loss, sil = silhouette_loss(proj, centroids_tensor, args.sil_goal)
            (loss / n_batches).backward()
            epoch_loss += loss.item()
            epoch_sil  += sil.item()

        optimizer.step()
        scheduler.step()

        avg_loss = epoch_loss / n_batches
        avg_sil  = epoch_sil  / n_batches

        print(f"[Pretrain] Ep {epoch:>2}/{args.pretrain_epochs} | "
              f"Loss: {avg_loss:.4f} | Sil: {avg_sil:.4f} | Erank: {erank:.2f}")

        if args.use_wandb:
            wandb.log({
                "pretrain/loss":       avg_loss,
                "pretrain/silhouette": avg_sil,
                "pretrain/erank":      erank,
                "pretrain/epoch":      epoch,
            })

# ─────────────────────────────────────────────
# 10. Phase 3 – Linear Probe
# ─────────────────────────────────────────────

def linear_probe(backbone, train_loader, test_loader, args, device):
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # Input dim is BERT_HIDDEN_DIM (768) — hardcoded
    classifier = nn.Linear(BERT_HIDDEN_DIM, args.num_classes).to(device)
    optimizer  = torch.optim.AdamW(
        classifier.parameters(),
        lr           = args.probe_lr,
        weight_decay = args.probe_weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.probe_epochs):

        # Train on TREC6 train set
        classifier.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0

        for batch in train_loader:
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.no_grad():
                emb = backbone(ids, mask)

            logits = classifier(emb)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            train_correct  += (logits.argmax(dim=1) == labels).sum().item()
            train_total    += labels.size(0)

        train_loss = train_loss_sum / train_total
        train_acc  = train_correct  / train_total

        # Evaluate on TREC6 test set
        classifier.eval()
        test_correct, test_total = 0, 0

        with torch.no_grad():
            for batch in test_loader:
                ids    = batch["input_ids"].to(device)
                mask   = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                emb    = backbone(ids, mask)
                logits = classifier(emb)
                test_correct += (logits.argmax(dim=1) == labels).sum().item()
                test_total   += labels.size(0)

        test_acc = test_correct / test_total

        print(f"[LinearProbe] Ep {epoch:>2}/{args.probe_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

        if args.use_wandb:
            wandb.log({
                "probe/train_loss": train_loss,
                "probe/train_acc":  train_acc,
                "probe/test_acc":   test_acc,
                "probe/epoch":      epoch,
            })

# ─────────────────────────────────────────────
# 11. Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load 20 Newsgroups (all) for pretraining ─────────────────────────
    print("Loading 20 Newsgroups (all)...")
    ng_data  = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    ng_texts = ng_data.data
    print(f"20NG samples: {len(ng_texts)}")

    # ── Load TREC6 ───────────────────────────────────────────────────────
    print("Loading TREC6...")
    trec_train_texts, trec_train_labels = parse_trec_file(args.trec_train)
    trec_test_texts,  trec_test_labels  = parse_trec_file(args.trec_test)
    print(f"TREC6 train: {len(trec_train_texts)} | test: {len(trec_test_texts)}")

    # ── Tokenizer ────────────────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    pretrain_dataset   = TextDataset(ng_texts,         tokenizer, args.max_len)
    trec_train_dataset = TextDataset(trec_train_texts, tokenizer, args.max_len, labels=trec_train_labels)
    trec_test_dataset  = TextDataset(trec_test_texts,  tokenizer, args.max_len, labels=trec_test_labels)

    pretrain_loader   = DataLoader(pretrain_dataset,   batch_size=args.batch_size,
                                   shuffle=True,  num_workers=4, pin_memory=True)
    trec_train_loader = DataLoader(trec_train_dataset, batch_size=args.batch_size,
                                   shuffle=True,  num_workers=4, pin_memory=True)
    trec_test_loader  = DataLoader(trec_test_dataset,  batch_size=args.batch_size,
                                   shuffle=False, num_workers=4, pin_memory=True)

    # ── WandB ─────────────────────────────────────────────────────────────
    if args.use_wandb:
        run_name = f"{args.wandb_run_name}_K{args.k}"
        wandb.init(
            project   = args.wandb_project,
            name      = run_name,
            config    = vars(args),
            save_code = True,
        )

    # ── Build backbone + projector ────────────────────────────────────────
    backbone = BERTBackbone().to(device)

    projector = MLPProjector(
        in_dim     = BERT_HIDDEN_DIM,   # 768
        hidden_dim = args.proj_hidden_dim,
        out_dim    = args.proj_out_dim,
    ).to(device)

    # ── Phase 1: Pretraining on 20NG ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 1: CARL-G Pretraining on 20NG (all)  |  K={args.k}")
    print(f"{'='*60}")
    pretrain(backbone, projector, pretrain_loader, args, device)

    # ── Phase 2: Drop projector ───────────────────────────────────────────
    print("\nPhase 2: Dropping MLP projector.")
    del projector

    # ── Phase 3: Linear Probe on TREC6 ───────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 3: Linear Probe on TREC6")
    print(f"{'='*60}")
    linear_probe(backbone, trec_train_loader, trec_test_loader, args, device)

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
