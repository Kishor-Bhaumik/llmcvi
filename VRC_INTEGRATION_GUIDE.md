# VRC Integration Guide

## Where is VRC Implemented?

**VRC (Variance Ratio Criterion / Calinski-Harabasz Index)** is implemented in:
```
utils/fast_vrc.py
```

This file contains three main functions:

1. **`_compute_vrc_loss(num_clusters, embeddings, goal=1000.0)`**
   - Uses K-means clustering to assign cluster labels
   - Computes VRC score based on cluster assignments
   - Returns: (vrc_score, vrc_loss)

2. **`wo_compute_vrc_loss(embeddings, labels, goal=1000.0)`**
   - Uses ground truth labels directly (no K-means)
   - Computes VRC score based on ground truth clusters
   - Returns: (vrc_score, vrc_loss)

3. **`compute_vrc_score(feats, cluster_labels)`**
   - Core VRC computation logic
   - Calculates between-cluster and within-cluster dispersion
   - Returns: vrc_score

## How to Integrate VRC into Training

To use VRC in your training scripts (e.g., `cvi_lgt.py` or `cvi_qlora.py`), follow these steps:

### Step 1: Import VRC functions

```python
# In cvi_lgt.py or cvi_qlora.py, add this import
from utils.fast_vrc import _compute_vrc_loss
```

### Step 2: Initialize VRC in BertClassifier.__init__()

```python
class BertClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Choose which CVI metric to use
        self.use_silhouette = self.hparams.get('USE_SILHOUETTE', False)
        self.use_vrc = self.hparams.get('USE_VRC', False)  # Add this
        
        # Configure CVI parameters
        self.cvi_weight = self.hparams.get('CVI_WEIGHT', 0.1)
        
        if self.use_silhouette:
            from utils.fast_slt import _compute_silhouette_loss
            self.cvi_loss_fn = _compute_silhouette_loss
            self.cvi_goal = self.hparams.get('SLT_score_GOAL', 0.8)
        elif self.use_vrc:
            from utils.fast_vrc import _compute_vrc_loss
            self.cvi_loss_fn = _compute_vrc_loss
            self.cvi_goal = self.hparams.get('VRC_GOAL', 1000.0)
        
        # ... rest of initialization
```

### Step 3: Use VRC in training_step()

```python
def training_step(self, batch, batch_idx):
    input_ids, attention_mask, labels = batch
    logits, embeddings = self(input_ids, attention_mask)
    
    # Main classification loss
    classification_loss = self.loss_fn(logits, labels)
    total_loss = classification_loss
    self.num_clusters = len(torch.unique(labels))
    
    # Add CVI loss if enabled and after warmup
    if (self.use_silhouette or self.use_vrc) and \
       self.current_epoch >= self.hparams.get('CVI_START_EPOCH', 0):
        cvi_score, cvi_loss = self.cvi_loss_fn(
            self.num_clusters, 
            embeddings,
            goal=self.cvi_goal
        )
        
        if not torch.isnan(cvi_loss):
            total_loss = classification_loss + self.cvi_weight * cvi_loss
            
            # Log appropriate metric name
            metric_name = 'vrc_score' if self.use_vrc else 'silhouette_score'
            loss_name = 'vrc_loss' if self.use_vrc else 'silhouette_loss'
            
            self.log(metric_name, cvi_score, on_step=True, on_epoch=True)
            self.log(loss_name, cvi_loss, on_step=True, on_epoch=True)
    
    # ... rest of training step
```

### Step 4: Update Configuration (configs/defaults.yaml)

```yaml
# Choose which CVI metric to use
USE_SILHOUETTE: False  # Set to True to use Silhouette
USE_VRC: True          # Set to True to use VRC

# VRC-specific parameters
VRC_GOAL: 1000.0       # Target VRC score (higher is better)
                       # Typical good clustering: VRC > 100
                       # Excellent clustering: VRC > 1000

# CVI general parameters
CVI_WEIGHT: 1.0        # λ parameter (weight for CVI loss)
CVI_START_EPOCH: 0     # When to start applying CVI loss
```

### Step 5: Run Training

```bash
# Train with VRC
python cvi_lgt.py USE_VRC=True VRC_GOAL=1000.0 RUN_NAME='with_vrc'

# Train without CVI
python cvi_lgt.py USE_VRC=False RUN_NAME='baseline'

# Compare VRC vs Silhouette
python cvi_lgt.py USE_VRC=True RUN_NAME='vrc_experiment'
python cvi_lgt.py USE_SILHOUETTE=True RUN_NAME='silhouette_experiment'
```

## Understanding VRC Scores

**VRC Formula:**
```
VRC = (B / W) * ((n - k) / (k - 1))
```

Where:
- **B**: Between-cluster dispersion (variance between centroids)
- **W**: Within-cluster dispersion (variance within clusters)
- **n**: Total number of samples
- **k**: Number of clusters

**Interpretation:**
- Higher VRC = Better clustering
- VRC > 100: Good clustering (compact, well-separated)
- VRC > 1000: Excellent clustering
- VRC < 10: Poor clustering (overlapping clusters)

**Comparison with Silhouette:**

| Metric | Range | Higher is Better? | Typical Good Value |
|--------|-------|-------------------|-------------------|
| Silhouette | -1 to 1 | Yes | > 0.5 |
| VRC | 0 to ∞ | Yes | > 100 |

**When to use VRC vs Silhouette:**
- **VRC**: Faster to compute, better for large datasets, sensitive to cluster compactness
- **Silhouette**: More interpretable, better for comparing different k values, bounded range

## Example Use Cases

### 1. Few-shot Learning with VRC
```bash
python cvi_qlora.py \
    DEVICE=0 \
    USE_VRC=True \
    VRC_GOAL=500.0 \
    "+few_shots={0:50, 1:50, 2:50, 3:50}" \
    RUN_NAME='fewshot50_vrc'
```

### 2. Fine-tuning with VRC
```bash
python cvi_lgt.py \
    USE_VRC=True \
    VRC_GOAL=1000.0 \
    CVI_WEIGHT=0.5 \
    TRAIN_SUBSET_RATIO=0.1 \
    RUN_NAME='vrc_experiment'
```

### 3. Hyperparameter Search
```bash
# Try different VRC goals
for goal in 100 500 1000 2000; do
    python cvi_lgt.py \
        USE_VRC=True \
        VRC_GOAL=$goal \
        RUN_NAME="vrc_goal_${goal}"
done
```

## Monitoring VRC During Training

When using WandB logging, VRC metrics will be logged as:
- `vrc_score`: The computed VRC value (higher is better)
- `vrc_loss`: Distance from goal |goal - vrc_score|

You can monitor these metrics to:
1. Verify that clusters are becoming more separated during training
2. Detect when clustering quality plateaus
3. Compare VRC values across different experiments
