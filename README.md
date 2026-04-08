# llmcvi

Large Language Model training with Cluster Validity Indices (CVI)

## Cluster Validity Indices Implemented

This repository implements various cluster validity indices for improving LLM training:

### 1. Silhouette Coefficient
**Implementation Location:** `utils/fast_slt.py` and `utils/fast_slt_wo_kmeans.py`

The Silhouette coefficient measures how similar a point is to its own cluster compared to other clusters. Values range from -1 to 1, where:
- Higher values indicate better clustering
- Value near 1: Point is well-matched to its cluster
- Value near 0: Point is on the boundary between clusters
- Value near -1: Point might be assigned to the wrong cluster

**Usage in code:**
- With K-means clustering: `_compute_silhouette_loss()` in `utils/fast_slt.py`
- With ground truth labels: `wo_compute_silhouette_loss()` in `utils/fast_slt_wo_kmeans.py`

### 2. VRC (Variance Ratio Criterion / Calinski-Harabasz Index)
**Implementation Location:** `utils/fast_vrc.py`

The VRC measures the ratio of between-cluster dispersion to within-cluster dispersion. Formula:
```
VRC = (B / W) * ((n - k) / (k - 1))
```
where:
- B: Between-cluster dispersion (variance between cluster centroids and global centroid)
- W: Within-cluster dispersion (sum of variances within each cluster)
- n: Total number of samples
- k: Number of clusters

**Properties:**
- Higher VRC values indicate better clustering (well-separated, compact clusters)
- Typical range: 0 to several thousand (unbounded)
- Good clustering typically has VRC > 100

**Usage in code:**
- With K-means clustering: `_compute_vrc_loss()` in `utils/fast_vrc.py`
- With ground truth labels: `wo_compute_vrc_loss()` in `utils/fast_vrc.py`

## Training Scripts

The CVI metrics are integrated into the following training scripts:
- `cvi_lgt.py` - Base BERT classifier with CVI (currently uses Silhouette)
- `cvi_qlora.py` - QLoRA fine-tuning with CVI (currently uses Silhouette)
- `loracvi/cvi_lgt.py` - LoRA version
- `loracvi/cvi_qlora.py` - LoRA + QLoRA version

## Configuration

Configure CVI usage in `configs/defaults.yaml`:
- `USE_SILHOUETTE`: Enable/disable Silhouette coefficient (currently the only one integrated)
- `SILHOUETTE_WEIGHT`: Weight for Silhouette loss (λ parameter)
- `SLT_score_GOAL`: Target Silhouette score
- `SILHOUETTE_START_EPOCH`: Epoch to start applying CVI loss

**Note:** VRC implementation is available in `utils/fast_vrc.py` but not yet integrated into the training scripts. To use VRC, similar configuration parameters would need to be added.
