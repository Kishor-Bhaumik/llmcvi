
import torch
from sklearn.cluster import MiniBatchKMeans
import torch.nn.functional as F
#import faiss
import numpy as np

def wo_compute_silhouette_loss(embeddings, labels, goal=0.8):
    cluster_labels = labels.to(embeddings.device).long().detach()   # use your labels
    normalized_embeddings = F.normalize(embeddings, dim=1)          # keep grads
    silhouette_score = get_fast_silhouette_score(normalized_embeddings, cluster_labels)
    silhouette_loss = abs(goal - silhouette_score)
    return silhouette_score,silhouette_loss


def get_fast_silhouette_score(feats: torch.Tensor, cluster_labels: torch.Tensor):
    device = feats.device
    unique_cluster_labels = torch.unique(cluster_labels)

    k = len(unique_cluster_labels)
    if k <= 1 or k >= len(feats):
        return torch.tensor(float('nan'), device=device)
    scores = []
    cluster_centroids_list = []

    for lbl_idx, current_lbl_val in enumerate(unique_cluster_labels):
        curr_cluster_feats = feats[cluster_labels == current_lbl_val]
        if len(curr_cluster_feats) == 0:
            continue
        cluster_centroids_list.append(curr_cluster_feats.mean(dim=0))

    if not cluster_centroids_list:
        return torch.tensor(float('nan'), device=device)

    cluster_centroids = torch.vstack(cluster_centroids_list)
    k_actual = cluster_centroids.shape[0]
    if k_actual <= 1:
        return torch.tensor(float('nan'), device=device)

    for lbl_idx, current_lbl_val in enumerate(unique_cluster_labels):
        centroid_idx_for_current_label = current_lbl_val.item()
        curr_cluster_feats = feats[cluster_labels == current_lbl_val]
        if len(curr_cluster_feats) == 0:
            continue
        dists_to_centroids = torch.cdist(curr_cluster_feats, cluster_centroids)
        #dists_to_centroids = 1 - (curr_cluster_feats @ cluster_centroids.T)
        a = dists_to_centroids[:, lbl_idx]  # use lbl_idx, not label value
        other_centroids_mask = torch.ones(k_actual, dtype=torch.bool, device=device)
        other_centroids_mask[lbl_idx] = False

        if torch.sum(other_centroids_mask) == 0:
             b = torch.zeros_like(a)
        else:
            b = torch.min(dists_to_centroids[:, other_centroids_mask], dim=1).values

        sils = (b - a) / (torch.max(a, b) + 1e-9)
        scores.append(sils)

    if not scores:
        return torch.tensor(float('nan'), device=device)

    scores_cat = torch.cat(scores, dim=0)
    return torch.mean(scores_cat)