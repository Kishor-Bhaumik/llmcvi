
import torch
from sklearn.cluster import MiniBatchKMeans
import torch.nn.functional as F
import faiss
import numpy as np


# def _compute_silhouette_loss(num_clusters, embeddings, goal=0.8):
#     # Detached clustering (no gradients through cluster assignments)
#     with torch.no_grad():
#         gpu_id = embeddings.device.index
#         normalized_embeddings_detached = F.normalize(embeddings.detach(), dim=1)
#         kmeans_data = normalized_embeddings_detached.cpu().numpy().astype(np.float32)
#         d = kmeans_data.shape[1]  # dimension   
#         #res = faiss.StandardGpuResources()
#         kmeans = faiss.Kmeans(d=d, k=num_clusters, niter=20, verbose=False, gpu=gpu_id)
#         kmeans.train(kmeans_data)
#         _, cluster_labels_np = kmeans.index.search(kmeans_data, 1)
#         cluster_labels = torch.tensor(cluster_labels_np.ravel(), device=embeddings.device)
#     # Normalize WITH gradients for silhouette computation
#     normalized_embeddings = F.normalize(embeddings, dim=1)
#     silhouette_score = get_fast_silhouette_score(normalized_embeddings, cluster_labels)
#     if torch.isnan(silhouette_score):
#         return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
#     silhouette_loss = abs(goal - silhouette_score)
#     return silhouette_loss


def _compute_silhouette_loss(num_clusters, embeddings, goal=0.8):
    # Detached clustering (no gradients through cluster assignments)
    with torch.no_grad():
        # Normalize and convert to numpy
        normalized_embeddings_detached = F.normalize(embeddings.detach(), dim=1)
        kmeans_data = normalized_embeddings_detached.cpu().numpy().astype(np.float32)
        kmeans = MiniBatchKMeans( n_clusters=num_clusters,max_iter=20, batch_size=min(1024, len(kmeans_data)),random_state=42,  n_init=3, reassignment_ratio=0.01)
        cluster_labels_np = kmeans.fit_predict(kmeans_data)
        cluster_labels = torch.tensor(cluster_labels_np, device=embeddings.device)
    # Normalize WITH gradients for silhouette computation
    normalized_embeddings = F.normalize(embeddings, dim=1)
    silhouette_score = get_fast_silhouette_score(normalized_embeddings, cluster_labels)
    if torch.isnan(silhouette_score):
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    
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
        a = dists_to_centroids[:, centroid_idx_for_current_label]
        other_centroids_mask = torch.ones(k_actual, dtype=torch.bool, device=device)
        other_centroids_mask[centroid_idx_for_current_label] = False

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