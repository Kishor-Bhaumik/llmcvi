
import torch
from sklearn.cluster import MiniBatchKMeans
import torch.nn.functional as F


def _compute_silhouette_loss(num_clusters, embeddings, goal=0.8):
    # Detached clustering (no gradients through cluster assignments)
    with torch.no_grad():
        normalized_embeddings = F.normalize(embeddings, dim=1)
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, n_init=1)
        cluster_labels = torch.tensor(
            kmeans.fit_predict(normalized_embeddings.cpu().numpy()), 
            device=embeddings.device)
    normalized_embeddings = F.normalize(embeddings, dim=1)
    silhouette_score = get_fast_silhouette_score(normalized_embeddings, cluster_labels)
    silhouette_loss = abs(goal - silhouette_score)
    
    return silhouette_loss

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
