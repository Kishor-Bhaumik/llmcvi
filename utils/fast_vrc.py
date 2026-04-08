
import torch
from sklearn.cluster import MiniBatchKMeans
import torch.nn.functional as F
import numpy as np


def _compute_vrc_loss(num_clusters, embeddings, goal=1000.0):
    """
    Compute VRC (Variance Ratio Criterion / Calinski-Harabasz Index) loss.
    
    VRC = (B / W) * ((n - k) / (k - 1))
    where:
    - B: Between-cluster dispersion (variance between cluster centroids and global centroid)
    - W: Within-cluster dispersion (sum of variances within each cluster)
    - n: Total number of samples
    - k: Number of clusters
    
    Higher VRC values indicate better clustering (well-separated, compact clusters).
    
    Args:
        num_clusters: Number of clusters to use
        embeddings: Tensor of shape (batch_size, embedding_dim)
        goal: Target VRC score (default: 1000.0, higher is better)
        
    Returns:
        vrc_score: The computed VRC score (with gradients)
        vrc_loss: Distance from goal (abs(goal - vrc_score))
    """
    # Detached clustering (no gradients through cluster assignments)
    with torch.no_grad():
        # Normalize and convert to numpy
        normalized_embeddings_detached = F.normalize(embeddings.detach(), dim=1)
        kmeans_data = normalized_embeddings_detached.cpu().numpy().astype(np.float32)
        
        kmeans = MiniBatchKMeans(
            n_clusters=num_clusters,
            max_iter=20, 
            batch_size=min(1024, len(kmeans_data)),
            random_state=123, 
            n_init=3, 
            reassignment_ratio=0.01
        )
        cluster_labels_np = kmeans.fit_predict(kmeans_data)
        
        # Relabel to ensure consecutive cluster IDs from 0 to n-1
        unique_labels = np.unique(cluster_labels_np)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        cluster_labels_np = np.array([label_mapping[label] for label in cluster_labels_np])
        
        # Verify we have the correct number of clusters
        actual_clusters = len(unique_labels)
        if actual_clusters != num_clusters:
            print(f"Warning: Expected {num_clusters} clusters, got {actual_clusters}")
        
        cluster_labels = torch.tensor(cluster_labels_np, device=embeddings.device)
    
    # Normalize WITH gradients for VRC computation
    normalized_embeddings = F.normalize(embeddings, dim=1)
    vrc_score = compute_vrc_score(normalized_embeddings, cluster_labels)
    
    if torch.isnan(vrc_score):
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True), torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    
    vrc_loss = torch.abs(goal - vrc_score)
    return vrc_score, vrc_loss


def compute_vrc_score(feats: torch.Tensor, cluster_labels: torch.Tensor):
    """
    Compute the VRC (Calinski-Harabasz) score for given features and cluster labels.
    
    Args:
        feats: Tensor of shape (n_samples, n_features) - normalized embeddings
        cluster_labels: Tensor of shape (n_samples,) - cluster assignments
        
    Returns:
        vrc_score: The VRC score (higher is better)
    """
    device = feats.device
    unique_cluster_labels = torch.unique(cluster_labels)
    
    n = feats.shape[0]  # Total number of samples
    k = len(unique_cluster_labels)  # Number of clusters
    
    # Need at least 2 clusters for VRC
    if k <= 1 or k >= n:
        return torch.tensor(float('nan'), device=device)
    
    # Compute global centroid (mean of all points)
    global_centroid = feats.mean(dim=0)
    
    # Compute between-cluster dispersion (B)
    B = torch.tensor(0.0, device=device, requires_grad=True)
    # Compute within-cluster dispersion (W)
    W = torch.tensor(0.0, device=device, requires_grad=True)
    
    for cluster_label in unique_cluster_labels:
        # Get points in this cluster
        cluster_mask = cluster_labels == cluster_label
        cluster_points = feats[cluster_mask]
        n_cluster = cluster_points.shape[0]
        
        if n_cluster == 0:
            continue
        
        # Compute cluster centroid
        cluster_centroid = cluster_points.mean(dim=0)
        
        # Add to between-cluster dispersion
        # B += n_cluster * ||cluster_centroid - global_centroid||^2
        centroid_diff = cluster_centroid - global_centroid
        B = B + n_cluster * torch.sum(centroid_diff ** 2)
        
        # Add to within-cluster dispersion
        # W += sum of ||point - cluster_centroid||^2 for all points in cluster
        point_diffs = cluster_points - cluster_centroid
        W = W + torch.sum(point_diffs ** 2)
    
    # Avoid division by zero
    if W < 1e-10:
        return torch.tensor(float('nan'), device=device)
    
    # Compute VRC = (B / W) * ((n - k) / (k - 1))
    vrc_score = (B / W) * ((n - k) / (k - 1))
    
    return vrc_score


def wo_compute_vrc_loss(embeddings, labels, goal=1000.0):
    """
    Compute VRC loss using ground truth labels (without K-means clustering).
    
    Args:
        embeddings: Tensor of shape (batch_size, embedding_dim)
        labels: Ground truth labels
        goal: Target VRC score (default: 1000.0)
        
    Returns:
        vrc_score: The computed VRC score (with gradients)
        vrc_loss: Distance from goal (abs(goal - vrc_score))
    """
    cluster_labels = labels.to(embeddings.device).long().detach()  # use ground truth labels
    normalized_embeddings = F.normalize(embeddings, dim=1)  # keep grads
    vrc_score = compute_vrc_score(normalized_embeddings, cluster_labels)
    
    if torch.isnan(vrc_score):
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True), torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    
    vrc_loss = torch.abs(goal - vrc_score)
    return vrc_score, vrc_loss
