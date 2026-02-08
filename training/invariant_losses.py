
import torch
import torch.nn as nn


def compute_mmd_loss(source_features, target_features, kernel='rbf', bandwidth=1.0):
    """
    Maximum Mean Discrepancy (MMD) loss.
    
    Minimizes distribution difference between source and target domains
    using kernel two-sample test.
    
    MMD² = E[k(x_s, x_s')] + E[k(x_t, x_t')] - 2*E[k(x_s, x_t)]
    
    Args:
        source_features: Features from source domain (Batch1, feature_dim)
        target_features: Features from target domain (Batch2, feature_dim)
        kernel: Kernel type ('rbf' or 'linear')
        bandwidth: RBF kernel bandwidth (sigma)
    
    Returns:
        MMD loss (scalar)
    """
    if source_features.size(0) == 0 or target_features.size(0) == 0:
        return torch.tensor(0.0, device=source_features.device)
    
    # Compute kernel matrices
    if kernel == 'rbf':
        xx = rbf_kernel(source_features, source_features, bandwidth)
        yy = rbf_kernel(target_features, target_features, bandwidth)
        xy = rbf_kernel(source_features, target_features, bandwidth)
    else:  # linear
        xx = linear_kernel(source_features, source_features)
        yy = linear_kernel(target_features, target_features)
        xy = linear_kernel(source_features, target_features)
    
    # MMD²
    mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    
    return mmd


def rbf_kernel(x1, x2, bandwidth=1.0):
    """
    Radial Basis Function (RBF) kernel.
    
    k(x, y) = exp(-||x - y||² / (2 * sigma²))
    
    Args:
        x1: First set of samples (N1, D)
        x2: Second set of samples (N2, D)
        bandwidth: Kernel bandwidth (sigma)
    
    Returns:
        Kernel matrix (N1, N2)
    """
    # Compute pairwise distances
    x1_norm = (x1 ** 2).sum(dim=1, keepdim=True)  # (N1, 1)
    x2_norm = (x2 ** 2).sum(dim=1, keepdim=True)  # (N2, 1)
    
    # ||x1 - x2||² = ||x1||² + ||x2||² - 2*x1·x2
    dist_sq = x1_norm + x2_norm.t() - 2 * torch.mm(x1, x2.t())
    
    # RBF kernel
    kernel_matrix = torch.exp(-dist_sq / (2 * bandwidth ** 2))
    
    return kernel_matrix


def linear_kernel(x1, x2):
    """
    Linear kernel: k(x, y) = x·y
    
    Args:
        x1: First set of samples (N1, D)
        x2: Second set of samples (N2, D)
    
    Returns:
        Kernel matrix (N1, N2)
    """
    return torch.mm(x1, x2.t())


def compute_coral_loss(source_features, target_features):
    """
    CORAL (Correlation Alignment) loss.
    
    Aligns second-order statistics (covariance matrices) between domains.
    
    CORAL = ||C_s - C_t||²_F / (4 * d²)
    
    where C_s, C_t are covariance matrices and ||·||_F is Frobenius norm.
    
    Args:
        source_features: Features from source domain (Batch1, feature_dim)
        target_features: Features from target domain (Batch2, feature_dim)
    
    Returns:
        CORAL loss (scalar)
    """
    if source_features.size(0) <= 1 or target_features.size(0) <= 1:
        return torch.tensor(0.0, device=source_features.device)
    
    d = source_features.size(1)  # feature dimension
    
    # Compute covariance matrices
    source_cov = compute_covariance(source_features)
    target_cov = compute_covariance(target_features)
    
    # Frobenius norm of difference
    diff = source_cov - target_cov
    loss = torch.sum(diff ** 2) / (4 * d * d)
    
    return loss


def compute_covariance(features):
    """
    Compute covariance matrix.
    
    Args:
        features: Feature matrix (N, D)
    
    Returns:
        Covariance matrix (D, D)
    """
    n = features.size(0)
    
    # Center features
    features_centered = features - features.mean(dim=0, keepdim=True)
    
    # Covariance
    cov = torch.mm(features_centered.t(), features_centered) / (n - 1)
    
    return cov


def compute_contrastive_subject_loss(features, labels, subject_ids, temperature=0.5):
    """
    Contrastive subject loss.
    
    Pulls together: same stress state, different subjects
    Pushes apart: different stress states
    
    This encourages subject-invariant representations.
    
    Args:
        features: Latent features (Batch, feature_dim)
        labels: Stress labels (Batch,)
        subject_ids: Subject IDs (Batch,)
        temperature: Temperature for contrastive loss
    
    Returns:
        Contrastive loss (scalar)
    """
    batch_size = features.size(0)
    
    if batch_size < 2:
        return torch.tensor(0.0, device=features.device)
    
    # Normalize features
    features_norm = nn.functional.normalize(features, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.mm(features_norm, features_norm.t()) / temperature
    
    # Create mask for positive pairs
    # Positive: same label, different subject
    label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # (Batch, Batch)
    subject_mask = subject_ids.unsqueeze(0) != subject_ids.unsqueeze(1)  # (Batch, Batch)
    positive_mask = label_mask & subject_mask
    
    # Create mask for negative pairs (different labels)
    negative_mask = ~label_mask
    
    # Remove diagonal (self-similarity)
    mask_eye = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    positive_mask = positive_mask & ~mask_eye
    negative_mask = negative_mask & ~mask_eye
    
    # Compute loss
    losses = []
    
    for i in range(batch_size):
        # Positive pairs for sample i
        pos_mask_i = positive_mask[i]
        
        if pos_mask_i.sum() > 0:
            # Similarities to positive samples
            pos_sim = similarity_matrix[i, pos_mask_i]
            
            # Similarities to all samples (for normalization)
            all_sim = similarity_matrix[i]
            all_sim = all_sim[~mask_eye[i]]  # Remove self
            
            # InfoNCE-style loss
            # log( exp(pos) / sum(exp(all)) )
            numerator = torch.exp(pos_sim).mean()
            denominator = torch.exp(all_sim).sum()
            
            loss_i = -torch.log(numerator / (denominator + 1e-8))
            losses.append(loss_i)
    
    if len(losses) == 0:
        return torch.tensor(0.0, device=features.device)
    
    return torch.stack(losses).mean()


class SubjectInvariantLoss(nn.Module):
    """
    Combined subject-invariant loss module.
    
    Combines:
    1. MMD loss - Minimize distribution difference
    2. CORAL loss - Align covariance matrices
    3. Contrastive subject loss - Subject-invariant representations
    """
    
    def __init__(self, mmd_weight=1.0, coral_weight=1.0, contrastive_weight=1.0,
                 mmd_kernel='rbf', mmd_bandwidth=1.0, contrastive_temp=0.5):
        """
        Args:
            mmd_weight: Weight for MMD loss
            coral_weight: Weight for CORAL loss
            contrastive_weight: Weight for contrastive loss
            mmd_kernel: Kernel for MMD ('rbf' or 'linear')
            mmd_bandwidth: Bandwidth for RBF kernel
            contrastive_temp: Temperature for contrastive loss
        """
        super(SubjectInvariantLoss, self).__init__()
        
        self.mmd_weight = mmd_weight
        self.coral_weight = coral_weight
        self.contrastive_weight = contrastive_weight
        self.mmd_kernel = mmd_kernel
        self.mmd_bandwidth = mmd_bandwidth
        self.contrastive_temp = contrastive_temp
    
    def forward(self, features, labels, subject_ids):
        """
        Compute combined subject-invariant loss.
        
        Args:
            features: Latent features (Batch, feature_dim)
            labels: Stress labels (Batch,)
            subject_ids: Subject IDs (Batch,)
        
        Returns:
            Tuple of (total_loss, mmd_loss, coral_loss, contrastive_loss)
        """
        batch_size = features.size(0)
        
        # Split batch into source and target domains (first half vs second half)
        mid = batch_size // 2
        
        if mid < 2:
            # Not enough samples for domain split
            mmd_loss = torch.tensor(0.0, device=features.device)
            coral_loss = torch.tensor(0.0, device=features.device)
        else:
            source_features = features[:mid]
            target_features = features[mid:]
            
            # MMD loss
            mmd_loss = compute_mmd_loss(
                source_features, target_features, 
                kernel=self.mmd_kernel, 
                bandwidth=self.mmd_bandwidth
            )
            
            # CORAL loss
            coral_loss = compute_coral_loss(source_features, target_features)
        
        # Contrastive subject loss
        contrastive_loss = compute_contrastive_subject_loss(
            features, labels, subject_ids, 
            temperature=self.contrastive_temp
        )
        
        # Total loss
        total_loss = (
            self.mmd_weight * mmd_loss +
            self.coral_weight * coral_loss +
            self.contrastive_weight * contrastive_loss
        )
        
        return total_loss, mmd_loss, coral_loss, contrastive_loss
