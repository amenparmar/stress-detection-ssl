
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


class TrajectoryAnalyzer(nn.Module):
    """
    Analyzes latent space trajectories for continuous stress monitoring.
    
    Instead of classifying each window independently, this module:
    1. Extracts a baseline (neutral) representation per subject
    2. Measures deviation from baseline over time
    3. Applies temporal smoothing
    4. Predicts stress based on trajectory patterns
    
    This enables:
    - Continuous stress scores (not just discrete classes)
    - Personalized baselines per subject
    - Temporal consistency in predictions
    - Detection of stress onset/offset dynamics
    """
    
    def __init__(self, feature_dim=256, num_classes=3, smoothing_window=5):
        """
        Args:
            feature_dim: Dimension of latent features
            num_classes: Number of stress classes (0=baseline, 1=amusement, 2=stress)
            smoothing_window: Window size for temporal smoothing (in samples)
        """
        super(TrajectoryAnalyzer, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.smoothing_window = smoothing_window
        
        # Learned deviation thresholds per class
        self.deviation_classifier = nn.Sequential(
            nn.Linear(1, 32),  # Input: deviation magnitude
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
        # Subject-specific baselines (populated during training/testing)
        self.subject_baselines = {}
        
        # Temporal smoothing buffer
        self.deviation_history = []
    
    def compute_baseline(self, features, labels, subject_ids):
        """
        Compute baseline (neutral state) representation for each subject.
        
        Uses average of all Class 0 (baseline) samples per subject.
        
        Args:
            features: Latent features (N, feature_dim)
            labels: Stress labels (N,)
            subject_ids: Subject IDs (N,)
        
        Returns:
            Dictionary: {subject_id: baseline_feature_vector}
        """
        baselines = {}
        
        unique_subjects = torch.unique(subject_ids)
        
        for subject_id in unique_subjects:
            subject_id = subject_id.item()
            
            # Get all samples for this subject with label 0 (baseline)
            mask = (subject_ids == subject_id) & (labels == 0)
            
            if mask.sum() > 0:
                baseline_features = features[mask]
                # Average baseline features
                baseline = baseline_features.mean(dim=0)
                baselines[subject_id] = baseline
            else:
                # Fallback: use all samples from this subject
                mask = (subject_ids == subject_id)
                if mask.sum() > 0:
                    baselines[subject_id] = features[mask].mean(dim=0)
                else:
                    # Last resort: zero vector
                    baselines[subject_id] = torch.zeros(self.feature_dim, device=features.device)
        
        return baselines
    
    def compute_deviation(self, features, subject_ids):
        """
        Compute deviation from baseline for each sample.
        
        Args:
            features: Latent features (Batch, feature_dim)
            subject_ids: Subject IDs (Batch,)
        
        Returns:
            Deviation magnitudes (Batch,)
        """
        deviations = []
        
        for i in range(features.size(0)):
            subject_id = subject_ids[i].item()
            
            # Get baseline for this subject
            if subject_id in self.subject_baselines:
                baseline = self.subject_baselines[subject_id]
                # Ensure baseline is on the same device as features
                baseline = baseline.to(features.device)
                
                # Compute L2 deviation from baseline
                deviation = torch.norm(features[i] - baseline, p=2)
            else:
                # No baseline yet, use zero deviation
                deviation = torch.tensor(0.0, device=features.device)
            deviations.append(deviation)
        
        return torch.stack(deviations)
    
    def apply_temporal_smoothing(self, deviations):
        """
        Apply moving average smoothing to deviation trajectory.
        
        Args:
            deviations: Deviation values (Batch,) or (Sequence,)
        
        Returns:
            Smoothed deviations
        """
        if isinstance(deviations, torch.Tensor):
            deviations = deviations.cpu().numpy()
        
        smoothed = np.convolve(
            deviations, 
            np.ones(self.smoothing_window) / self.smoothing_window, 
            mode='same'
        )
        
        return torch.tensor(smoothed, dtype=torch.float32)
    
    def predict_from_deviation(self, deviations):
        """
        Predict stress class from deviation magnitude.
        
        Args:
            deviations: Deviation magnitudes (Batch,)
        
        Returns:
            Class logits (Batch, num_classes)
        """
        # Reshape for classifier
        deviations = deviations.unsqueeze(1)  # (Batch, 1)
        logits = self.deviation_classifier(deviations)
        return logits
    
    def forward(self, features, subject_ids, apply_smoothing=True):
        """
        Forward pass: compute deviations and predict stress.
        
        Args:
            features: Latent features (Batch, feature_dim)
            subject_ids: Subject IDs (Batch,)
            apply_smoothing: Whether to apply temporal smoothing
        
        Returns:
            Tuple of (class_logits, deviations, continuous_scores)
        """
        # Compute deviation from baseline
        deviations = self.compute_deviation(features, subject_ids)
        
        # Apply temporal smoothing if enabled
        if apply_smoothing:
            deviations = self.apply_temporal_smoothing(deviations)
            deviations = deviations.to(features.device)
        
        # Predict class from deviation
        logits = self.predict_from_deviation(deviations)
        
        # Continuous stress score (normalized deviation)
        max_deviation = deviations.max() if deviations.max() > 0 else 1.0
        continuous_scores = deviations / max_deviation
        
        return logits, deviations, continuous_scores


def extract_subject_baselines(encoder, data_loader, device):
    """
    Extract baseline representations for all subjects from data.
    
    Args:
        encoder: Trained encoder model
        data_loader: Data loader (must return (data, labels, subject_ids))
        device: Computation device
    
    Returns:
        Dictionary: {subject_id: baseline_feature_vector}
    """
    encoder.eval()
    
    all_features = []
    all_labels = []
    all_subject_ids = []
    
    with torch.no_grad():
        for batch_data in data_loader:
            if len(batch_data) == 3:
                data, labels, subject_ids = batch_data
            else:
                data, labels = batch_data
                subject_ids = torch.zeros(data.size(0), dtype=torch.long)
            
            data = data.to(device)
            
            # Extract features
            features = encoder(data)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
            all_subject_ids.append(subject_ids)
    
    # Concatenate all batches
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_subject_ids = torch.cat(all_subject_ids, dim=0)
    
    # Compute baselines
    trajectory_analyzer = TrajectoryAnalyzer(feature_dim=all_features.size(1))
    baselines = trajectory_analyzer.compute_baseline(all_features, all_labels, all_subject_ids)
    
    return baselines


def compute_trajectory_consistency_loss(features_seq, temperature=0.1):
    """
    Temporal consistency loss: encourage smooth trajectories in latent space.
    
    Penalizes large jumps between consecutive time steps.
    
    Args:
        features_seq: Sequential features (Sequence, Batch, feature_dim)
        temperature: Smoothness coefficient
    
    Returns:
        Consistency loss (scalar)
    """
    if features_seq.size(0) < 2:
        return torch.tensor(0.0, device=features_seq.device)
    
    # Compute differences between consecutive steps
    diffs = features_seq[1:] - features_seq[:-1]  # (Sequence-1, Batch, feature_dim)
    
    # L2 norm of differences
    diff_norms = torch.norm(diffs, p=2, dim=2)  # (Sequence-1, Batch)
    
    # Mean squared difference
    consistency_loss = (diff_norms ** 2).mean() / temperature
    
    return consistency_loss
