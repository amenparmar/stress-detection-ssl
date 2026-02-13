
import torch
import numpy as np

class SignalAugmentation:
    """
    Vectorized data augmentation for multimodal physiological signals.
    Accepts batches of shape (Batch, Channels, Time).
    """
    
    def __init__(self, noise_factor=0.03, scale_range=(0.9, 1.1), 
                 magnitude_warp_sigma=0.2):
        self.noise_factor = noise_factor
        self.scale_range = scale_range
        self.magnitude_warp_sigma = magnitude_warp_sigma
    
    def add_noise(self, x):
        """Add Gaussian noise to a batch of signals."""
        return x + torch.randn_like(x) * self.noise_factor
    
    def scale(self, x):
        """Random scaling of signal magnitude for each sample in batch."""
        batch_size = x.shape[0]
        scales = torch.empty(batch_size, 1, 1, device=x.device).uniform_(*self.scale_range)
        return x * scales
    
    def magnitude_warp(self, x):
        """Apply smooth magnitude warping to a batch of signals."""
        batch_size, channels, length = x.shape
        device = x.device
        
        # Create smooth random curves for each sample and channel
        num_knots = max(4, length // 20)
        knots = torch.randn(batch_size, channels, num_knots, device=device) * self.magnitude_warp_sigma + 1.0
        
        # Interpolate to full length
        warp_curves = torch.nn.functional.interpolate(
            knots,
            size=length,
            mode='linear',
            align_corners=False
        )
        
        return x * warp_curves
    
    def __call__(self, x, num_augmentations=2):
        """
        Apply random combination of augmentations to a batch.
        
        Args:
            x: Input signal batch (Batch, Channels, Time)
            num_augmentations: Number of augmentation types to apply
            
        Returns:
            Augmented signal batch
        """
        augmentations = [
            self.add_noise,
            self.scale,
            self.magnitude_warp,
        ]
        
        # Shuffle augmentations and apply first N
        # (Using a simpler approach for vectorization: apply with fixed probability or fixed count)
        indices = torch.randperm(len(augmentations))[:num_augmentations]
        
        augmented = x.clone()
        for idx in indices:
            augmented = augmentations[idx](augmented)
        
        return augmented
