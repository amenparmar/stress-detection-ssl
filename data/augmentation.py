
import torch
import numpy as np

class SignalAugmentation:
    """
    Data augmentation for multimodal physiological signals.
    Applies transformations suitable for EDA, TEMP, and BVP signals.
    """
    
    def __init__(self, noise_factor=0.03, scale_range=(0.9, 1.1), 
                 time_warp_sigma=0.2, magnitude_warp_sigma=0.2):
        """
        Args:
            noise_factor: Standard deviation of Gaussian noise
            scale_range: Range for random scaling (min, max)
            time_warp_sigma: Sigma for time warping
            magnitude_warp_sigma: Sigma for magnitude warping
        """
        self.noise_factor = noise_factor
        self.scale_range = scale_range
        self.time_warp_sigma = time_warp_sigma
        self.magnitude_warp_sigma = magnitude_warp_sigma
    
    def add_noise(self, signal):
        """Add Gaussian noise to signal."""
        noise = torch.randn_like(signal, device=signal.device) * self.noise_factor
        return signal + noise
    
    def scale(self, signal):
        """Random scaling of signal magnitude."""
        scale_factor = torch.FloatTensor(1).uniform_(*self.scale_range).to(signal.device)
        return signal * scale_factor
    
    def time_warp(self, signal):
        """
        Time warping using interpolation.
        Slightly stretches or compresses the signal in time.
        """
        device = signal.device
        length = signal.shape[-1]
        # Create warped time indices
        orig_steps = torch.arange(length, dtype=torch.float32, device=device)
        warp = torch.randn(length, device=device) * self.time_warp_sigma
        warped_steps = orig_steps + warp
        warped_steps = torch.clamp(warped_steps, 0, length - 1)
        
        # Interpolate
        warped_signal = torch.nn.functional.interpolate(
            signal.unsqueeze(0), 
            size=length, 
            mode='linear', 
            align_corners=False
        ).squeeze(0)
        
        return warped_signal
    
    def magnitude_warp(self, signal):
        """
        Apply smooth magnitude warping.
        Multiplies signal by a smooth random curve.
        """
        device = signal.device  # Get device from input signal
        length = signal.shape[-1]
        # Create smooth random curve
        num_knots = max(4, length // 20)
        knots = torch.randn(num_knots, device=device) * self.magnitude_warp_sigma + 1.0
        
        # Interpolate to full length
        warp_curve = torch.nn.functional.interpolate(
            knots.unsqueeze(0).unsqueeze(0),
            size=length,
            mode='linear',
            align_corners=False
        ).squeeze()
        
        return signal * warp_curve
    
    def __call__(self, signal, num_augmentations=2):
        """
        Apply random combination of augmentations.
        
        Args:
            signal: Input signal tensor (Channels, Time)
            num_augmentations: Number of augmentation types to apply
            
        Returns:
            Augmented signal
        """
        augmentations = [
            self.add_noise,
            self.scale,
            self.magnitude_warp,
        ]
        
        # Randomly select augmentations
        selected = np.random.choice(augmentations, num_augmentations, replace=False)
        
        augmented = signal.clone()
        for aug_fn in selected:
            augmented = aug_fn(augmented)
        
        return augmented
