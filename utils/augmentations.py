
import numpy as np
import torch

class Augmentations:
    def __init__(self):
        pass

    def jitter(self, x, sigma=0.03):
        # x shape: (Channels, Time)
        noise = np.random.normal(loc=0, scale=sigma, size=x.shape)
        return x + noise

    def scaling(self, x, sigma=0.1):
        # x shape: (Channels, Time)
        factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], 1))
        return x * factor

    def permutation(self, x, max_segments=5, seg_mode="equal"):
        # x shape: (Channels, Time)
        orig_steps = np.arange(x.shape[1])
        
        num_segs = np.random.randint(1, max_segments)
        
        if num_segs > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1]-2, num_segs-1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs)
            
            perm = np.random.permutation(len(splits))
            warped_steps = np.concatenate([splits[i] for i in perm]).astype(int)
            return x[:, warped_steps]
        else:
            return x

    def __call__(self, x):
        # Apply random augmentations
        # x is a numpy array (Channels, Time)
        
        # Randomly apply jitter
        if np.random.rand() > 0.5:
            x = self.jitter(x)
            
        # Randomly apply scaling
        if np.random.rand() > 0.5:
            x = self.scaling(x)
            
        # Randomly apply permutation
        if np.random.rand() > 0.5:
             x = self.permutation(x)
             
        return x
