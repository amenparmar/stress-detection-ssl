
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.augmentation import SignalAugmentation

def train_simclr(train_loader, encoder, projection_head, optimizer, scheduler, criterion, epochs, device):
    """
    Self-supervised contrastive learning training loop (SimCLR).
    Applies robust data augmentation to create positive pairs.
    """
    encoder.train()
    projection_head.train()
    
    # Initialize augmentation with optimized parameters
    augmentor = SignalAugmentation(
        noise_factor=0.05,
        scale_range=(0.85, 1.15),
        magnitude_warp_sigma=0.3
    )
    
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(device) # Shape: (Batch, Channels, Time)
            
            # Apply augmentations to get two different views
            batch_size = data.shape[0]
            x_i_list = []
            x_j_list = []
            
            for i in range(batch_size):
                signal = data[i]  # (Channels, Time)
                x_i_list.append(augmentor(signal, num_augmentations=2))
                x_j_list.append(augmentor(signal, num_augmentations=2))
            
            x_i = torch.stack(x_i_list)
            x_j = torch.stack(x_j_list)
            
            # Forward pass
            h_i = encoder(x_i)
            h_j = encoder(x_j)
            
            z_i = projection_head(h_i)
            z_j = projection_head(h_j)
            
            # Compute contrastive loss
            loss = criterion(z_i, z_j)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")
        
        if scheduler:
            scheduler.step()
    
    # Save the pre-trained encoder
    os.makedirs('stress_detection/models', exist_ok=True)
    torch.save(encoder.state_dict(), 'stress_detection/models/encoder_pretrained.pth')
    print("Pre-training complete. Model saved.")
