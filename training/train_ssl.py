
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
    Optimized SimCLR training loop with vectorized augmentations and AMP.
    """
    encoder.train()
    projection_head.train()
    
    # Initialize vectorized augmentor
    augmentor = SignalAugmentation(
        noise_factor=0.05,
        scale_range=(0.85, 1.15),
        magnitude_warp_sigma=0.3
    )
    
    # Initialize AMP scaler
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            if len(batch_data) == 3:
                data, _, _ = batch_data
            else:
                data, _ = batch_data
            
            data = data.to(device, non_blocking=True)
            
            # Vectorized augmentation (much faster)
            x_i = augmentor(data)
            x_j = augmentor(data)
            
            # Concatenate for single forward pass (more efficient kernel launches)
            x_combined = torch.cat([x_i, x_j], dim=0)
            
            optimizer.zero_grad(set_to_none=True)
            
            # AMP: Automatic Mixed Precision
            with torch.cuda.amp.autocast():
                h_combined = encoder(x_combined)
                z_combined = projection_head(h_combined)
                
                # Split back into i and j
                z_i, z_j = torch.split(z_combined, x_i.shape[0], dim=0)
                loss = criterion(z_i, z_j)
            
            # Scale loss and step optimizer
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")
        
        if scheduler:
            scheduler.step()
    
    os.makedirs('stress_detection/models', exist_ok=True)
    torch.save(encoder.state_dict(), 'stress_detection/models/encoder_pretrained.pth')
    print("Pre-training complete. Model saved.")
