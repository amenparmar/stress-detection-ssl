
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import ChannelAttention, TemporalAttention

class ModalityEncoder(nn.Module):
    """
    Encoder for a single modality (EDA, TEMP, or BVP).
    """
    def __init__(self, base_filters=32, output_dim=128):
        super(ModalityEncoder, self).__init__()
        
        self.conv1 = nn.Conv1d(1, base_filters, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv1d(base_filters, base_filters*2, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(base_filters*2)
        
        self.conv3 = nn.Conv1d(base_filters*2, base_filters*4, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(base_filters*4)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters*4, output_dim)
    
    def forward(self, x):
        # x: (Batch, 1, Time) - single modality
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MultiModalFusionEncoder(nn.Module):
    """
    Multi-modal fusion encoder with separate encoders for EDA, TEMP, and BVP.
    Fuses features using attention-based mechanism.
    """
    def __init__(self, base_filters=32, modality_dim=128, output_dim=256):
        super(MultiModalFusionEncoder, self).__init__()
        
        # Separate encoders for each modality
        self.eda_encoder = ModalityEncoder(base_filters, modality_dim)
        self.temp_encoder = ModalityEncoder(base_filters, modality_dim)
        self.bvp_encoder = ModalityEncoder(base_filters, modality_dim)
        
        # Fusion method 1: Attention-based fusion
        self.modality_attention = nn.Sequential(
            nn.Linear(modality_dim * 3, modality_dim),
            nn.ReLU(),
            nn.Linear(modality_dim, 3),
            nn.Softmax(dim=1)
        )
        
        # Fusion method 2: Cross-modality interaction
        self.fusion_layer = nn.Sequential(
            nn.Linear(modality_dim * 3, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x):
        # x: (Batch, 3, Time) - multimodal input
        # Split into separate modalities
        eda = x[:, 0:1, :]    # (Batch, 1, Time)
        temp = x[:, 1:2, :]   # (Batch, 1, Time)
        bvp = x[:, 2:3, :]    # (Batch, 1, Time)
        
        # Encode each modality separately
        eda_feat = self.eda_encoder(eda)      # (Batch, modality_dim)
        temp_feat = self.temp_encoder(temp)   # (Batch, modality_dim)
        bvp_feat = self.bvp_encoder(bvp)      # (Batch, modality_dim)
        
        # Concatenate modality features
        concat_feat = torch.cat([eda_feat, temp_feat, bvp_feat], dim=1)  # (Batch, modality_dim*3)
        
        # Attention-based weighting of modalities
        attn_weights = self.modality_attention(concat_feat)  # (Batch, 3)
        attn_weights = attn_weights.unsqueeze(2)  # (Batch, 3, 1)
        
        # Stack modality features and apply attention
        stacked_feat = torch.stack([eda_feat, temp_feat, bvp_feat], dim=1)  # (Batch, 3, modality_dim)
        weighted_feat = stacked_feat * attn_weights  # Apply attention weights
        weighted_feat = weighted_feat.sum(dim=1)  # (Batch, modality_dim)
        
        # Fusion: combine weighted features with concatenated features
        final_concat = torch.cat([weighted_feat, concat_feat], dim=1)  # (Batch, modality_dim*4)
        # Reduce to output_dim*3 first
        final_concat_reduced = final_concat[:, :concat_feat.shape[1]]  # Use concat only
        
        # Final fusion layer
        fused_output = self.fusion_layer(final_concat_reduced)  # (Batch, output_dim)
        
        return fused_output
