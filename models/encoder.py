
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import ChannelAttention, TemporalAttention

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, 0),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    """
    Enhanced encoder with attention mechanisms for improved feature learning.
    Includes channel and temporal attention layers.
    """
    def __init__(self, input_channels=3, base_filters=32, output_dim=256):
        super(Encoder, self).__init__()
        
        # Initial Conv
        self.conv1 = nn.Conv1d(input_channels, base_filters, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Channel attention after first conv (focuses on important modalities)
        self.channel_attn1 = ChannelAttention(num_channels=base_filters, reduction_ratio=2)
        
        # ResBlocks
        self.layer1 = ResBlock1D(base_filters, base_filters)
        self.layer2 = ResBlock1D(base_filters, base_filters*2, stride=2)
        self.layer3 = ResBlock1D(base_filters*2, base_filters*4, stride=2)
        self.layer4 = ResBlock1D(base_filters*4, base_filters*8, stride=2)
        
        # Temporal attention before pooling (focuses on important time steps)
        self.temporal_attn = TemporalAttention(hidden_dim=base_filters*8)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Final FC to output dim
        self.fc = nn.Linear(base_filters*8, output_dim)

    def forward(self, x):
        # Initial conv with channel attention
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.channel_attn1(x)  # Apply channel attention
        
        # ResBlocks for feature extraction
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Temporal attention before pooling
        x = self.temporal_attn(x)  # Apply temporal attention
        
        # Global pooling and FC
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
