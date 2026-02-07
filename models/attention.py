
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """
    Channel attention mechanism to learn importance of different signal modalities.
    Helps the model focus on most informative channels (EDA, TEMP, BVP).
    """
    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        mid_channels = max(num_channels // reduction_ratio, 1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, num_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (Batch, Channels, Time)
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        out = avg_out + max_out
        attention = self.sigmoid(out).unsqueeze(-1)
        return x * attention


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism to focus on important time steps in signals.
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (Batch, Channels, Time)
        attention_weights = self.attention(x)
        return x * attention_weights


class SelfAttention(nn.Module):
    """
    Multi-head self-attention for capturing long-range dependencies in signals.
    """
    def __init__(self, embed_dim, num_heads=4):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x: (Batch, Channels, Time)
        # Reshape for attention: (Batch, Time, Channels)
        x = x.transpose(1, 2)
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.norm(x + attn_output)
        # Reshape back: (Batch, Channels, Time)
        return x.transpose(1, 2)
