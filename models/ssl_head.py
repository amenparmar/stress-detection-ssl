
import torch.nn as nn

class SSLHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SSLHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
