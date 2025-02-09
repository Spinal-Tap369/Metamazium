# metamazium/snail_performer/cnn_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """
    CNN encoder for input:
      - When used standalone, it expects a 6-channel input (3 image channels + 3 scalar channels).
      - When used in CombinedEncoder, it will be instantiated with 3 channels (only the image channels).
    Projects input to a 256-dimensional embedding.
    """
    def __init__(self, in_channels=6):
        super().__init__()
        # Use the provided in_channels.
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=2)
        # Use adaptive pooling to force a fixed spatial size.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 7))
        # Fully connected layer; note: 16 * 5 * 7 = 560.
        self.fc = nn.Linear(16 * 5 * 7, 256)

    def forward(self, obs):
        """
        Args:
            obs (Tensor): shape (B, in_channels, H, W)
        Returns:
            Tensor: shape (B, 256) embedding of the input
        """
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = self.adaptive_pool(x)
        x = x.reshape(x.size(0), -1)
        return F.relu(self.fc(x))
