# metamazium/snail_performer/cnn_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """
    CNN encoder for 6-channel input:
      - 3 image channels,
      - last_action channel,
      - last_reward channel,
      - boundary bit channel.
    Projects input to a 256-dimensional embedding.
    """
    def __init__(self):
        super().__init__()
        # Two convolutional layers followed by a fully connected layer.
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=2)
        self.fc = nn.Linear(16 * 5 * 7, 256)

    def forward(self, obs):
        """
        Args:
            obs (Tensor): shape (B, 6, H, W)
        Returns:
            Tensor: shape (B, 256) embedding of the input
        """
        x = F.relu(self.conv1(obs))   # Convolve and apply ReLU
        x = F.relu(self.conv2(x))     # Convolve and apply ReLU
        x = x.reshape(x.size(0), -1)  # Flatten for fully connected layer
        return F.relu(self.fc(x))     # Project to 256-dim and apply ReLU
