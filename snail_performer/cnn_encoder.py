# snail_performer/cnn_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """
    CNN encoder that expects 6 input channels:
      3 image channels,
      1 channel for last_action,
      1 channel for last_reward,
      1 channel for boundary bit.
    Outputs a 256-d embedding.
    Adjust kernel_size/strides to fit your (H, W).
    """
    def __init__(self):
        super().__init__()
        # in_channels=6 to accept additional data channels
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2)
        
        # Suppose input ~ (6, 30, 40). After 2 convs => output shape ~ (16, 5, 7).
        # Flatten => 16*5*7 = 560 => project to 256.
        self.fc = nn.Linear(16 * 5 * 7, 256)

    def forward(self, obs):
        """
        obs shape: (batch_size, 6, H, W)
        Returns a (batch_size, 256) embedding
        """
        x = F.relu(self.conv1(obs))   # => (B,16, ~15,~18) for input (6,30,40)
        x = F.relu(self.conv2(x))     # => (B,16, ~5, ~7)
        x = x.view(x.size(0), -1)     # flatten => (B,560)
        x = F.relu(self.fc(x))        # => (B,256)
        return x
