# snail_performer/cnn_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """
    CNN encoder for a first-person maze observation (3 x 30 x 40).
    Uses two 5x5 convolutional layers with stride=2 and 16 filters,
    followed by a fully connected layer to project to a 256-dimensional feature vector.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2)
        
        # For an input ~ (3, 30, 40):
        # after 2 convs with stride=2 => ~ (16, 5, 7) => flattened => ~560 dims
        # project to 256
        self.fc = nn.Linear(16 * 5 * 7, 256)

    def forward(self, obs):
        """
        obs shape: (batch, 3, 30, 40)
        Returns a (batch, 256) embedding.
        """
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc(x))
        return x
