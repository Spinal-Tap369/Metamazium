# lstm_ppo/cnn_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """
    CNN encoder for a first-person maze observation (3 x 30 x 40).
    Uses two 5x5 convolutional layers with stride 2 and 16 filters,
    followed by a fully connected layer to project to a 256-dimensional feature vector.
    Adjust dimensions if the input observation shape changes.
    """
    def __init__(self):
        super().__init__()
        # First convolutional layer: input channels = 3, output channels = 16
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2)
        # Second convolutional layer: input channels = 16, output channels = 16
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2)
        
        # After two conv layers on a 30x40 input, the feature map size ~ (16, 5, 7)
        # Flattened size ~ 16*5*7 = 560; project to 256-dim feature vector
        self.fc = nn.Linear(16 * 5 * 7, 256)

    def forward(self, obs):
        """
        Forward pass of the CNN encoder.
        Args:
            obs (Tensor): Batch of observations with shape (batch_size, 3, 30, 40).
        Returns:
            Tensor: Encoded features with shape (batch_size, 256).
        """
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten feature maps
        x = F.relu(self.fc(x))
        return x
