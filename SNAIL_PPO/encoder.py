# SNAIL_PPO/encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class CNNEncoder(nn.Module):
    """
    Convolutional Neural Network encoder that processes image observations into feature vectors.
    """
    def __init__(self, input_channels=12, feature_dim=256, height=40, width=30):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2)
        self._conv_output_size = self._get_conv_output(input_channels, height, width)
        self.fc = nn.Linear(self._conv_output_size, feature_dim)

    def _get_conv_output(self, channels, height, width):
        """
        Calculates the output size after convolutional layers.
        """
        with torch.no_grad():
            input_tensor = torch.zeros(1, channels, height, width)
            output = F.relu(self.conv1(input_tensor), inplace=False)  # Ensure ReLU is non-in-place
            output = F.relu(self.conv2(output), inplace=False)       # Ensure ReLU is non-in-place
            output_size = output.numel()
        logger.debug(f"CNNEncoder: Computed convolutional output size: {output_size}")
        return output_size

    def forward(self, x):
        """
        Forward pass of the CNN encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width)

        Returns:
            torch.Tensor: Feature tensor of shape (Batch, feature_dim)
        """
        x = F.relu(self.conv1(x), inplace=False)  # Ensure ReLU is non-in-place
        x = F.relu(self.conv2(x), inplace=False)  # Ensure ReLU is non-in-place
        x = x.reshape(x.size(0), -1)              # Flatten using reshape
        x = self.fc(x)
        return x
