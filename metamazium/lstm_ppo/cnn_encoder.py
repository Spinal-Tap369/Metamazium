import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """
    A convolutional neural network (CNN) encoder tailored for a 6-channel input:
      - 3 channels for RGB image data,
      - 1 channel representing the last action taken,
      - 1 channel representing the last received reward,
      - 1 channel for a boundary indicator bit.

    The encoder processes the input and outputs a 256-dimensional feature embedding.
    Adjust the kernel sizes and strides as needed if the input shape changes.
    """
    def __init__(self):
        super().__init__()
        # Initialize convolutional layers for 6 input channels.
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2)
        
        # Based on an example input shape (6, 30, 40), two conv layers result in an output ~ (16, 5, 7).
        # Flattening this output gives 16*5*7 = 560 features, which are then projected to a 256-dimensional space.
        self.fc = nn.Linear(16 * 5 * 7, 256)

    def forward(self, obs):
        """
        Forward pass of the CNN encoder.

        Args:
            obs (torch.Tensor): Input tensor of shape (batch_size, 6, H, W).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, 256) representing the encoded features.
        """
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)  # Flatten the convolutional output.
        x = F.relu(self.fc(x))
        return x
