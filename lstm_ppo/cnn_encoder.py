# lstm_ppo/cnn_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """
    CNN encoder for a (3 x 30 x 40) first-person maze observation (as in SNAIL paper).
    This is similar to the Duan et al. approach: two 5x5 conv layers with stride=2, 16 filters, ReLU.
    Then flatten and project to a 256-dim feature vector.
    If your environment observations differ (e.g. 3 x 40 x 30, or bigger),
    adjust the final shape accordingly.
    """
    def __init__(self):
        super().__init__()
        # You can tweak filter counts if you want it "smaller or bigger"
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2)
        
        # After two stride=2 convs on 30x40, shape is roughly (16, 6, 8) => ~768 dims
        # We'll map that to 256
        self.fc = nn.Linear(16 * 5 * 7, 256)

    def forward(self, obs):
        """
        obs shape: (batch, 3, 30, 40)
        Returns: (batch, 256)
        """
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc(x))
        return x
