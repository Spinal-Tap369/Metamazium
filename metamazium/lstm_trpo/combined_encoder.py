# metamazium/lstm_trpo/combined_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn_encoder import CNNEncoder  # Reuse the updated CNNEncoder

class ScalarEncoder(nn.Module):
    """
    A simple MLP that encodes scalar inputs (last_action, last_reward, boundary flag).
    Input: Tensor of shape (B, 3)
    Output: Tensor of shape (B, 64)
    """
    def __init__(self, in_dim=3, hidden_dim=32, out_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

class CombinedEncoder(nn.Module):
    """
    Splits a 6-channel input into:
      - A 3-channel image (first 3 channels) processed by a CNNEncoder (with in_channels=3),
      - A 3-channel scalar part (last 3 channels), assumed constant over spatial dimensions.
    It spatially aggregates the scalar channels (using mean over H and W), encodes them via ScalarEncoder,
    concatenates the two embeddings, and projects the result to a 256-dimensional vector.
    """
    def __init__(self, base_dim=256, scalar_out_dim=64):
        super().__init__()
        # Process only the image part (first 3 channels).
        self.cnn_encoder = CNNEncoder(in_channels=3)
        self.scalar_encoder = ScalarEncoder(in_dim=3, hidden_dim=32, out_dim=scalar_out_dim)
        # Project concatenated embedding (256 + scalar_out_dim) to base_dim.
        self.proj = nn.Linear(256 + scalar_out_dim, base_dim)
        
    def forward(self, obs):
        """
        Args:
            obs (Tensor): shape (B, 6, H, W)
        Returns:
            Tensor: shape (B, base_dim)
        """
        image_part = obs[:, :3, :, :]   # (B, 3, H, W)
        scalar_part = obs[:, 3:, :, :]   # (B, 3, H, W)
        # Since the scalar channels are constant over spatial dimensions,
        # take the mean over H and W.
        scalar_vec = scalar_part.mean(dim=[2, 3])  # (B, 3)
        img_embed = self.cnn_encoder(image_part)     # (B, 256)
        scalar_embed = self.scalar_encoder(scalar_vec) # (B, 64)
        combined = torch.cat([img_embed, scalar_embed], dim=1)  # (B, 320)
        return F.relu(self.proj(combined))  # (B, base_dim)
