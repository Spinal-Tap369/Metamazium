import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn_encoder import CNNEncoder

class StackedLSTMPolicy(nn.Module):
    """
    A stacked two-layer LSTM policy network that processes sequences of observations.

    The network takes an input tensor of shape (B, T, 6, H, W), where:
      - B is the batch size,
      - T is the sequence length,
      - 6 is the number of channels expected by the CNN encoder,
      - H and W are the height and width of the image observations.

    The processing pipeline is as follows:
      1. Flatten the input to shape (B*T, 6, H, W) and pass it through the CNN encoder to obtain embeddings of shape (B*T, 256).
      2. Reshape these embeddings to (B, T, 256) for sequential processing.
      3. Pass the sequence through a two-layer stacked LSTM to produce outputs of shape (B, T, hidden_size).
      4. Apply linear projections to produce policy logits of shape (B, T, action_dim) and value estimates of shape (B, T).
    """
    def __init__(self, action_dim=4, hidden_size=512, num_layers=2):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Initialize the CNN encoder to handle 6-channel input.
        self.cnn_encoder = CNNEncoder()

        # Two-layer LSTM with batch_first format.
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Linear heads to predict policy logits and state value estimates.
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass of the stacked LSTM policy network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, 6, H, W).

        Returns:
            tuple: A tuple containing:
              - policy_logits (torch.Tensor): Tensor of shape (B, T, action_dim).
              - values (torch.Tensor): Tensor of shape (B, T) representing value estimates.
        """
        B, T, C, H, W = x.shape
        # Reshape input to combine batch and time dimensions for CNN processing.
        x2 = x.view(B*T, C, H, W)  # Shape: (B*T, 6, H, W)
        feats = self.cnn_encoder(x2)  # Shape: (B*T, 256)

        # Reshape back to separate batch and sequence dimensions.
        feats_3d = feats.view(B, T, 256)  # Shape: (B, T, 256)
        lstm_out, _ = self.lstm(feats_3d)  # Shape: (B, T, hidden_size)

        # Compute policy logits and value estimates.
        policy_logits = self.policy_head(lstm_out)      # Shape: (B, T, action_dim)
        values = self.value_head(lstm_out).squeeze(-1)  # Shape: (B, T)
        return policy_logits, values

    def forward_rollout(self, x):
        """
        Wrapper for forward pass to maintain naming consistency.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, 6, H, W).

        Returns:
            tuple: See forward method.
        """
        return self.forward(x)
