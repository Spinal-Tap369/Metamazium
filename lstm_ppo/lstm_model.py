# lstm_ppo/lstm_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn_encoder import CNNEncoder

class LSTMPolicy(nn.Module):
    def __init__(self, action_dim=4, hidden_size=512):
        """
        Initializes the LSTM-based policy model.
        
        Args:
            action_dim (int): Number of possible actions.
            hidden_size (int): Hidden size of the LSTM.
        """
        super().__init__()
        self.cnn_encoder = CNNEncoder()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size, batch_first=True)
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

        self._h = None
        self._c = None

    def reset_memory(self, batch_size=1, device=None):
        """
        Resets LSTM hidden and cell states.

        Args:
            batch_size (int): Batch size for the LSTM states.
            device (torch.device, optional): Device to initialize the states on.
        """
        if device is None:
            device = next(self.parameters()).device

        self._h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        self._c = torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward(self, obs_seq):
        """
        Processes a sequence of observations through the model.

        Args:
            obs_seq (torch.Tensor): Input tensor of shape (B, T, 3, 30, 40).
        
        Returns:
            tuple: Policy logits (B, T, action_dim) and value estimates (B, T).
        """
        B, T, C, H, W = obs_seq.shape
        x = obs_seq.view(B * T, C, H, W)
        feats = self.cnn_encoder(x)  # => (B*T, 256)
        feats = feats.view(B, T, -1)  # => (B, T, 256)

        lstm_out, (self._h, self._c) = self.lstm(feats, (self._h, self._c))
        policy_logits = self.policy_head(lstm_out)     # (B, T, action_dim)
        values = self.value_head(lstm_out).squeeze(-1) # (B, T)
        return policy_logits, values

    @torch.no_grad()
    def act_single_step(self, obs_single):
        """
        Performs a single step action for the environment.

        Args:
            obs_single (torch.Tensor): Input tensor of shape (3, 30, 40).
        
        Returns:
            tuple: Logits (1, action_dim) and value (1).
        """
        obs_single = obs_single.unsqueeze(0).unsqueeze(0)
        logits, vals = self.forward(obs_single)  # => shapes (1, 1, action_dim), (1, 1)
        return logits[:, -1, :], vals[:, -1]
