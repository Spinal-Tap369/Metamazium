# lstm_ppo/lstm_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn_encoder import CNNEncoder

class LSTMPolicy(nn.Module):
    """
    LSTM-based policy:
      1) Encodes observations with a CNN
      2) Processes embeddings through an LSTM
      3) Outputs policy logits and value estimates
    """
    def __init__(self, action_dim=4, hidden_size=512):
        """
        action_dim: Number of discrete actions.
        hidden_size: LSTM hidden size.
        """
        super().__init__()
        self.cnn_encoder = CNNEncoder()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size, batch_first=True)
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)
        self.reset_lstm_states()

    def reset_lstm_states(self, batch_size=1):
        """Reset LSTM hidden states."""
        device = next(self.parameters()).device
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        self.lstm_hidden = (h0, c0)

    def forward(self, obs_seq):
        """
        Forward pass for a sequence of observations.
        obs_seq shape: (batch, T, 3, 30, 40)
        """
        B, T, C, H, W = obs_seq.shape
        obs_reshaped = obs_seq.view(B*T, C, H, W)
        feats = self.cnn_encoder(obs_reshaped).view(B, T, -1)
        lstm_out, self.lstm_hidden = self.lstm(feats, self.lstm_hidden)
        policy_logits = self.policy_head(lstm_out)
        values = self.value_head(lstm_out).squeeze(-1)
        return policy_logits, values

    def act_single_step(self, obs_single):
        """
        Process a single observation step.
        obs_single shape: (3, 30, 40)
        """
        obs_single = obs_single.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            policy_logits, values = self.forward(obs_single)
        return policy_logits[:, -1, :], values[:, -1]
