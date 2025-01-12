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
        self.reset_lstm_states(mode='act')  # Initialize acting hidden state
        self.reset_lstm_states(mode='update')  # Initialize updating hidden state

    def reset_lstm_states(self, batch_size=1, mode='act'):
        """Reset LSTM hidden states for acting or updating."""
        device = next(self.parameters()).device
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        if mode == 'act':
            self.lstm_hidden_act = (h0, c0)
        elif mode == 'update':
            self.lstm_hidden_update = (h0, c0)
        else:
            raise ValueError("Mode must be either 'act' or 'update'.")

    def forward_act(self, obs_seq):
        """
        Forward pass for acting steps.
        """
        B, T, C, H, W = obs_seq.shape
        obs_reshaped = obs_seq.view(B * T, C, H, W)
        feats = self.cnn_encoder(obs_reshaped).view(B, T, -1)
        lstm_out, self.lstm_hidden_act = self.lstm(feats, self.lstm_hidden_act)
        policy_logits = self.policy_head(lstm_out)
        values = self.value_head(lstm_out).squeeze(-1)
        return policy_logits, values

    def forward_update(self, obs_seq):
        """
        Forward pass for PPO update steps.
        """
        B, T, C, H, W = obs_seq.shape
        obs_reshaped = obs_seq.view(B * T, C, H, W)
        feats = self.cnn_encoder(obs_reshaped).view(B, T, -1)
        lstm_out, self.lstm_hidden_update = self.lstm(feats, self.lstm_hidden_update)
        policy_logits = self.policy_head(lstm_out)
        values = self.value_head(lstm_out).squeeze(-1)
        return policy_logits, values

    def forward(self, obs_seq, mode='act'):
        """
        General forward method to handle different modes.
        """
        if mode == 'act':
            return self.forward_act(obs_seq)
        elif mode == 'update':
            return self.forward_update(obs_seq)
        else:
            raise ValueError("Mode must be either 'act' or 'update'.")

    def act_single_step(self, obs_single):
        """
        Process a single observation step.
        obs_single shape: (3, 30, 40)
        """
        obs_single = obs_single.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 3, 30, 40)
        with torch.no_grad():
            policy_logits, values = self.forward(obs_single, mode='act')
        return policy_logits[:, -1, :], values[:, -1]
