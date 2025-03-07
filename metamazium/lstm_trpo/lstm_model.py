# metamazium/lstm_trpo/lstm_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .combined_encoder import CombinedEncoder

class StackedLSTMPolicyValueNet(nn.Module):
    """
    A stacked two-layer LSTM network that outputs both policy logits and value estimates.
    
    This model is adapted for a discrete action space.
    Input shape (for batched usage): (B, T, 6, H, W)
    Outputs:
      - policy_logits: (B, T, action_dim)
      - values: (B, T)
    
    Single-step usage via act_single_step() uses an internal hidden state.
    """
    def __init__(self, action_dim=4, hidden_size=512, num_layers=2):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Use CombinedEncoder for splitting the 6-channel input.
        self.combined_encoder = CombinedEncoder(base_dim=256, scalar_out_dim=64)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

        self.hidden = None
        self.reset_memory()

    def reset_memory(self, batch_size=1, device=None):
        if device is None:
            device = next(self.parameters()).device
        self.hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        )

    def forward_with_state(self, x, rnn_state):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.combined_encoder(x)  # (B*T, 256)
        feats = feats.view(B, T, 256)
        lstm_out, new_rnn_state = self.lstm(feats, rnn_state)
        policy_logits = self.policy_head(lstm_out)  # (B, T, action_dim)
        values = self.value_head(lstm_out).squeeze(-1)  # (B, T)
        return policy_logits, values, new_rnn_state

    def act_single_step(self, x):
        """
        Single-step forward pass for environment rollout.
        x: Tensor of shape (B, 1, 6, H, W) (typically B=1)
        Returns:
          - policy_logits: (B, 1, action_dim)
          - values: (B, 1)
        Updates self.hidden.
        """
        self.lstm.flatten_parameters()
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.combined_encoder(x)  # (B*T, 256)
        feats = feats.view(B, T, 256)
        lstm_out, self.hidden = self.lstm(feats, self.hidden)
        # Detach hidden state to prevent backpropagating over previous timesteps.
        self.hidden = tuple(h.detach() for h in self.hidden)
        policy_logits = self.policy_head(lstm_out)
        values = self.value_head(lstm_out).squeeze(-1)
        return policy_logits, values

    def value(self, obs, prev_action, prev_reward, rnn_state, training=False):
        _, values, new_rnn_state = self.forward_with_state(obs, rnn_state)
        return values, new_rnn_state

    def pi(self, obs, prev_action, prev_reward, rnn_state, action=None, training=False):
        logits, _, new_rnn_state = self.forward_with_state(obs, rnn_state)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action) if action is not None else None
        return dist, log_prob, new_rnn_state

    def policy_parameters(self):
        return list(self.combined_encoder.parameters()) + \
               list(self.lstm.parameters()) + \
               list(self.policy_head.parameters())

    def value_parameters(self):
        return list(self.combined_encoder.parameters()) + \
               list(self.lstm.parameters()) + \
               list(self.value_head.parameters())
