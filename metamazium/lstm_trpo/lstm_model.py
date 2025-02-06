# lstm_ppo/lstm_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn_encoder import CNNEncoder

class StackedLSTMPolicyValueNet(nn.Module):
    """
    A stacked two-layer LSTM network that outputs both policy logits and value estimates.
    - Single-step usage (e.g. env rollout) can use `act_single_step()`, which stores
      the hidden state internally (self.hidden).
    - Batched usage (e.g. PPO updates) should use `forward_with_state(obs, rnn_state)`
      which accepts an external hidden state for shape consistency.
      
    Input shape for batched usage: (B, T, 6, H, W)
    Output:
      - policy_logits: (B, T, action_dim)
      - values: (B, T)
    """

    def __init__(self, action_dim=4, hidden_size=512, num_layers=2):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Shared encoder + LSTM
        self.cnn_encoder = CNNEncoder()
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Separate heads for policy vs. value
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

        # For single-step usage only:
        self.hidden = None
        self.reset_memory()

    def reset_memory(self, batch_size=1, device=None):
        """
        Resets the LSTM's hidden and cell states for single-step usage (self.hidden).
        Typically call at the start of a new trial if you're using `act_single_step()`.
        """
        if device is None:
            device = next(self.parameters()).device
        self.hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        )

    def forward_with_state(self, x, rnn_state):
        """
        For batched usage in PPO/TRPO updates, where x is (B, T, 6, H, W) and
        rnn_state is ( (num_layers, B, hidden_size), (num_layers, B, hidden_size) ).

        Returns: (policy_logits, values, new_rnn_state)
          - policy_logits shape: (B, T, action_dim)
          - values shape: (B, T)
          - new_rnn_state: updated (hidden, cell)
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn_encoder(x)
        feats = feats.view(B, T, 256)

        lstm_out, new_rnn_state = self.lstm(feats, rnn_state)
        policy_logits = self.policy_head(lstm_out)  # (B, T, action_dim)
        values = self.value_head(lstm_out).squeeze(-1)  # (B, T)
        return policy_logits, values, new_rnn_state

    def act_single_step(self, x):
        """
        Single-step forward pass for environment rollout.
        x shape: (B, 1, 6, H, W). Typically B=1 for single-step usage.

        Returns: (policy_logits, values) each with shape (B, 1, ...).
        Uses and updates self.hidden in-place.
        """
        self.lstm.flatten_parameters()

        B, T, C, H, W = x.shape  # T=1 usually
        x = x.view(B * T, C, H, W)
        feats = self.cnn_encoder(x)
        feats = feats.view(B, T, 256)

        lstm_out, self.hidden = self.lstm(feats, self.hidden)
        # Detach so that we don't backprop through previous timesteps
        self.hidden = tuple(h.detach() for h in self.hidden)

        policy_logits = self.policy_head(lstm_out)      # shape (B, 1, action_dim)
        values = self.value_head(lstm_out).squeeze(-1)  # shape (B, 1)
        return policy_logits, values

    def value(self, obs, prev_action, prev_reward, rnn_state, training=False):
        """
        This method is for externally computing values with an external rnn_state.
        obs shape: (B, T, 6, H, W)
        rnn_state: (hidden, cell)
        Returns: (values, new_rnn_state)
        """
        # We can ignore prev_action / prev_reward or incorporate them if needed.
        policy_logits, values, new_rnn_state = self.forward_with_state(obs, rnn_state)
        return values, new_rnn_state

    def pi(self, obs, prev_action, prev_reward, rnn_state, action=None, training=False):
        """
        Returns a distribution over actions (dist) and optional log_prob if 'action' is given.
        obs shape: (B, T, 6, H, W)
        rnn_state: (hidden, cell)
        Returns: (dist, log_prob, new_rnn_state)
        """
        logits, values, new_rnn_state = self.forward_with_state(obs, rnn_state)
        dist = torch.distributions.Categorical(logits=logits)

        if action is not None:
            log_prob = dist.log_prob(action)
        else:
            log_prob = None

        return dist, log_prob, new_rnn_state

    # ------------------------------------------------------------------------
    # Separate parameter lists for policy vs. value
    # ------------------------------------------------------------------------
    def policy_parameters(self):
        """
        Return a list of parameters that affect the policy output.
        Typically includes shared encoder + LSTM + policy head.
        """
        return list(self.cnn_encoder.parameters()) + \
               list(self.lstm.parameters()) + \
               list(self.policy_head.parameters())

    def value_parameters(self):
        """
        Return a list of parameters that affect the value output.
        Typically includes shared encoder + LSTM + value head.
        """
        return list(self.cnn_encoder.parameters()) + \
               list(self.lstm.parameters()) + \
               list(self.value_head.parameters())
