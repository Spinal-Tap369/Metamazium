# lstm_ppo/lstm_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn_encoder import CNNEncoder

class StackedLSTMPolicyValueNet(nn.Module):
    """
    A stacked two-layer LSTM network that outputs both policy logits and value estimates.
    
    This model is adapted for a 3-action discrete environment:
      1. Move forward
      2. Turn slightly left (15 degrees)
      3. Turn slightly right (15 degrees)
    
    - Single-step usage (e.g. during environment rollout) can use `act_single_step()`, which stores
      the hidden state internally (self.hidden).
    - Batched usage (e.g. during PPO updates) should use `forward_with_state(obs, rnn_state)`,
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
        Typically call this at the start of a new trial if you're using `act_single_step()`.
        """
        if device is None:
            device = next(self.parameters()).device
        self.hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        )

    def forward_with_state(self, x, rnn_state):
        """
        For batched usage (e.g., in PPO/TRPO updates), where:
          - x is of shape (B, T, 6, H, W)
          - rnn_state is a tuple of (hidden, cell) of shapes ((num_layers, B, hidden_size), (num_layers, B, hidden_size))
        
        Returns:
          - policy_logits: (B, T, action_dim)
          - values: (B, T)
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
        
        Args:
          x: Tensor of shape (B, 1, 6, H, W). Typically B=1 for single-step usage.
          
        Returns:
          - policy_logits: Tensor of shape (B, 1, action_dim)
          - values: Tensor of shape (B, 1)
        
        Uses and updates self.hidden in-place.
        """
        self.lstm.flatten_parameters()

        B, T, C, H, W = x.shape  # T is usually 1
        x = x.view(B * T, C, H, W)
        feats = self.cnn_encoder(x)
        feats = feats.view(B, T, 256)

        lstm_out, self.hidden = self.lstm(feats, self.hidden)
        # Detach the hidden state to prevent backpropagating through previous timesteps
        self.hidden = tuple(h.detach() for h in self.hidden)

        policy_logits = self.policy_head(lstm_out)  # (B, 1, action_dim)
        values = self.value_head(lstm_out).squeeze(-1)  # (B, 1)
        return policy_logits, values

    def value(self, obs, prev_action, prev_reward, rnn_state, training=False):
        """
        Computes value estimates given observations and an external RNN state.
        
        Args:
          obs: Tensor of shape (B, T, 6, H, W)
          rnn_state: Tuple (hidden, cell)
          
        Returns:
          - values: Tensor of shape (B, T)
          - new_rnn_state: updated RNN state
        """
        _, values, new_rnn_state = self.forward_with_state(obs, rnn_state)
        return values, new_rnn_state

    def pi(self, obs, prev_action, prev_reward, rnn_state, action=None, training=False):
        """
        Computes a distribution over actions along with the corresponding log probability (if action is provided).
        
        Args:
          obs: Tensor of shape (B, T, 6, H, W)
          rnn_state: Tuple (hidden, cell)
          action: (optional) Tensor of actions for which to compute the log probability
          
        Returns:
          - dist: A Categorical distribution over actions
          - log_prob: Log probability of the provided action (or None if no action is given)
          - new_rnn_state: updated RNN state
        """
        logits, _, new_rnn_state = self.forward_with_state(obs, rnn_state)
        dist = torch.distributions.Categorical(logits=logits)

        if action is not None:
            log_prob = dist.log_prob(action)
        else:
            log_prob = None

        return dist, log_prob, new_rnn_state

    def policy_parameters(self):
        """
        Returns a list of parameters that affect the policy output.
        This typically includes the shared encoder, LSTM, and policy head.
        """
        return list(self.cnn_encoder.parameters()) + \
               list(self.lstm.parameters()) + \
               list(self.policy_head.parameters())

    def value_parameters(self):
        """
        Returns a list of parameters that affect the value output.
        This typically includes the shared encoder, LSTM, and value head.
        """
        return list(self.cnn_encoder.parameters()) + \
               list(self.lstm.parameters()) + \
               list(self.value_head.parameters())
