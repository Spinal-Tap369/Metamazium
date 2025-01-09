# lstm_ppo/lstm_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn_encoder import CNNEncoder

class LSTMPolicy(nn.Module):
    """
    A 'powerful' LSTM-based policy that:
      1) Encodes the visual observation with a CNN
      2) Passes the 256-dim embedding into an LSTM
      3) Outputs policy logits and value estimate
    We assume a discrete action space (e.g., 4 actions).
    """

    def __init__(self, action_dim=4, hidden_size=512):
        """
        action_dim: # of discrete actions in MetaMazeDiscrete3D (often 4).
        hidden_size: LSTM hidden size. 512 is a decent starting point.
                     If memory is too high, reduce to 256 or 128.
        """
        super().__init__()
        self.cnn_encoder = CNNEncoder()
        
        self.hidden_size = hidden_size
        # Single-layer LSTM to handle up to T=500 steps if needed
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size, batch_first=True)
        
        # We produce policy logits and value
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)
        
        # We'll keep track of the LSTM hidden states as we step
        self.reset_lstm_states()

    def reset_lstm_states(self, batch_size=1):
        """
        Clears hidden states for the LSTM at the start of an episode (or new rollout).
        By default, we set batch=1 for a single environment. 
        If you're using multi-env, pass batch_size = num_envs.
        """
        device = next(self.parameters()).device
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        self.lstm_hidden = (h0, c0)

    def forward(self, obs_seq):
        """
        For a sequence of observations. 
        obs_seq shape: (batch, T, 3, 30, 40) if each obs is an image of shape (3x30x40).
        
        We do:
         1) Flatten out (batch*T) -> pass CNN -> reshape to (batch, T, 256)
         2) LSTM -> (batch, T, hidden_size)
         3) policy_head, value_head -> (batch, T, action_dim) and (batch, T)
        """
        B, T, C, H, W = obs_seq.shape
        # (B*T, 3, 30, 40)
        obs_reshaped = obs_seq.view(B*T, C, H, W)
        feats = self.cnn_encoder(obs_reshaped)  # (B*T, 256)
        feats = feats.view(B, T, -1)            # (B, T, 256)
        
        # Pass through LSTM
        # We do a single forward pass on the entire sequence if we have enough memory
        lstm_out, self.lstm_hidden = self.lstm(feats, self.lstm_hidden)  # shape: (B, T, hidden_size)
        
        # produce policy logits and value
        policy_logits = self.policy_head(lstm_out)          # (B, T, action_dim)
        values = self.value_head(lstm_out).squeeze(-1)      # (B, T)
        
        return policy_logits, values

    def act_single_step(self, obs_single):
        """
        For a single step environment usage (obs_single shape: (3,30,40)).
        We'll do:
          1) Expand to (1,1,3,30,40)
          2) forward pass
          3) extract the last step's policy logits, value
        """
        obs_single = obs_single.unsqueeze(0).unsqueeze(0)  # shape -> (1,1,3,30,40)
        with torch.no_grad():
            policy_logits, values = self.forward(obs_single)
        # shape: policy_logits -> (1,1,action_dim), values -> (1,1)
        return policy_logits[:, -1, :], values[:, -1]
