# SNAIL_PPO/value_network.py

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class SNAILValue(nn.Module):
    """
    State-Value Network for PPO using SNAILAgent's features.
    """
    def __init__(self, snail_agent, feature_dim):
        super(SNAILValue, self).__init__()
        self.snail_agent = snail_agent
        self.value_head = nn.Linear(feature_dim, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0)
        logger.info("Initialized SNAILValue network.")
    
    def forward(self, state):
        """
        Forward pass to compute the state-value.

        Args:
            state (torch.Tensor): Tensor of states.

        Returns:
            torch.Tensor: Estimated value of the state.
        """
        features = self.snail_agent(state)
        value = self.value_head(features).squeeze(-1)  # Shape: (batch_size,)
        return value

    def evaluate_actions(self, observations, actions=None):
        """
        Evaluate state-value estimates for given observations.

        Args:
            observations (torch.Tensor): Batch of observations.
            actions (torch.Tensor, optional): Ignored, kept for compatibility.

        Returns:
            torch.Tensor: State-value estimates.
        """
        return self.forward(observations)
