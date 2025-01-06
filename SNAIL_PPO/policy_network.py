# SNAIL_PPO/policy_network.py

import torch
import torch.nn as nn
from torch.distributions import Categorical
import cherry as ch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SNAILPolicy(ch.nn.Policy):
    """
    Policy network that uses SNAILAgent's features to output action distributions.
    """
    def __init__(self, snail_agent, num_actions):
        super(SNAILPolicy, self).__init__()
        self.snail_agent = snail_agent
        self.actor = torch.nn.Linear(self.snail_agent.feature_dim, num_actions)
        # Initialize weights 
        torch.nn.init.orthogonal_(self.actor.weight, gain=np.sqrt(2))
        torch.nn.init.constant_(self.actor.bias, 0)
        logger.info("Initialized SNAILPolicy network.")

    def forward(self, state):
        """
        Forward pass to compute the action distribution.

        Args:
            state (torch.Tensor): Tensor of states.

        Returns:
            Distribution: Cherry's distribution instance.
        """
        features = self.snail_agent(state)  # Extract features using SNAILAgent
        logits = self.actor(features)
        return Categorical(logits=logits)
