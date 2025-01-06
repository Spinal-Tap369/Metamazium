# SNAIL_PPO/agent.py

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from collections import deque

from .encoder import CNNEncoder
from .network import SNAILNetwork

import logging

logger = logging.getLogger(__name__)

class SNAILAgent(nn.Module):
    def __init__(
        self,
        feature_dim=256,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        seq_length=4,  # Number of steps in the sequence
        num_actions=16
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_length = seq_length
        self.encoder = CNNEncoder(
            input_channels=12,  # 4 frames stacked with 3 channels each
            feature_dim=feature_dim,
            height=40,
            width=30
        )
        self.snail = SNAILNetwork(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            seq_length=seq_length,
            num_actions=num_actions
        )

        self.num_actions = num_actions
        self.buffer = deque(maxlen=self.seq_length)

        self.device = torch.device("cpu")

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def reset_buffer(self):
        self.buffer.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.debug(f"Original input shape: {x.shape}")
        if x.dim() == 3:
            x = x.unsqueeze(0)  # [1, H, W, C * Seq_length]
            logger.debug(f"Added batch dimension, new shape: {x.shape}")
        elif x.dim() != 4:
            raise ValueError(f"Unexpected input dimensions: {x.dim()} (expected 3 or 4)")

        features = self.encoder(x)  # [Batch, Feature_dim]
        logger.debug(f"Encoded features shape: {features.shape}")

        self.buffer.append(features)

        if len(self.buffer) < self.seq_length:
            padding = self.seq_length - len(self.buffer)
            padded_features = torch.zeros((features.size(0), padding, self.feature_dim)).to(features.device)
            for i, feat in enumerate(self.buffer):
                padded_features[:, i, :] = feat
            logger.debug(f"Padded features shape: {padded_features.shape}")
            return padded_features
        else:
            stacked_features = torch.stack(self.buffer, dim=1)  # [Batch, Seq_length, Feature_dim]
            logger.debug(f"Stacked features shape: {stacked_features.shape}")
            return stacked_features

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        sequence = self.forward(x)  # [Batch, Seq_length, Feature_dim]
        logger.debug(f"Sequence shape for SNAILNetwork: {sequence.shape}")

        policy_logits, value = self.snail(sequence)  # [Batch, num_actions], [Batch]
        logger.debug(f"Policy logits shape: {policy_logits.shape}, Value shape: {value.shape}")

        probs = Categorical(logits=policy_logits)

        if action is None:
            action = probs.sample()

        logprob = probs.log_prob(action)
        entropy = probs.entropy()

        logger.debug(f"Action shape: {action.shape}, Logprob shape: {logprob.shape}, Entropy shape: {entropy.shape}")

        return action, logprob, entropy, value

    @property
    def policy_parameters(self):
        return list(self.encoder.parameters()) + list(self.snail.transformer.parameters()) + list(self.snail.policy_head.parameters())

    @property
    def value_parameters(self):
        return list(self.snail.value_head.parameters())
