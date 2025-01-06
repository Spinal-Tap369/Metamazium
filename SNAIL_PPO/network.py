# SNAIL_PPO/network.py

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class SNAILNetwork(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers, num_heads, seq_length, num_actions):
        super(SNAILNetwork, self).__init__()
        # Example: Using Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.policy_head = nn.Linear(feature_dim, num_actions)
        self.value_head = nn.Linear(feature_dim, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [Batch, Seq_length, Feature_dim]
        Returns:
            policy_logits (torch.Tensor): [Batch, num_actions]
            value (torch.Tensor): [Batch]
        """
        if x.dim() != 3:
            raise ValueError(f"Unexpected input dimensions: {x.dim()} (expected 3)")
        x = x.permute(1, 0, 2)  # [Seq_length, Batch, Feature_dim]
        logger.debug(f"SNAILNetwork Forward: Permuted input shape: {x.shape}")

        transformer_output = self.transformer(x)  # [Seq_length, Batch, Feature_dim]
        logger.debug(f"SNAILNetwork Forward: Transformer output shape: {transformer_output.shape}")

        last_output = transformer_output[-1, :, :]  # [Batch, Feature_dim]
        logger.debug(f"SNAILNetwork Forward: Last output shape: {last_output.shape}")

        policy_logits = self.policy_head(last_output)  # [Batch, num_actions]
        value = self.value_head(last_output).squeeze(-1)  # [Batch]

        return policy_logits, value
