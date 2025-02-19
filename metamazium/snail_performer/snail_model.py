import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from metamazium.snail_performer.cnn_encoder import CNNEncoder

# ScalarEncoder module 
class ScalarEncoder(nn.Module):
    """
    A simple MLP that encodes scalar inputs (last_action, last_reward, termination flag).
    Input: Tensor of shape (B, 3)
    Output: Tensor of shape (B, 64)
    """
    def __init__(self, in_dim=3, hidden_dim=32, out_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

# CombinedEncoder module       
class CombinedEncoder(nn.Module):
    """
    Splits a 6-channel input into:
      - A 3-channel image (first 3 channels), processed via a CNNEncoder.
      - A 3-channel scalar part (last 3 channels), assumed to be constant over H, W.
    It extracts the scalar values (by averaging spatially), encodes them via ScalarEncoder,
    then concatenates the two embeddings and projects them back to a 256-dimensional vector.
    """
    def __init__(self, base_dim=256, scalar_out_dim=64):
        super().__init__()
        # Process only the image part (first 3 channels).
        self.cnn_encoder = CNNEncoder(in_channels=3)
        self.scalar_encoder = ScalarEncoder(in_dim=3, hidden_dim=32, out_dim=scalar_out_dim)
        # Project concatenated embedding (256 + scalar_out_dim) to base_dim.
        self.proj = nn.Linear(256 + scalar_out_dim, base_dim)
        
    def forward(self, obs):
        """
        Args:
            obs (Tensor): shape (B, 6, H, W)
        Returns:
            Tensor: shape (B, base_dim)
        """
        # Split the channels.
        image_part = obs[:, :3, :, :]   # shape: (B, 3, H, W)
        scalar_part = obs[:, 3:, :, :]   # shape: (B, 3, H, W)
        # Since the scalar channels are constant across spatial dimensions,
        # take the mean over H and W.
        scalar_vec = scalar_part.mean(dim=[2, 3])  # shape: (B, 3)
        img_embed = self.cnn_encoder(image_part)     # shape: (B, 256)
        scalar_embed = self.scalar_encoder(scalar_vec) # shape: (B, 64)
        # Concatenate and project.
        combined = torch.cat([img_embed, scalar_embed], dim=1)  # shape: (B, 256+64)
        return F.relu(self.proj(combined))  # shape: (B, base_dim)

# DenseBlock module (unchanged)
class DenseBlock(nn.Module):
    def __init__(self, in_dim, dilation, filters):
        super().__init__()
        self.out_dim = in_dim + filters
        self.conv_f = nn.Conv1d(in_dim, filters, kernel_size=2, dilation=dilation, padding=dilation)
        self.conv_g = nn.Conv1d(in_dim, filters, kernel_size=2, dilation=dilation, padding=dilation)

    def forward(self, x):
        B, C, T = x.shape
        xf = self.conv_f(x)[:, :, :T]
        xg = self.conv_g(x)[:, :, :T]
        act = torch.tanh(xf) * torch.sigmoid(xg)
        return torch.cat([x, act], dim=1)

# TCBlock module (unchanged)
class TCBlock(nn.Module):
    def __init__(self, in_dim, seq_len, filters):
        super().__init__()
        ms = math.ceil(math.log2(seq_len + 1))
        blocks = []
        cur_dim = in_dim
        for i in range(ms):
            blk = DenseBlock(cur_dim, dilation=2**i, filters=filters)
            blocks.append(blk)
            cur_dim = blk.out_dim
        self.blocks = nn.ModuleList(blocks)
        self.out_dim = cur_dim

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

# New Attention Block following SNAIL's paper
class SnailAttentionBlock(nn.Module):
    """
    Implements the attention block as described in the SNAIL paper:
      - Applies separate linear layers to compute query, keys, and values.
      - Computes scaled dot-product attention with a causal mask.
      - Computes a "read" vector and concatenates it with the input.
    """
    def __init__(self, in_dim, key_size, value_size):
        """
        Args:
            in_dim (int): Dimension of the input features.
            key_size (int): Dimension of the key (and query) vectors.
            value_size (int): Dimension of the value vectors.
        """
        super().__init__()
        self.linear_query = nn.Linear(in_dim, key_size)
        self.linear_keys = nn.Linear(in_dim, key_size)
        self.linear_values = nn.Linear(in_dim, value_size)
        self.sqrt_key_size = math.sqrt(key_size)
        # The output dimension is the concatenation of the input and the read vector.
        self.out_dim = in_dim + value_size

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (B, in_dim, T)
        Returns:
            Tensor: shape (B, in_dim + value_size, T)
        """
        B, C, T = x.shape
        # Permute to (B, T, in_dim) for linear layers.
        x_bt = x.permute(0, 2, 1)  # (B, T, in_dim)
        
        # Compute queries, keys, and values.
        query = self.linear_query(x_bt)   # (B, T, key_size)
        keys = self.linear_keys(x_bt)       # (B, T, key_size)
        values = self.linear_values(x_bt)   # (B, T, value_size)
        
        # Compute scaled dot-product attention.
        logits = torch.bmm(query, keys.transpose(1, 2))  # (B, T, T)
        logits = logits / self.sqrt_key_size

        # Create a causal mask so that each timestep t can only attend to time steps <= t.
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        logits.masked_fill_(mask, -float('inf'))

        # Compute attention weights.
        attn_weights = F.softmax(logits, dim=-1)  # (B, T, T)
        
        # Compute the read vector.
        read = torch.bmm(attn_weights, values)  # (B, T, value_size)
        
        # Concatenate the read vector with the original input.
        out = torch.cat([x_bt, read], dim=-1)  # (B, T, in_dim + value_size)
        
        # Permute back to (B, out_dim, T)
        return out.permute(0, 2, 1).contiguous()

# SNAIL Policy & Value Network
class SNAILPolicyValueNet(nn.Module):
    def __init__(
        self,
        action_dim=4,
        base_dim=256,
        policy_filters=32,
        policy_attn_dim=16, 
        value_filters=16,
        seq_len=800
    ):
        super().__init__()
        self.action_dim = action_dim
        # Use the CombinedEncoder to process the 6-channel input:
        self.combined_encoder = CombinedEncoder(base_dim=base_dim, scalar_out_dim=64)
        self.base_dim = base_dim
        self.seq_len = seq_len

        # Policy branch.
        self.policy_block1 = TCBlock(in_dim=base_dim, seq_len=seq_len, filters=policy_filters)
        self.attn1 = SnailAttentionBlock(
            in_dim=self.policy_block1.out_dim,
            key_size=policy_attn_dim,
            value_size=policy_attn_dim
        )
        self.policy_block2 = TCBlock(in_dim=self.attn1.out_dim, seq_len=seq_len, filters=policy_filters)
        self.attn2 = SnailAttentionBlock(
            in_dim=self.policy_block2.out_dim,
            key_size=policy_attn_dim,
            value_size=policy_attn_dim
        )
        self.policy_out_dim = self.attn2.out_dim
        self.policy_head = nn.Conv1d(self.policy_out_dim, action_dim, kernel_size=1)

        # Value branch.
        self.value_block1 = TCBlock(in_dim=base_dim, seq_len=seq_len, filters=value_filters)
        self.value_block2 = TCBlock(in_dim=self.value_block1.out_dim, seq_len=seq_len, filters=value_filters)
        self.value_out_dim = self.value_block2.out_dim
        self.value_head = nn.Conv1d(self.value_out_dim, 1, kernel_size=1)

        # For TRPO_FO compatibility.
        self.num_layers = 1
        self.hidden_size = base_dim

    def forward(self, x):
        """
        x: Tensor of shape (B, T, 6, H, W) with 6 channels (3 image, 3 scalar).
        Returns:
            policy_logits: (B, T, action_dim)
            values: (B, T)
        """
        B, T, C, H, W = x.shape
        # Process each time-step independently:
        x2 = x.view(B * T, C, H, W)
        # Get the base embedding using CombinedEncoder.
        feats = self.combined_encoder(x2)  # (B*T, base_dim)
        feats_1D = feats.view(B, T, self.base_dim).permute(0, 2, 1).contiguous()  # (B, base_dim, T)

        # Policy branch.
        p_out = self.policy_block1(feats_1D)
        p_out = self.attn1(p_out)
        p_out = self.policy_block2(p_out)
        p_out = self.attn2(p_out)
        logits_1D = self.policy_head(p_out)  # (B, action_dim, T)
        policy_logits = logits_1D.permute(0, 2, 1).contiguous()  # (B, T, action_dim)

        # Value branch.
        v_out = self.value_block1(feats_1D)
        v_out = self.value_block2(v_out)
        v_1D = self.value_head(v_out)  # (B, 1, T)
        values = v_1D.squeeze(1).permute(0, 1).contiguous()  # (B, T)

        return policy_logits, values

    def forward_rollout(self, x):
        return self.forward(x)

    def act_single_step(self, x):
        """
        Given a trajectory x (shape: (B, L, 6, H, W)), returns the logits and value at the final timestep.
        """
        logits, values = self.forward(x)
        return logits[:, -1, :], values[:, -1]

    def forward_with_state(self, x, dummy_state):
        """
        For TRPO_FO compatibility. Since SNAIL is feedforward, we return a dummy state.
        """
        logits, values = self.forward_rollout(x)
        batch_size = x.size(0)
        dummy = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        return logits, values, (dummy, dummy)

    def policy_parameters(self):
        return list(self.parameters())

    def value_parameters(self):
        return list(self.parameters())
