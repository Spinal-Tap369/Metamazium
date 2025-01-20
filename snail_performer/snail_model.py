# snail_performer/snail_model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_encoder import CNNEncoder

def orthogonal_random_matrix(nb_rows, nb_columns, device=None, scaling=1):
    """
    Generate an orthogonal random matrix of shape (nb_rows, nb_columns).
    """
    unstructured_block = torch.randn((nb_rows, nb_columns), device=device)
    q, _ = torch.linalg.qr(unstructured_block, mode='reduced')
    if scaling == 1:
        return math.sqrt(nb_columns) * q
    else:
        row_norms = unstructured_block.norm(dim=1)
        diag_mat = torch.diag(row_norms)
        return diag_mat @ q

def performer_feature_map(x, proj_matrix, eps=1e-6):
    """
    Apply exponent-based random feature mapping to input tensor.
    """
    x_proj = torch.einsum('bthd,fd->bthf', x, proj_matrix)
    return torch.exp(x_proj) + eps

class DenseBlock(nn.Module):
    """
    Dense block with gated dilated convolutions.
    """
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

class TCBlock(nn.Module):
    """
    Temporal convolutional block with multiple DenseBlocks for multi-scale feature extraction.
    """
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

class PerformerAttnBlock(nn.Module):
    """
    Performer-like attention block for approximating global attention.
    """
    def __init__(self, in_dim, embed_dim, num_heads=1, nb_features=64):
        super().__init__()
        self.num_heads = num_heads
        self.nb_features = nb_features
        self.d_head = embed_dim // num_heads

        self.q_proj = nn.Linear(in_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        proj = orthogonal_random_matrix(nb_features, self.d_head, scaling=1)
        self.register_buffer("proj_matrix", proj)
        self.out_dim = in_dim + embed_dim

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, in_dim, T)
        Returns:
            Tensor of shape (B, in_dim+embed_dim, T)
        """
        B, C, T = x.shape
        x_bt_c = x.permute(0, 2, 1)  # (B, T, in_dim)
        Q = self.q_proj(x_bt_c)
        K = self.k_proj(x_bt_c)
        V = self.v_proj(x_bt_c)

        # Reshape for multi-head processing
        Q = Q.view(B, T, self.num_heads, self.d_head)
        K = K.view(B, T, self.num_heads, self.d_head)
        V = V.view(B, T, self.num_heads, self.d_head)

        # Apply random feature mapping
        Q_prime = performer_feature_map(Q, self.proj_matrix)
        K_prime = performer_feature_map(K, self.proj_matrix)

        K_sum = K_prime.sum(dim=1, keepdim=True)
        KV = torch.einsum('bthf,bthd->bhfd', K_prime, V)
        KV_sum = KV.unsqueeze(1)

        Q_expand = Q_prime.unsqueeze(3)
        denom = torch.einsum('bthzf,bzhf->bthz', Q_expand, K_sum).clamp_min(1e-6)
        numer = torch.einsum('bthzf,bzhfd->bthd', Q_expand, KV_sum)

        attn_out = numer / denom
        attn_out = attn_out.reshape(B, T, self.num_heads*self.d_head)
        out2 = self.out_proj(attn_out)

        # Concatenate attention output with input features
        cat_bt = torch.cat([x_bt_c, out2], dim=-1)
        return cat_bt.permute(0, 2, 1).contiguous()

class SNAILPolicyValueNet(nn.Module):
    """
    SNAIL network with Performer attention blocks for policy and value estimation.
    """
    def __init__(self,
                 action_dim=4,
                 base_dim=256,
                 policy_filters=32,
                 policy_attn_dim=16,
                 value_filters=16,
                 seq_len=800,
                 num_policy_attn=2,
                 nb_features=64,
                 num_heads=1):
        super().__init__()
        self.action_dim = action_dim
        self.cnn_encoder = CNNEncoder()
        self.base_dim = base_dim
        self.seq_len = seq_len

        # Policy branch initialization
        self.policy_block1 = TCBlock(in_dim=base_dim, seq_len=seq_len, filters=policy_filters)
        self.attn1 = PerformerAttnBlock(self.policy_block1.out_dim, policy_attn_dim,
                                        num_heads=num_heads, nb_features=nb_features)
        self.policy_block2 = TCBlock(in_dim=self.attn1.out_dim, seq_len=seq_len, filters=policy_filters)
        self.attn2 = PerformerAttnBlock(self.policy_block2.out_dim, policy_attn_dim,
                                        num_heads=num_heads, nb_features=nb_features)
        self.policy_out_dim = self.attn2.out_dim
        self.policy_head = nn.Conv1d(self.policy_out_dim, action_dim, kernel_size=1)

        # Value branch initialization
        self.value_block1 = TCBlock(in_dim=base_dim, seq_len=seq_len, filters=value_filters)
        self.value_block2 = TCBlock(in_dim=self.value_block1.out_dim, seq_len=seq_len, filters=value_filters)
        self.value_out_dim = self.value_block2.out_dim
        self.value_head = nn.Conv1d(self.value_out_dim, 1, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B,T,6,H,W)
        Returns:
            policy_logits: Tensor of shape (B,T,action_dim)
            values: Tensor of shape (B,T)
        """
        B, T, C, H, W = x.shape
        # Encode each observation with CNN
        x2 = x.view(B*T, C, H, W)
        feats = self.cnn_encoder(x2)
        feats_1D = feats.view(B, T, self.base_dim).permute(0, 2, 1).contiguous()

        # Policy branch processing
        p_out = self.policy_block1(feats_1D)
        p_out = self.attn1(p_out)
        p_out = self.policy_block2(p_out)
        p_out = self.attn2(p_out)
        logits_1D = self.policy_head(p_out)
        policy_logits = logits_1D.permute(0, 2, 1).contiguous()

        # Value branch processing
        v_out = self.value_block1(feats_1D)
        v_out = self.value_block2(v_out)
        v_1D = self.value_head(v_out)
        values = v_1D.squeeze(1).permute(0, 1).contiguous()

        return policy_logits, values

    def forward_rollout(self, x):
        # For consistency with PPO code
        return self.forward(x)
