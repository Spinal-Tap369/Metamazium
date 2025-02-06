# metamazium/snail_performer/snail_model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from metamazium.snail_performer.cnn_encoder import CNNEncoder

from metamazium.performer.performer_pytorch import SelfAttention


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


class SnailSelfAttnBlock(nn.Module):
    """
    - Projects input (in_dim -> embed_dim)
    - Applies SelfAttention(dim=embed_dim, causal=True, ...)
    - Concatenates [ original_input, attention_output ]
    => shape (B, in_dim+embed_dim, T)
    """
    def __init__(self, in_dim, embed_dim, num_heads=1, dropout=0.0, local_heads=0):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # linear: in_dim -> embed_dim
        self.in_proj = nn.Linear(in_dim, embed_dim, bias=False)

        # Performer SelfAttention
        self.attn = SelfAttention(
            dim=embed_dim,
            heads=num_heads,
            dim_head=max(1, embed_dim // num_heads),
            causal=True,
            local_heads=local_heads,  # rename from local_attn_heads -> local_heads
            dropout=dropout
        )
        self.out_dim = in_dim + embed_dim

    def forward(self, x):
        # x: (B, in_dim, T)
        B, C, T = x.shape
        x_bt_c = x.permute(0, 2, 1)   # => (B,T,C)
        proj = self.in_proj(x_bt_c)  # => (B,T,embed_dim)
        attn_out = self.attn(proj)   # => (B,T,embed_dim)

        cat_bt = torch.cat([x_bt_c, attn_out], dim=-1)  # => (B,T,C+embed_dim)
        return cat_bt.permute(0,2,1).contiguous()       # => (B,C+embed_dim,T)


class SNAILPolicyValueNet(nn.Module):
    def __init__(
        self,
        action_dim=4,
        base_dim=256,
        policy_filters=32,
        policy_attn_dim=16,
        value_filters=16,
        seq_len=800,
        num_policy_attn=2,
        nb_features=64,
        num_heads=1
    ):
        super().__init__()
        self.action_dim = action_dim
        self.cnn_encoder = CNNEncoder()
        self.base_dim = base_dim
        self.seq_len = seq_len

        # -------------- Policy --------------
        self.policy_block1 = TCBlock(in_dim=base_dim, seq_len=seq_len, filters=policy_filters)
        self.attn1 = SnailSelfAttnBlock(
            in_dim=self.policy_block1.out_dim,
            embed_dim=policy_attn_dim,
            num_heads=num_heads,
            local_heads=0    # or set local_heads if you want local attention
        )
        self.policy_block2 = TCBlock(in_dim=self.attn1.out_dim, seq_len=seq_len, filters=policy_filters)
        self.attn2 = SnailSelfAttnBlock(
            in_dim=self.policy_block2.out_dim,
            embed_dim=policy_attn_dim,
            num_heads=num_heads,
            local_heads=0
        )
        self.policy_out_dim = self.attn2.out_dim
        self.policy_head = nn.Conv1d(self.policy_out_dim, action_dim, kernel_size=1)

        # -------------- Value --------------
        self.value_block1 = TCBlock(in_dim=base_dim, seq_len=seq_len, filters=value_filters)
        self.value_block2 = TCBlock(in_dim=self.value_block1.out_dim, seq_len=seq_len, filters=value_filters)
        self.value_out_dim = self.value_block2.out_dim
        self.value_head = nn.Conv1d(self.value_out_dim, 1, kernel_size=1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x2 = x.view(B*T, C, H, W)
        feats = self.cnn_encoder(x2)
        feats_1D = feats.view(B, T, self.base_dim).permute(0,2,1).contiguous()

        # policy
        p_out = self.policy_block1(feats_1D)
        p_out = self.attn1(p_out)
        p_out = self.policy_block2(p_out)
        p_out = self.attn2(p_out)
        logits_1D = self.policy_head(p_out)
        policy_logits = logits_1D.permute(0,2,1).contiguous()  # => (B,T,action_dim)

        # value
        v_out = self.value_block1(feats_1D)
        v_out = self.value_block2(v_out)
        v_1D = self.value_head(v_out)
        values = v_1D.squeeze(1).permute(0,1).contiguous()

        return policy_logits, values

    def forward_rollout(self, x):
        return self.forward(x)
