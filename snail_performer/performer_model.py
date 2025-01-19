# snail_performer/performer_model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_encoder import CNNEncoder

def orthogonal_matrix_chunk(cols, device):
    """Generates a random orthonormal matrix chunk of shape (cols, cols)."""
    block = torch.randn((cols, cols), device=device)
    q, _ = torch.qr(block, some=True)
    return q.t()

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, device=None):
    """
    Generates a random Gaussian orthonormal matrix with specified scaling.
    
    scaling=0: scales each row by its norm.
    scaling=1: scales entire matrix by sqrt(nb_columns).
    """
    unstructured_block = torch.randn((nb_rows, nb_columns), device=device)
    q, _ = torch.linalg.qr(unstructured_block, mode='reduced')

    if scaling == 0:
        row_norms = unstructured_block.norm(dim=1)
        diag_mat = torch.diag(row_norms)
        return diag_mat @ q
    elif scaling == 1:
        scale = math.sqrt(float(nb_columns))
        return scale * q
    else:
        raise ValueError(f"Invalid scaling={scaling}, must be 0 or 1.")

def softmax_kernel(data, projection_matrix, is_query, eps=1e-4):
    """
    Applies softmax kernel random feature mapping to input data.
    
    data: (b, h, s, d)
    projection_matrix: (f, d)
    returns: (b, h, s, f)
    """
    b, h, s, d = data.shape
    f = projection_matrix.shape[0]

    data_normalizer = d ** -0.25
    ratio = f ** -0.5

    proj = projection_matrix.unsqueeze(0).unsqueeze(0).expand(b, h, f, d).to(data.device)

    data = data_normalizer * data
    data_dash = torch.einsum('bhsd,bhfd->bhsf', data, proj)

    diag_data = (data ** 2).sum(dim=-1) * (data_normalizer ** 2) / 2.0
    diag_data = diag_data.unsqueeze(-1)

    if is_query:
        data_dash = data_dash - torch.amax(data_dash, dim=-1, keepdim=True).detach()

    data_dash = torch.exp(data_dash - diag_data) + eps
    return ratio * data_dash

def causal_dot_product(q, k, v, eps=1e-6):
    """
    Naive causal dot product computation for Performer attention.
    
    q, k, v: (b, h, s, d)
    returns: (b, h, s, d)
    """
    b, h, s, d = q.shape
    out = []
    k_cumsum  = torch.zeros((b, h, d), device=q.device, dtype=q.dtype)
    kv_cumsum = torch.zeros((b, h, d, d), device=q.device, dtype=q.dtype)

    for t in range(s):
        qt = q[:, :, t, :]
        kt = k[:, :, t, :]
        vt = v[:, :, t, :]

        kt_vt = kt.unsqueeze(-1) * vt.unsqueeze(-2)
        kv_cumsum += kt_vt
        k_cumsum  += kt

        denominator = (qt * k_cumsum).sum(dim=-1, keepdim=True).clamp_min(eps)
        numerator  = torch.einsum('bhd,bhde->bhe', qt, kv_cumsum)
        out_t = numerator / denominator
        out.append(out_t.unsqueeze(2))

    return torch.cat(out, dim=2)

class MinimalPerformerAttention(nn.Module):
    """Minimal Performer attention module with fixed B approach."""
    def __init__(self, dim, heads=8, dim_head=64, nb_features=None, causal=False, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.causal = causal
        self.nb_features = nb_features or int(dim_head * math.log(dim_head))

        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.post_proj = nn.Linear(self.nb_features, self.dim_head, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        proj = gaussian_orthogonal_random_matrix(
            nb_rows=self.nb_features,
            nb_columns=dim_head,
            scaling=0,
            device=None
        )
        self.register_buffer('proj_matrix', proj)

    def forward(self, x):
        """
        Applies Performer attention to input tensor.
        
        x: (b, s, dim)
        returns: (b, s, dim)
        """
        b, s, d = x.shape
        h = self.heads

        q = self.to_q(x).view(b, h, s, self.dim_head)
        k = self.to_k(x).view(b, h, s, self.dim_head)
        v = self.to_v(x).view(b, h, s, self.dim_head)

        q_dash = softmax_kernel(q, self.proj_matrix, is_query=True)
        k_dash = softmax_kernel(k, self.proj_matrix, is_query=False)

        if not self.causal:
            k_sum = k_dash.sum(dim=2)
            denom = torch.einsum('bhsf,bhf->bhs', q_dash, k_sum).clamp_min(1e-8)
            denom_inv = 1.0 / denom

            context = torch.einsum('bhsf,bhsd->bhfd', k_dash, v)
            out = torch.einsum('bhsf,bhfd,bhs->bhsd', q_dash, context, denom_inv)
        else:
            out = causal_dot_product(q_dash, k_dash, v)

        b_h_s = b * h * s
        f = self.nb_features
        if out.numel() != (b_h_s * f):
            raise RuntimeError(
                f"Expected 'out' to have {b_h_s*f} elements, but got {out.numel()}."
            )

        # Use reshape() instead of view()
        out_2d = out.reshape(b_h_s, f)
        out_2d = self.post_proj(out_2d)
        out = out_2d.view(b, h, s, self.dim_head)

        out = out.permute(0, 2, 1, 3).reshape(b, s, h * self.dim_head)
        out = self.to_out(out)
        return self.dropout(out)

class DenseCausalConvBlock(nn.Module):
    """1D causal convolution with gating that increases channel dimension."""
    def __init__(self, in_dim, filters, kernel_size=2, dilation=1):
        super().__init__()
        self.conv_f = nn.Conv1d(
            in_dim, filters,
            kernel_size=kernel_size, stride=1,
            dilation=dilation,
            padding=dilation*(kernel_size-1), bias=True
        )
        self.conv_g = nn.Conv1d(
            in_dim, filters,
            kernel_size=kernel_size, stride=1,
            dilation=dilation,
            padding=dilation*(kernel_size-1), bias=True
        )
        self.out_dim = in_dim + filters
        self.in_dim = in_dim
        self.filters = filters

    def forward(self, x):
        """
        Applies causal convolution with gating to input.
        
        x: (b, seq, in_dim)
        returns: (b, seq, in_dim + filters)
        """
        b, seq, d = x.shape
        if d != self.in_dim:
            raise ValueError(f"Expected input dim {self.in_dim}, got {d}.")

        x_t = x.transpose(1, 2)  # (b, in_dim, seq)
        xf = self.conv_f(x_t)[:, :, :seq]  # (b, filters, seq)
        xg = self.conv_g(x_t)[:, :, :seq]  # (b, filters, seq)

        act = torch.tanh(xf) * torch.sigmoid(xg)  # (b, filters, seq)
        act = act.transpose(1, 2)  # (b, seq, filters)
        return torch.cat([x, act], dim=-1)  # (b, seq, in_dim + filters)

class PerformerAttnBlock(nn.Module):
    """Block combining Performer attention with input concatenation."""
    def __init__(self, in_dim, heads=4, dim_head=32, nb_features=None, causal=False, dropout=0.0):
        super().__init__()
        self.attn = MinimalPerformerAttention(
            dim=in_dim,
            heads=heads,
            dim_head=dim_head,
            nb_features=nb_features,
            causal=causal,
            dropout=dropout
        )
        self.out_dim = in_dim * 2
        self.in_dim = in_dim

    def forward(self, x):
        """
        Applies attention and concatenates with input.
        
        x: (b, seq, in_dim)
        returns: (b, seq, in_dim * 2)
        """
        attn_out = self.attn(x)
        return torch.cat([x, attn_out], dim=-1)

class SNAILPerformerPolicy(nn.Module):
    """
    SNAIL + Performer-based policy.
    """
    def __init__(
        self,
        action_dim=4,
        base_dim=256,
        num_tc_blocks=2,
        tc_filters=32,
        attn_heads=4,
        attn_dim_head=32,
        attn_dropout=0.1,
        nb_features=None,
        causal=False
    ):
        super().__init__()
        self.cnn_encoder = CNNEncoder()  # Now expects 6 input channels
        self.init_dim = base_dim

        current_dim = self.init_dim
        snail_blocks = []

        for i in range(num_tc_blocks):
            block_tc = DenseCausalConvBlock(
                in_dim=current_dim,
                filters=tc_filters,
                kernel_size=2,
                dilation=2**i
            )
            snail_blocks.append(block_tc)
            current_dim = block_tc.out_dim

            block_attn = PerformerAttnBlock(
                in_dim=current_dim,
                heads=attn_heads,
                dim_head=attn_dim_head,
                nb_features=nb_features,
                causal=causal,
                dropout=attn_dropout
            )
            snail_blocks.append(block_attn)
            current_dim = block_attn.out_dim

        self.snail_blocks = nn.ModuleList(snail_blocks)
        self.final_dim = current_dim

        self.norm = nn.LayerNorm(self.final_dim)
        self.policy_head = nn.Linear(self.final_dim, action_dim)
        self.value_head = nn.Linear(self.final_dim, 1)

    def forward(self, obs_seq):
        """
        obs_seq shape: (B, T, 6, H, W).
        Returns: policy_logits (B,T,action_dim), values (B,T).
        """
        B, T, C, H, W = obs_seq.shape  # C=6
        x = obs_seq.reshape((B*T, C, H, W))  # Use reshape instead of view
        feats = self.cnn_encoder(x)  # => (B*T, 256)
        feats = feats.reshape(B, T, -1)  # Use reshape instead of view

        out = feats
        for blk in self.snail_blocks:
            out = blk(out)

        out = self.norm(out)
        policy_logits = self.policy_head(out)
        values = self.value_head(out).squeeze(-1)
        return policy_logits, values

    @torch.no_grad()
    def act_online_sequence(self, obs_seq):
        """
        Fully online SNAIL approach:
        - obs_seq shape: (1, t, 6, H, W).
        - Runs forward(...) on entire sequence.
        - Returns last step's distribution => shape (1, action_dim).
        """
        logits, _ = self.forward(obs_seq)
        last_logits = logits[:, -1, :]
        return last_logits

    def reset_sequence_states(self):
        pass  # Not required for this policy model

    def reset_lstm_states(self, batch_size=1):
        pass  # Not required for this policy model
