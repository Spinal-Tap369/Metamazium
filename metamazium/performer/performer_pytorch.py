# metamazium/performer/performer_pytorch.py

import math
import functools
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from einops import rearrange, repeat

from functools import partial
from contextlib import contextmanager

from local_attention import LocalAttention
from axial_positional_embedding import AxialPositionalEmbedding
from .reversible import ReversibleSequence, SequentialSequence

from distutils.version import LooseVersion

TORCH_GE_1_8_0 = LooseVersion(torch.__version__) >= LooseVersion('1.8.0')

# ------------------------------------------------------------------------
# Custom float_function to handle half->float casts (replacing apex.amp.float_function)
# ------------------------------------------------------------------------
def float_function(fn):
    """
    A custom decorator to emulate the old apex.amp.float_function,
    casting any half-precision cuda tensor to float before calling fn.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        def maybe_to_float(t):
            if isinstance(t, torch.cuda.HalfTensor) or (t.dtype == torch.float16):
                return t.float()
            return t

        new_args = [maybe_to_float(a) for a in args]
        new_kwargs = {k: maybe_to_float(v) for k, v in kwargs.items()}
        return fn(*new_args, **new_kwargs)
    return wrapper

# ------------------------------------------------------------------------
# Attempt to import FlashAttention
# ------------------------------------------------------------------------
try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False

# helpers

def exists(val):
    return val is not None

def empty(tensor):
    return tensor.numel() == 0

def default(val, d):
    return val if exists(val) else d

@contextmanager
def null_context():
    """A context manager that does nothing."""
    yield

def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val

def get_module_device(module):
    return next(module.parameters()).device

def find_modules(nn_module, type):
    return [m for m in nn_module.modules() if isinstance(m, type)]

class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, *args, **kwargs):
        return self.val

# token shifting helper and classes

def shift(t, amount, mask=None):
    if amount == 0:
        return t
    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)
    return F.pad(t, (0, 0, amount, -amount), value=0.)

class PreShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        segments = len(self.shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim=-1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        shifted = []
        for seg, amt in zip(segments_to_shift, self.shifts):
            shifted.append(shift(seg, amt, mask=mask))
        x = torch.cat((*shifted, *rest), dim=-1)
        return self.fn(x, **kwargs)

# ------------------------------------------------------------------------
# Projection & kernel functions (Performers)
# ------------------------------------------------------------------------

def softmax_kernel(
    data, *,
    projection_matrix,
    is_query,
    normalize_data=True,
    eps=1e-4,
    device=None
):
    b, h, *_ = data.shape
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.
    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps)
    else:
        data_dash = ratio * (torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()) + eps)

    return data_dash.type_as(data)

def generalized_kernel(
    data,
    *,
    projection_matrix,
    kernel_fn=nn.ReLU(),
    kernel_epsilon=0.001,
    normalize_data=True,
    device=None
):
    b, h, *_ = data.shape
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)
    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)

def orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    if TORCH_GE_1_8_0:
        q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    else:
        q, r = torch.qr(unstructured_block.cpu(), some=True)
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt(float(nb_columns)) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix

# ------------------------------------------------------------------------
# Linear attention (non-causal)
# ------------------------------------------------------------------------

def linear_attention(q, k, v):
    """
    Non-causal linear attention from the Performer paper.
    """
    k_cumsum = k.sum(dim=-2)
    D_inv = 1.0 / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out

# ------------------------------------------------------------------------
# FlashAttention-based Causal Implementation
# ------------------------------------------------------------------------

@float_function
def _flash_causal_forward(q, k, v):
    """
    Calls flash_attn_unpadded for causal attention,
    with no dropout in the attention matrix.
    Assumes q, k, v are float32.
    """
    return flash_attn_unpadded(q, k, v, causal=True, dropout_p=0.0)

def causal_linear_attention_noncuda(q, k, v, chunk_size=128, eps=1e-6):
    """
    Non-CUDA fallback for causal linear attention, memory-inefficient but functional.
    """
    last_k_cumsum = 0
    last_context_cumsum = 0
    outs = []

    for q_chunk, k_chunk, v_chunk in zip(*map(lambda t: t.chunk(chunk_size, dim=-2), (q, k, v))):
        k_cumsum = last_k_cumsum + k_chunk.cumsum(dim=-2)
        D_inv = 1.0 / torch.einsum('...nd,...nd->...n', q_chunk, k_cumsum.type_as(q_chunk) + eps)
        context = torch.einsum('...nd,...ne->...nde', k_chunk, v_chunk)
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)
        out = torch.einsum('...nde,...nd,...n->...ne', context_cumsum, q_chunk, D_inv)

        last_k_cumsum = k_cumsum[:, :, -1:]
        last_context_cumsum = context_cumsum[:, :, -1:]
        outs.append(out)

    return torch.cat(outs, dim=-2)

def causal_linear_attention_flash(q, k, v):
    """
    Causal linear attention leveraging flash_attn if installed.
    """
    if not FLASH_AVAILABLE:
        return causal_linear_attention_noncuda(q, k, v)

    autocast_enabled = torch.is_autocast_enabled()
    if autocast_enabled:
        with torch.cuda.amp.autocast(enabled=False):
            out = _flash_causal_forward(q, k, v)
    else:
        out = _flash_causal_forward(q, k, v)
    return out

# ------------------------------------------------------------------------
# FastAttention Module
# ------------------------------------------------------------------------

class FastAttention(nn.Module):
    """
    Handles both:
    - Non-causal linear attention (Performer)
    - Causal linear attention (via FlashAttention if available, else fallback)
    """
    def __init__(
        self,
        dim_heads,
        nb_features=None,
        ortho_scaling=0,
        causal=False,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        no_projection=False
    ):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(
            gaussian_orthogonal_random_matrix,
            nb_rows=self.nb_features,
            nb_columns=dim_heads,
            scaling=ortho_scaling
        )
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn
        self.no_projection = no_projection
        self.causal = causal

        # Decide which "causal" function to use
        if self.causal:
            self.causal_linear_fn = causal_linear_attention_flash
        else:
            self.causal_linear_fn = linear_attention

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device
        if self.no_projection:
            if self.causal:
                q = q.softmax(dim=-1)
                k = torch.exp(k)
            else:
                q = q.softmax(dim=-1)
                k = k.softmax(dim=-2)
        elif self.generalized_attention:
            create_kernel = partial(
                generalized_kernel,
                kernel_fn=self.kernel_fn,
                projection_matrix=self.projection_matrix,
                device=device
            )
            q, k = map(create_kernel, (q, k))
        else:
            create_kernel = partial(
                softmax_kernel,
                projection_matrix=self.projection_matrix,
                device=device
            )
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)

        attn_fn = self.causal_linear_fn if self.causal else linear_attention
        out = attn_fn(q, k, v)
        return out

# ------------------------------------------------------------------------
# A module for redrawing projection matrices on intervals
# ------------------------------------------------------------------------

class ProjectionUpdater(nn.Module):
    def __init__(self, instance, feature_redraw_interval):
        super().__init__()
        self.instance = instance
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer('calls_since_last_redraw', torch.tensor(0))

    def fix_projections_(self):
        self.feature_redraw_interval = None

    def redraw_projections(self):
        model = self.instance
        if not self.training:
            return
        if exists(self.feature_redraw_interval) and self.calls_since_last_redraw >= self.feature_redraw_interval:
            device = get_module_device(model)
            fas = find_modules(model, FastAttention)
            for fa in fas:
                fa.redraw_projection_matrix(device)
            self.calls_since_last_redraw.zero_()
            return
        self.calls_since_last_redraw += 1

    def forward(self, x):
        raise NotImplementedError

# ------------------------------------------------------------------------
# Classes: ReZero, Norms, etc.
# ------------------------------------------------------------------------

class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(1e-3))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.g

class PreScaleNorm(nn.Module):
    def __init__(self, dim, fn, eps=1e-5):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x, **kwargs):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        x = x / n * self.g
        return self.fn(x, **kwargs)

class PreLayerNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim=-1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        pieces = x.chunk(self.chunks, dim=self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in pieces], dim=self.dim)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0., activation=None, glu=False):
        super().__init__()
        activation = default(activation, nn.GELU)
        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v
        x = self.dropout(x)
        x = self.w2(x)
        return x

# ------------------------------------------------------------------------
# Attention classes
# ------------------------------------------------------------------------

def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rotary_pos_emb(q, k, sinu_pos):
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j=2)
    sin, cos = sinu_pos.unbind(dim=-2)
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j=2), (sin, cos))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal=False,
        heads=8,
        dim_head=64,
        local_heads=0,
        local_window_size=256,
        nb_features=None,
        feature_redraw_interval=1000,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        dropout=0.,
        no_projection=False,
        qkv_bias=False,
        attn_out_bias=True
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads

        self.fast_attention = FastAttention(
            dim_head,
            nb_features,
            causal=causal,
            generalized_attention=generalized_attention,
            kernel_fn=kernel_fn,
            no_projection=no_projection
        )

        self.heads = heads
        self.global_heads = heads - local_heads

        self.local_attn = None
        if local_heads > 0:
            self.local_attn = LocalAttention(
                window_size=local_window_size,
                causal=causal,
                autopad=True,
                dropout=dropout,
                look_forward=int(not causal),
                rel_pos_emb_config=(dim_head, local_heads)
            )

        self.to_q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=attn_out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb=None, context=None, mask=None, context_mask=None, **kwargs):
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads
        cross_attend = exists(context)

        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # separate local vs global heads
        (q_main, q_local), (k_main, k_local), (v_main, v_local) = map(
            lambda t: (t[:, :gh], t[:, gh:]),
            (q, k, v)
        )
        attn_outs = []

        # global heads => fast attention
        if not empty(q_main):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v_main.masked_fill_(~global_mask, 0.)

            if exists(pos_emb) and not cross_attend:
                q_main, k_main = apply_rotary_pos_emb(q_main, k_main, pos_emb)

            out = self.fast_attention(q_main, k_main, v_main)
            attn_outs.append(out)

        # local heads => local attention
        if not empty(q_local):
            assert not cross_attend, 'local attention is not compatible with cross-attention'
            out_local = self.local_attn(q_local, k_local, v_local, input_mask=mask)
            attn_outs.append(out_local)

        # combine global+local heads
        out = torch.cat(attn_outs, dim=1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)

class SelfAttention(Attention):
    def forward(self, *args, context=None, **kwargs):
        # self-attention cannot receive external context
        assert not exists(context), 'self attention should not receive context'
        return super().forward(*args, **kwargs)

class CrossAttention(Attention):
    def forward(self, *args, context=None, **kwargs):
        # cross attention must receive external context
        assert exists(context), 'cross attention should receive context'
        return super().forward(*args, context=context, **kwargs)

# positional embeddings

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)

class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum('i,j->ij', position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('emb', emb)

    def forward(self, x):
        return self.emb[None, :x.shape[1], :].to(x)

# ------------------------------------------------------------------------
# Performer class definitions
# ------------------------------------------------------------------------

class Performer(nn.Module):
    """
    Stacks multiple layers of (Self)Attention + FeedForward,
    with potential local attention heads, cross attention, etc.
    """
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        local_attn_heads=0,
        local_window_size=256,
        causal=False,
        ff_mult=4,
        nb_features=None,
        feature_redraw_interval=1000,
        reversible=False,
        ff_chunks=1,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        use_scalenorm=False,
        use_rezero=False,
        ff_glu=False,
        ff_dropout=0.,
        attn_dropout=0.,
        cross_attend=False,
        no_projection=False,
        auto_check_redraw=True,
        qkv_bias=True,
        attn_out_bias=True,
        shift_tokens=False
    ):
        super().__init__()
        layers = nn.ModuleList([])
        local_attn_heads = cast_tuple(local_attn_heads)
        local_attn_heads = local_attn_heads * depth if len(local_attn_heads) == 1 else local_attn_heads
        assert len(local_attn_heads) == depth, 'local_attn_heads must have length equal to depth'
        assert all(0 <= n <= heads for n in local_attn_heads), 'local attn heads must be <= total heads'

        if use_scalenorm:
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, dim)

        for _, lheads in zip(range(depth), local_attn_heads):
            attn = SelfAttention(
                dim,
                causal=causal,
                heads=heads,
                dim_head=dim_head,
                local_heads=lheads,
                local_window_size=local_window_size,
                nb_features=nb_features,
                feature_redraw_interval=feature_redraw_interval,
                generalized_attention=generalized_attention,
                kernel_fn=kernel_fn,
                dropout=attn_dropout,
                no_projection=no_projection,
                qkv_bias=qkv_bias,
                attn_out_bias=attn_out_bias
            )
            ff = Chunk(
                ff_chunks,
                FeedForward(dim, mult=ff_mult, dropout=ff_dropout, glu=ff_glu),
                along_dim=1
            )

            if shift_tokens:
                shift = (0, 1) if causal else (-1, 0, 1)
                attn = PreShiftTokens(shift, attn)
                ff = PreShiftTokens(shift, ff)

            attn = wrapper_fn(attn)
            ff = wrapper_fn(ff)

            layers.append(nn.ModuleList([attn, ff]))

            # cross attention
            if cross_attend:
                layers.append(nn.ModuleList([
                    wrapper_fn(CrossAttention(
                        dim,
                        heads=heads,
                        dim_head=dim_head,
                        nb_features=nb_features,
                        generalized_attention=generalized_attention,
                        kernel_fn=kernel_fn,
                        dropout=attn_dropout,
                        no_projection=no_projection,
                        qkv_bias=qkv_bias,
                        attn_out_bias=attn_out_bias
                    )),
                    wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult=ff_mult, dropout=ff_dropout, glu=ff_glu), along_dim=1))
                ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence

        route_attn = ((True, False),) * depth * (2 if cross_attend else 1)
        route_context = ((False, False), (True, False)) * depth
        attn_route_map = {'mask': route_attn, 'pos_emb': route_attn}
        context_route_map = {}
        if cross_attend:
            context_route_map = {'context': route_context, 'context_mask': route_context}

        self.net = execute_type(layers, args_route={**attn_route_map, **context_route_map})

        self.auto_check_redraw = auto_check_redraw
        self.proj_updater = ProjectionUpdater(self.net, feature_redraw_interval)

    def fix_projection_matrices_(self):
        self.proj_updater.feature_redraw_interval = None

    def forward(self, x, **kwargs):
        if self.auto_check_redraw:
            self.proj_updater.redraw_projections()
        return self.net(x, **kwargs)

class PerformerLM(nn.Module):
    """
    Language modeling architecture on top of the Performer.
    """
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        dim,
        depth,
        heads,
        dim_head=64,
        local_attn_heads=0,
        local_window_size=256,
        causal=False,
        ff_mult=4,
        nb_features=None,
        feature_redraw_interval=1000,
        reversible=False,
        ff_chunks=1,
        ff_glu=False,
        emb_dropout=0.,
        ff_dropout=0.,
        attn_dropout=0.,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        use_scalenorm=False,
        use_rezero=False,
        cross_attend=False,
        no_projection=False,
        tie_embed=False,
        rotary_position_emb=True,
        axial_position_emb=False,
        axial_position_shape=None,
        auto_check_redraw=True,
        qkv_bias=False,
        attn_out_bias=False,
        shift_tokens=False
    ):
        super().__init__()
        local_attn_heads = cast_tuple(local_attn_heads)

        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)

        # Choose positional embedding strategy
        if rotary_position_emb:
            self.pos_emb = FixedPositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = FixedPositionalEmbedding(dim_head, max_seq_len)
        elif axial_position_emb:
            axial_position_shape = default(axial_position_shape, (math.ceil(max_seq_len / 64), 64))
            self.pos_emb = AxialPositionalEmbedding(dim, axial_position_shape)
            self.layer_pos_emb = Always(None)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = Always(None)

        self.dropout = nn.Dropout(emb_dropout)

        self.performer = Performer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            local_attn_heads=local_attn_heads,
            local_window_size=local_window_size,
            causal=causal,
            ff_mult=ff_mult,
            nb_features=nb_features,
            feature_redraw_interval=feature_redraw_interval,
            reversible=reversible,
            ff_chunks=ff_chunks,
            generalized_attention=generalized_attention,
            kernel_fn=kernel_fn,
            use_scalenorm=use_scalenorm,
            use_rezero=use_rezero,
            ff_glu=ff_glu,
            ff_dropout=ff_dropout,
            attn_dropout=attn_dropout,
            cross_attend=cross_attend,
            no_projection=no_projection,
            auto_check_redraw=auto_check_redraw,
            qkv_bias=qkv_bias,
            attn_out_bias=attn_out_bias,
            shift_tokens=shift_tokens
        )
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, num_tokens) if not tie_embed else None

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, x, return_encodings=False, **kwargs):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be <= {self.max_seq_len}'

        # token + position embeddings
        x = self.token_emb(x)
        x += self.pos_emb(x)
        x = self.dropout(x)

        # performer layers
        layer_pos = self.layer_pos_emb(x)
        x = self.performer(x, pos_emb=layer_pos, **kwargs)

        # final norm + optional final projection
        x = self.norm(x)
        if return_encodings:
            return x
        if exists(self.to_out):
            return self.to_out(x)
        # fallback if tie_embed is True
        return x @ self.token_emb.weight.t()
