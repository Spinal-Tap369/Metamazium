# metamazium/snail_performer/snail_c.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from metamazium.performer.performer_pytorch import SelfAttention

class OmniglotEmbedding(nn.Module):
    """
    Standard 4-layer CNN for Omniglot few-shot tasks (grayscale 28x28).
    Repeats the following block four times:
      { 3x3 conv (64 channels), batch norm, ReLU, 2x2 max pool },
    and then applies a single fully-connected layer to output a 64-dimensional feature vector.
    """
    def __init__(self, out_dim=64):
        super(OmniglotEmbedding, self).__init__()
        layers = []
        in_channels = 1  # Omniglot is grayscale
        hidden_channels = 64
        for _ in range(4):
            layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            in_channels = hidden_channels
        self.cnn = nn.Sequential(*layers)
        self.fc = nn.Linear(64, out_dim)
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CasualConv1d(nn.Module):
    """
    1D causal convolution with kernel_size=2.
    Pads appropriately and then slices off the last d timesteps to enforce causality.
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
        super(CasualConv1d, self).__init__()
        self.dilation = dilation
        padding = dilation * (kernel_size - 1)
        self.conv1d = nn.Conv1d(in_channels, out_channels,
                                kernel_size=kernel_size, dilation=dilation,
                                padding=padding)
    def forward(self, x):
        out = self.conv1d(x)
        return out[:, :, :-self.dilation]

class DenseBlock(nn.Module):
    """
    Applies two causal convolutions (xf and xg), applies tanh and sigmoid,
    multiplies elementwise, then concatenates with the input.
    """
    def __init__(self, in_channels, dilation, filters, kernel_size=2):
        super(DenseBlock, self).__init__()
        self.casualconv1 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)
        self.casualconv2 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)
    def forward(self, x):
        xf = self.casualconv1(x)
        xg = self.casualconv2(x)
        activations = torch.tanh(xf) * torch.sigmoid(xg)
        return torch.cat((x, activations), dim=1)

class TCBlock(nn.Module):
    """
    Temporal Convolution Block: creates a list of DenseBlocks with exponentially increasing dilation.
    The number of blocks is ceil(log2(seq_length)).
    """
    def __init__(self, in_channels, seq_length, filters):
        super(TCBlock, self).__init__()
        num_blocks = int(math.ceil(math.log(seq_length, 2)))
        dense_blocks = nn.ModuleList([
            DenseBlock(in_channels + i * filters, dilation=2**(i+1), filters=filters)
            for i in range(num_blocks)
        ])
        self.dense_blocks = dense_blocks
        self.out_channels = in_channels + num_blocks * filters
    def forward(self, x):
        # x: (N, T, in_channels) → transpose to (N, in_channels, T)
        x = x.transpose(1, 2)
        for block in self.dense_blocks:
            x = block(x)
        return x.transpose(1, 2)

class AttentionBlock(nn.Module):
    """
    Standard Attention Block as described in the paper.
    It computes query, keys, and values using linear projections,
    applies a causal mask and softmax over the scaled dot products,
    and concatenates the computed read vector with the original input.
    """
    def __init__(self, in_channels, key_size, value_size):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)
    def forward(self, x):
        # x: (N, T, in_channels)
        N, T, _ = x.shape
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        query = self.linear_query(x)    # (N, T, key_size)
        keys = self.linear_keys(x)        # (N, T, key_size)
        values = self.linear_values(x)    # (N, T, value_size)
        logits = torch.bmm(query, keys.transpose(1, 2))  # (N, T, T)
        logits.masked_fill_(mask, -float('inf'))
        attn = F.softmax(logits / self.sqrt_key_size, dim=-1)  # (N, T, T)
        read = torch.bmm(attn, values)    # (N, T, value_size)
        return torch.cat((x, read), dim=2)  # (N, T, in_channels + value_size)



class SNAILFewShot(nn.Module):
    """
    SNAIL for few-shot classification following the paper.
    
    For an N-way, K-shot episode:
      - The model receives NK support examples (each with its one-hot label)
        in random order, followed by a single query example (with its label omitted).
      - Thus, the sequence length T = NK + 1.
    
    Architecture:
      1. Input Projection: Map each input (concatenated embedding and one-hot label of dimension 64+N) to 64 dimensions.
      2. AttentionBlock(64, 32) → output dimension becomes 64 + 32 = 96.
      3. TCBlock(T, 128) → then project to 256.
      4. AttentionBlock(256, 128) → output dimension becomes 256 + 128 = 384.
      5. TCBlock(T, 128) → then project to 512.
      6. AttentionBlock(512, 256) → output dimension becomes 512 + 256 = 768.
      7. Final 1×1 convolution maps the final representation to num_classes outputs.
    """
    def __init__(self, num_classes, seq_len, device="cuda", embedding_type="omniglot"):
        super(SNAILFewShot, self).__init__()
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.device = device
        
        # Embedding network
        if embedding_type == "omniglot":
            self.embedding_net = OmniglotEmbedding(out_dim=64)
        elif embedding_type == "miniimagenet":
            from metamazium.snail_performer.snail_c import MiniImageNetEmbedding
            self.embedding_net = MiniImageNetEmbedding(out_dim=384, in_size=84)
        else:
            raise ValueError("Unknown embedding_type: " + embedding_type)
        
        # Input Projection: Concatenate embedding (64-d) with one-hot label (num_classes) and project to 64-d.
        self.input_proj = nn.Linear(64 + num_classes, 64)
        
        # Block 1: AttentionBlock(64, 32)
        self.attn1 = AttentionBlock(64, key_size=32, value_size=32)
        # Output dimension becomes 64 + 32 = 96.
        
        # Block 2: TCBlock(T, 128) followed by projection to 256.
        self.tc1 = TCBlock(96, seq_len, filters=128)
        self.proj_after_tc1 = nn.Linear(self.tc1.out_channels, 256)
        
        # Block 3: AttentionBlock(256, 128)
        self.attn2 = AttentionBlock(256, key_size=128, value_size=128)
        # Output becomes 256 + 128 = 384.
        
        # Block 4: TCBlock(T, 128) followed by projection to 512.
        self.tc2 = TCBlock(384, seq_len, filters=128)
        self.proj_after_tc2 = nn.Linear(self.tc2.out_channels, 512)
        
        # Block 5: AttentionBlock(512, 256)
        self.attn3 = AttentionBlock(512, key_size=256, value_size=256)
        # Output becomes 512 + 256 = 768.
        
        # Final 1×1 convolution to map to num_classes outputs.
        self.final_conv = nn.Conv1d(768, num_classes, kernel_size=1)
        
    def forward_single_query(self, support_images, support_labels, query_image):
        """
        Process a single query.
          support_images: (N*K, C, H, W)
          support_labels: (N*K, N) one-hot vectors.
          query_image: (1, C, H, W)
        """
        # 1. Embed support and query images.
        sup_feat = self.embedding_net(support_images)   # (N*K, feature_dim)
        qry_feat = self.embedding_net(query_image)        # (1, feature_dim)
        
        # 2. Concatenate support embedding with its one-hot label.
        sup_cat = torch.cat([sup_feat, support_labels], dim=1)  # (N*K, feature_dim + N)
        sup_proj = self.input_proj(sup_cat)              # (N*K, 64)
        
        # 3. For query, append a zero label and project.
        zero_label = torch.zeros(1, self.num_classes, device=query_image.device)
        qry_cat = torch.cat([qry_feat, zero_label], dim=1)  # (1, feature_dim + N)
        qry_proj = self.input_proj(qry_cat)                # (1, 64)
        
        # 4. Build the sequence: support examples followed by query.
        # Sequence shape: (T, 64) where T = N*K + 1.
        seq = torch.cat([sup_proj, qry_proj], dim=0).unsqueeze(0)  # (1, T, 64)
        # Do not permute: AttentionBlock and TCBlock expect (N, T, channels)
        
        # 5. Block 1: AttentionBlock(64, 32)
        out = self.attn1(seq)   # (1, T, 64+32) → (1, T, 96)
        
        # 6. Block 2: TCBlock(T, 128), then project to 256.
        out = self.tc1(out)     # (1, T, tc1_channels)
        B, T, C = out.shape
        out = out.reshape(-1, C)           # (T, tc1_channels)
        out = self.proj_after_tc1(out)       # (T, 256)
        out = out.reshape(B, T, 256)          # (1, T, 256)
        
        # 7. Block 3: AttentionBlock(256, 128)
        out = self.attn2(out)   # (1, T, 256+128) → (1, T, 384)
        
        # 8. Block 4: TCBlock(T, 128), then project to 512.
        out = self.tc2(out)     # (1, T, tc2_channels)
        B, T, C = out.shape
        out = out.reshape(-1, C)           # (T, tc2_channels)
        out = self.proj_after_tc2(out)       # (T, 512)
        out = out.reshape(B, T, 512)          # (1, T, 512)
        
        # 9. Block 5: AttentionBlock(512, 256)
        out = self.attn3(out)   # (1, T, 512+256) → (1, T, 768)
        
        # 10. Before final convolution, transpose to (N, channels, T) for Conv1d.
        out = out.transpose(1, 2)  # (1, 768, T)
        
        # 11. Final 1×1 convolution to get final logits.
        logits_seq = self.final_conv(out)  # (1, num_classes, T)
        # 12. Use the last timestep (the query's output) as the prediction.
        final_logits = logits_seq[:, :, -1]  # (1, num_classes)
        return final_logits

    def forward(self, support_images, support_labels, query_images):
        """
        Process a single query episode.
        We assume one query per episode.
        """
        return self.forward_single_query(support_images, support_labels, query_images)

