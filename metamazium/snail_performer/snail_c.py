# metamazium/snail_performer/snail_c.py

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CasualConv1d(nn.Module):
    """
    1D causal convolution.
    Pads appropriately (based on dilation) and slices off the last few timesteps to enforce causality.
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, dilation=1, groups=1, bias=True):
        super(CasualConv1d, self).__init__()
        self.dilation = dilation
        padding = dilation * (kernel_size - 1)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                                padding, dilation, groups, bias)

    def forward(self, x):
        # x shape: (N, in_channels, T)
        out = self.conv1d(x)
        # Enforce causality by removing the last "dilation" timesteps
        return out[:, :, :-self.dilation] if self.dilation > 0 else out

class DenseBlock(nn.Module):
    """
    Applies two causal convolutions with tanh and sigmoid non-linearities,
    then concatenates the original input with the elementwise product.
    """
    def __init__(self, in_channels, dilation, filters, kernel_size=2):
        super(DenseBlock, self).__init__()
        self.casualconv1 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)
        self.casualconv2 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)

    def forward(self, x):
        # x shape: (N, in_channels, T)
        xf = self.casualconv1(x)
        xg = self.casualconv2(x)
        activations = torch.tanh(xf) * torch.sigmoid(xg)
        return torch.cat((x, activations), dim=1)

class TCBlock(nn.Module):
    """
    Temporal Convolution Block that stacks several DenseBlocks with exponentially increasing dilation.
    """
    def __init__(self, in_channels, seq_length, filters):
        super(TCBlock, self).__init__()
        num_blocks = int(math.ceil(math.log(seq_length, 2)))
        self.dense_blocks = nn.ModuleList([
            DenseBlock(in_channels + i * filters, dilation=2 ** (i+1), filters=filters)
            for i in range(num_blocks)
        ])

    def forward(self, x):
        # x shape: (N, T, in_channels) → transpose to (N, in_channels, T)
        x = x.transpose(1, 2)
        for block in self.dense_blocks:
            x = block(x)
        return x.transpose(1, 2)

class AttentionBlock(nn.Module):
    """
    Attention Block that computes query, key, and value projections,
    applies a causal mask, and concatenates the attention output to the original input.
    """
    def __init__(self, in_channels, key_size, value_size):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)

    def forward(self, x):
        # x shape: (N, T, in_channels)
        N, T, _ = x.size()
        # Build a causal mask: mask[i, j] = 1 if i > j, else 0.
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1).to(x.device)
        keys = self.linear_keys(x)      # (N, T, key_size)
        query = self.linear_query(x)    # (N, T, key_size)
        values = self.linear_values(x)  # (N, T, value_size)
        attn_logits = torch.bmm(query, keys.transpose(1, 2))  # (N, T, T)
        attn_logits.data.masked_fill_(mask, -float('inf'))
        attn = F.softmax(attn_logits / self.sqrt_key_size, dim=1)
        attn_out = torch.bmm(attn, values)  # (N, T, value_size)
        return torch.cat((x, attn_out), dim=2)  # (N, T, in_channels + value_size)

class OmniglotNet(nn.Module):
    """
    A simple 4-layer CNN for Omniglot.
    Input: (N, 1, 28, 28)
    Output: a 64-d feature vector.
    """
    def __init__(self):
        super(OmniglotNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        # x: (N, 1, 28, 28)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # (N, 64)
        return x

class SNAILFewShot(nn.Module):
    """
    SNAIL model for few-shot classification.
    For an N-way, K-shot episode, the input sequence length is T = N*K + 1 (support images + query image).
    The encoder output (64-d for Omniglot) is concatenated with the one-hot label (dimension N) to form
    an input of dimension 64+N.
    
    The architecture follows:
      AttentionBlock → TCBlock → AttentionBlock → TCBlock → AttentionBlock → FC.
    The final prediction is taken from the last timestep (the query).
    """
    def __init__(self, N, K, task='omniglot', use_cuda=False):
        super(SNAILFewShot, self).__init__()
        self.N = N  # number of classes per episode
        self.K = K  # number of support samples per class
        if task == 'omniglot':
            self.encoder = OmniglotNet()
            num_channels = 64 + N  # encoder output (64-d) plus one-hot label (N-d)
        else:
            raise ValueError("Task not recognized. Only 'omniglot' is supported.")
        # Total timesteps per episode: T = N*K + 1 (support + query)
        seq_length = N * K + 1
        num_filters = int(math.ceil(math.log(seq_length, 2)))
        
        self.attention1 = AttentionBlock(num_channels, 64, 32)
        num_channels += 32
        self.tc1 = TCBlock(num_channels, seq_length, 128)
        num_channels += num_filters * 128
        self.attention2 = AttentionBlock(num_channels, 256, 128)
        num_channels += 128
        self.tc2 = TCBlock(num_channels, seq_length, 128)
        num_channels += num_filters * 128
        self.attention3 = AttentionBlock(num_channels, 512, 256)
        num_channels += 256
        self.fc = nn.Linear(num_channels, N)
        self.use_cuda = use_cuda

    def forward(self, images, labels):
        """
        images: (batch_size * T, C, H, W)
        labels: (batch_size * T, N) one-hot vectors.
          (For the query image, the label should be a dummy vector of zeros.)
        """
        x = self.encoder(images)  # (batch_size * T, 64)
        x = torch.cat((x, labels), dim=1)  # (batch_size * T, 64+N)
        # Reshape into episodes: (batch_size, T, feature_dim)
        batch_size = images.size(0) // (self.N * self.K + 1)
        T = self.N * self.K + 1
        x = x.view(batch_size, T, -1)
        x = self.attention1(x)
        x = self.tc1(x)
        x = self.attention2(x)
        x = self.tc2(x)
        x = self.attention3(x)
        logits = self.fc(x)  # (batch_size, T, N)
        # Use the last timestep (query) for classification.
        final_logits = logits[:, -1, :]  # (batch_size, N)
        return final_logits
