# metamazium/lstm_trpo/lstm_sl.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class LSTMSLFewShot(nn.Module):
    """
    LSTM-based meta learner for few-shot classification.
    
    For an N-way, K-shot episode, the input sequence length is:
      T = N*K + 1 (support images + query image).
    
    The encoder output (64-d for Omniglot) is concatenated with the one-hot label (dimension N)
    to form an input of dimension 64+N. This sequence is then fed into a stacked LSTM.
    
    The final prediction is taken from the last timestep (the query image).
    """
    def __init__(self, N, K, task='omniglot', hidden_size=256, num_layers=2):
        super(LSTMSLFewShot, self).__init__()
        self.N = N  # number of classes per episode
        self.K = K  # number of support samples per class
        if task == 'omniglot':
            self.encoder = OmniglotNet()
            self.input_dim = 64 + N  # encoder output plus one-hot label
        else:
            raise ValueError("Task not recognized. Only 'omniglot' is supported.")
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, N)

    def forward(self, images, labels):
        """
        images: Tensor of shape (batch_size * T, C, H, W)
        labels: Tensor of shape (batch_size * T, N) (one-hot vectors; query image uses dummy zeros)
        """
        x = self.encoder(images)  # (batch_size*T, 64)
        x = torch.cat((x, labels), dim=1)  # (batch_size*T, 64+N)
        T = self.N * self.K + 1
        batch_size = images.size(0) // T
        x = x.view(batch_size, T, -1)  # (batch_size, T, 64+N)
        lstm_out, _ = self.lstm(x)  # (batch_size, T, hidden_size)
        final_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        logits = self.fc(final_out)  # (batch_size, N)
        return logits
