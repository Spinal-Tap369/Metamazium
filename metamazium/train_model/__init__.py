# train_model/__init__.py

from .train_snail_performer import main as train_snail_performer
from .train_lstm_trpo import main as train_lstm_ppo

__all__ = ["train_snail_performer",
           "train_lstm_ppo"
           ]
