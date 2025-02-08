# metamazium/train_model/__init__.py

from .train_lstm_trpo import main as train_lstm_trpo_main
from .train_lstm_trpo_fo import main as train_lstm_trpo_fo_main
from .train_snail_trpo_fo import main as train_snail_trpo_fo_main

__all__ = [
    "train_lstm_trpo_main",
    "train_lstm_trpo_fo_main",
    "train_snail_trpo_fo_main",
]
