# eval_model/__init__.py

from .test_snail import main as test_snail_main
from .test_lstm import main as test_lstm_main

__all__ = ["test_snail_main", "test_lstm_main"]