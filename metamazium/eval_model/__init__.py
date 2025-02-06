# eval_model/__init__.py

from .test_snail import main as test_snail
from .test_lstm import main as test_lstm

__all__ = ["test_snail", "test_lstm"]