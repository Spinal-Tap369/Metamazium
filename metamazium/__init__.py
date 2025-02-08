# metamazium/__init__.py

# Importing environment-related modules
from metamazium.env.maze_env import MetaMazeDiscrete3D
import gymnasium as gym

# Importing Performer components
from metamazium.performer.performer_pytorch import PerformerLM, Performer, SelfAttention, CrossAttention
from metamazium.performer.autoregressive_wrapper import AutoregressiveWrapper
from metamazium.performer.performer_enc_dec import PerformerEncDec

# Importing SNAIL Performer components
from metamazium.snail_performer.snail_model import SNAILPolicyValueNet
from metamazium.snail_performer.trpo_fo import TRPO_FO
from metamazium.snail_performer.cnn_encoder import CNNEncoder

from metamazium.train_model import train_lstm_trpo_fo_main
from metamazium.train_model import train_snail_trpo_fo_main

__all__ = [
    'MetaMazeDiscrete3D',
    'MetaMaze2D',
    'PerformerLM',
    'Performer',
    'SelfAttention',
    'CrossAttention',
    'AutoregressiveWrapper',
    'PerformerEncDec',
    'SNAILPolicyValueNet',
    'TRPO_FO',
    'CNNEncoder',
    'train_lstm_trpo_fo_main',
    'train_snail_trpo_fo_main'
]
