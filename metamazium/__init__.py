# metamazium/__init__.py

# Importing environment-related modules
from .env.maze_env import MetaMazeContinuous3D, MetaMazeDiscrete3D, MetaMaze2D

# Importing Performer components
from .performer.performer_pytorch import PerformerLM, Performer, SelfAttention, CrossAttention
from .performer.autoregressive_wrapper import AutoregressiveWrapper
from .performer.performer_enc_dec import PerformerEncDec

# Importing SNAIL Performer components
from .snail_performer.snail_model import SNAILPolicyValueNet
from .snail_performer.ppo import PPOTrainer
from .snail_performer.cnn_encoder import CNNEncoder


# Define the public API of the package
__all__ = [
    'MetaMazeContinuous3D',
    'MetaMazeDiscrete3D',
    'MetaMaze2D',
    'PerformerLM',
    'Performer',
    'SelfAttention',
    'CrossAttention',
    'AutoregressiveWrapper',
    'PerformerEncDec',
    'SNAILPolicyValueNet',
    'PPOTrainer',
    'CNNEncoder',
]
