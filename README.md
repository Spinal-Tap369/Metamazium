!!WORK IN PROGRESS!!

Training an agent to navigate a maze using Meta Reinforcement Learning techniques. Based on the following existing work - 
1. RL2: Fast Reinforcement Learning via Slow Reinforcement Learning - https://arxiv.org/pdf/1611.02779
2. Learning to reinforcement learn - https://arxiv.org/abs/1611.05763
3. SIMPLE NEURAL ATTENTIVE META-LEARNER - https://arxiv.org/pdf/1707.03141
4. Trust Region Policy Optimization - https://arxiv.org/abs/1502.05477, https://github.com/ajlangley/trpo-pytorch
5. FAVOR+ (Rethinking Attention with Performers) - https://arxiv.org/abs/2009.14794v4, https://github.com/lucidrains/performer-pytorch
6. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (used in SNAIL implementation if training on CUDA) - https://arxiv.org/abs/2205.14135, https://github.com/Dao-AILab/flash-attention



![Alt text for image](/others/mazium.png)


NOTE - The original metamaze https://github.com/PaddlePaddle/MetaGym/tree/master/metagym/metamaze uses Gym which is no longer supported. I've adapted the discrete 3d maze for gymnasium and there have been some more changes to enable the agent to carry over memory across a trial.