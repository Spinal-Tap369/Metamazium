!!WORK IN PROGRESS!!

Training an agent to navigate a maze using Meta Reinforcement Learning techniques.

Working on implementing a Simple Neural Attentive Meta Learner (SNAIL) Mishra, - Nikhil, et al. "A simple neural attentive meta-learner." arXiv preprint arXiv:1707.03141 (2017). https://arxiv.org/pdf/1707.03141

Using PPO instead of TRPO GAE as in the original paper

![Alt text for image](/others/mazium.png)

- Update - Since implementing the architecture of SNAIL would mean having a time complexity of O(n)<sup>2</sup>, working on implementing any linear time attention mechanism which would be computationally efficient. Created a LSTM model to serve as the base line and to test a ppo training loop.


NOTE - The original metamaze https://github.com/PaddlePaddle/MetaGym/tree/master/metagym/metamaze uses Gym which is no longer supported. I've adapted the discrete 3d maze for gymnasium and it has some more changes to enable the agent to carry over memory across a trial.