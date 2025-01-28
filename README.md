!!WORK IN PROGRESS!!

Training an agent to navigate a maze using Meta Reinforcement Learning techniques.

Working on implementing a Simple Neural Attentive Meta Learner (SNAIL) Mishra, - Nikhil, et al. "A simple neural attentive meta-learner." arXiv preprint arXiv:1707.03141 (2017). https://arxiv.org/pdf/1707.03141

Using PPO instead of TRPO GAE as in the original paper

![Alt text for image](/others/mazium.png)

- Update
SNAIL's attention mechanism is being done using FAVOR+ for its linear time capabilities. 
SNAIL to be compared to a stacked LSTM(our baseline)


NOTE - The original metamaze https://github.com/PaddlePaddle/MetaGym/tree/master/metagym/metamaze uses Gym which is no longer supported. I've adapted the discrete 3d maze for gymnasium and it has some more changes to enable the agent to carry over memory across a trial.