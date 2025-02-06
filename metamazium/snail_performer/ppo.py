# metamazium/snail_performer/ppo.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PPOTrainer:
    """
    Proximal Policy Optimization trainer for sequence models.
    Processes a full (B,T) sequence per update without mini-batching.
    """
    def __init__(self,
                 policy_model,
                 lr=3e-4,
                 gamma=0.99,
                 gae_lambda=0.99,
                 clip_range=0.2,
                 n_epochs=1,
                 target_kl=0.01,
                 max_grad_norm=0.5,
                 entropy_coef=0.0,
                 value_coef=0.5):
        self.policy_model = policy_model
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

    def compute_gae(self, rewards, dones, values, next_value):
        """
        Compute Generalized Advantage Estimation for a 1D trajectory.
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma*(1.0 - dones[t])*next_value - values[t]
            gae = delta + self.gamma*self.gae_lambda*(1.0 - dones[t])*gae
            advantages[t] = gae
            next_value = values[t]
        return advantages

    def update(self, rollouts):
        """
        Perform a PPO update using provided rollouts.
        """
        # Unpack rollout data
        obs = rollouts["obs"]
        actions = rollouts["actions"]
        old_log_probs = rollouts["old_log_probs"]
        returns_ = rollouts["returns"]
        advantages_ = rollouts["advantages"]

        # future proof - in case this is good!
        # p1_steps = rollouts.get("p1_steps", None)
        # p2_steps = rollouts.get("p2_steps", None)
        # p1_goals = rollouts.get("p1_goals", None)
        # p2_goals = rollouts.get("p2_goals", None)

        # Convert data to tensors on the correct device
        device = next(self.policy_model.parameters()).device
        obs_t = torch.from_numpy(obs).float().to(device)
        actions_t = torch.from_numpy(actions).long().to(device)
        old_log_probs_t = torch.from_numpy(old_log_probs).float().to(device)
        returns_t = torch.from_numpy(returns_).float().to(device)
        advantages_t = torch.from_numpy(advantages_).float().to(device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        B, T_seq = actions_t.shape

        for epoch_i in range(self.n_epochs):
            # Forward pass through policy model
            logits_new, values_new = self.policy_model(obs_t)
            # Reshape outputs for loss computation
            logits_2d = logits_new.view(B*T_seq, -1)
            values_1d = values_new.view(B*T_seq)
            actions_1d = actions_t.view(B*T_seq)
            old_log_probs_1d = old_log_probs_t.view(B*T_seq)
            adv_1d = advantages_t.view(B*T_seq)
            returns_1d = returns_t.view(B*T_seq)

            # Compute new log probabilities and ratios for PPO
            dist = torch.distributions.Categorical(logits=logits_2d)
            new_log_probs_1d = dist.log_prob(actions_1d)
            ratio = torch.exp(new_log_probs_1d - old_log_probs_1d)

            # Surrogate loss with clipping
            surr1 = ratio * adv_1d
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_1d
            policy_loss = -torch.min(surr1, surr2).mean()

            # Compute value loss and entropy bonus
            value_loss = F.mse_loss(values_1d, returns_1d)
            entropy = dist.entropy().mean()
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            # Backpropagation and optimization step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Early stopping based on KL divergence
            approx_kl = 0.5 * torch.mean((new_log_probs_1d - old_log_probs_1d)**2).item()
            if approx_kl > self.target_kl:
                print(f"Early stopping at epoch={epoch_i} due to KL={approx_kl:.4f} > {self.target_kl}")
                break

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "approx_kl": approx_kl
        }
