import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PPOTrainer:
    """
    A simplified Proximal Policy Optimization (PPO) trainer that:
      - Processes a single rollout per update,
      - Does not employ mini-batches,
    aligning with the style of SNAIL-like PPO implementations.
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
        """
        Initializes the PPO trainer with hyperparameters and optimizer.

        Args:
            policy_model (nn.Module): The policy network to be trained.
            lr (float): Learning rate.
            gamma (float): Discount factor.
            gae_lambda (float): GAE (Generalized Advantage Estimation) lambda.
            clip_range (float): Clipping range for PPO.
            n_epochs (int): Number of training epochs per update.
            target_kl (float): Target KL divergence for early stopping.
            max_grad_norm (float): Maximum gradient norm for clipping.
            entropy_coef (float): Coefficient for entropy regularization.
            value_coef (float): Coefficient for value function loss.
        """
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
        Compute Generalized Advantage Estimation (GAE) for a 1D trajectory.

        Args:
            rewards (np.ndarray): Array of rewards along the trajectory.
            dones (np.ndarray): Array indicating episode termination flags.
            values (np.ndarray): Array of value function estimates.
            next_value (float): Value estimate for the state following the last one.

        Returns:
            np.ndarray: Array of advantage estimates.
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * (1.0 - dones[t]) * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
            next_value = values[t]
        return advantages

    def update(self, rollouts):
        """
        Perform a PPO update using collected rollouts.

        Args:
            rollouts (dict): Dictionary containing rollout data with keys:
              - "obs": Observation tensor of shape (B, T, 6, H, W)
              - "actions": Actions tensor of shape (B, T)
              - "old_log_probs": Log probabilities tensor of shape (B, T)
              - "values": Value estimates tensor of shape (B, T)
              - "returns": Returns tensor of shape (B, T)
              - "advantages": Advantages tensor of shape (B, T)

        Returns:
            dict: Statistics from the PPO update including policy loss, value loss, entropy, and approximate KL divergence.
        """
        obs = rollouts["obs"]          # Shape: (B, T, 6, H, W)
        actions = rollouts["actions"]  # Shape: (B, T)
        old_log_probs = rollouts["old_log_probs"]  # Shape: (B, T)
        returns_ = rollouts["returns"] # Shape: (B, T)
        old_values_ = rollouts["values"] # Shape: (B, T)
        advantages_ = rollouts["advantages"] # Shape: (B, T)

        device = next(self.policy_model.parameters()).device
        obs_t = torch.from_numpy(obs).float().to(device)
        actions_t = torch.from_numpy(actions).long().to(device)
        old_log_probs_t = torch.from_numpy(old_log_probs).float().to(device)
        returns_t = torch.from_numpy(returns_).float().to(device)
        advantages_t = torch.from_numpy(advantages_).float().to(device)

        # Normalize the advantages for numerical stability.
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        B, T = actions_t.shape

        for epoch_i in range(self.n_epochs):
            # Compute new policy logits and value estimates.
            logits_new, values_new = self.policy_model.forward_rollout(obs_t)

            # Reshape tensors for PPO calculations.
            logits_2d = logits_new.view(B*T, -1)
            values_1d = values_new.view(B*T)
            actions_1d = actions_t.view(B*T)
            old_log_probs_1d = old_log_probs_t.view(B*T)
            adv_1d = advantages_t.view(B*T)
            returns_1d = returns_t.view(B*T)

            # Calculate the probability ratio using the new and old log probabilities.
            dist = torch.distributions.Categorical(logits=logits_2d)
            new_log_probs_1d = dist.log_prob(actions_1d)

            ratio = torch.exp(new_log_probs_1d - old_log_probs_1d)
            surr1 = ratio * adv_1d
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_1d
            policy_loss = -torch.min(surr1, surr2).mean()

            # Compute value and entropy losses.
            value_loss = F.mse_loss(values_1d, returns_1d)
            entropy = dist.entropy().mean()

            # Aggregate total loss.
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Compute approximate KL divergence for early stopping.
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
