# snail_performer/ppo.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PPOTrainer:
    """
    PPO trainer for sequence-based policies without mode-specific calls.
    Processes entire sequences at once.
    """
    def __init__(self,
                 policy_model,
                 lr=3e-4,
                 gamma=0.99,
                 gae_lambda=0.99,
                 clip_range=0.2,
                 n_epochs=10,
                 batch_size=8,
                 target_kl=0.01,
                 max_grad_norm=0.5,
                 entropy_coef=0.0,
                 value_coef=0.5):
        """
        Initializes the PPO trainer.

        Args:
            policy_model (nn.Module): Policy network to train.
            lr (float): Learning rate.
            gamma (float): Discount factor.
            gae_lambda (float): GAE lambda for advantage estimation.
            clip_range (float): PPO clipping range.
            n_epochs (int): Number of training epochs.
            batch_size (int): Mini-batch size.
            target_kl (float): KL divergence threshold for early stopping.
            max_grad_norm (float): Gradient clipping norm.
            entropy_coef (float): Coefficient for entropy regularization.
            value_coef (float): Coefficient for value function loss.
        """
        self.policy_model = policy_model
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

    def compute_gae(self, rewards, dones, values, next_value):
        """
        Computes Generalized Advantage Estimation (GAE).

        Args:
            rewards (np.array): Rewards sequence of shape (T,).
            dones (np.array): Done flags of shape (T,).
            values (np.array): Value predictions of shape (T,).
            next_value (float): Value of the next state.

        Returns:
            np.array: Computed advantages of shape (T,).
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
        Performs PPO updates using rollout data.

        Args:
            rollouts (dict): Contains:
                - obs (np.array): Observations (N, T, 3, 30, 40).
                - actions (np.array): Actions taken (N, T).
                - old_log_probs (np.array): Log probabilities (N, T).
                - returns (np.array): Discounted returns (N, T).
                - values (np.array): Value predictions (N, T).
                - advantages (np.array): Computed advantages (N, T).

        Returns:
            dict: Training metrics including KL divergence, loss, and entropy.
        """
        obs = rollouts["obs"]
        actions = rollouts["actions"]
        old_log_probs = rollouts["old_log_probs"]
        returns_ = rollouts["returns"]
        values_ = rollouts["values"]
        advantages_ = rollouts["advantages"]

        N, T = actions.shape
        device = next(self.policy_model.parameters()).device

        obs_t = torch.from_numpy(obs).float().to(device)
        actions_t = torch.from_numpy(actions).long().to(device)
        old_log_probs_t = torch.from_numpy(old_log_probs).float().to(device)
        returns_t = torch.from_numpy(returns_).float().to(device)
        values_t = torch.from_numpy(values_).float().to(device)
        advantages_t = torch.from_numpy(advantages_).float().to(device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        clipfracs = []
        final_approx_kl = 0.0
        indices = np.arange(N)

        for epoch_i in range(self.n_epochs):
            np.random.shuffle(indices)
            start_idx = 0
            while start_idx < N:
                end_idx = start_idx + self.batch_size
                batch_idx = indices[start_idx:end_idx]
                start_idx = end_idx

                batch_obs = obs_t[batch_idx]
                batch_actions = actions_t[batch_idx]
                batch_old_logp = old_log_probs_t[batch_idx]
                batch_adv = advantages_t[batch_idx]
                batch_returns = returns_t[batch_idx]

                # Reset states for stateless policies (no-op for SNAIL)
                self.policy_model.reset_lstm_states(batch_size=len(batch_idx))

                # Forward pass
                logits, v_pred = self.policy_model(batch_obs)
                B_, seq_len, act_dim = logits.shape

                logits_2d = logits.view(B_ * seq_len, act_dim)
                v_pred_1d = v_pred.view(B_ * seq_len)

                actions_2d = batch_actions.view(B_ * seq_len)
                old_logp_1d = batch_old_logp.view(B_ * seq_len)
                adv_1d = batch_adv.view(B_ * seq_len)
                returns_1d = batch_returns.view(B_ * seq_len)

                dist = torch.distributions.Categorical(logits=logits_2d)
                new_log_probs_1d = dist.log_prob(actions_2d)
                ratio = torch.exp(new_log_probs_1d - old_logp_1d)

                surr1 = ratio * adv_1d
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_1d
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(v_pred_1d, returns_1d)
                entropy = dist.entropy().mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                approx_kl = 0.5 * torch.mean((new_log_probs_1d - old_logp_1d) ** 2).cpu().item()
                final_approx_kl = approx_kl

                clip_frac = ((ratio > (1 + self.clip_range)) | (ratio < (1 - self.clip_range))).float().mean()
                clipfracs.append(clip_frac.item())

                if approx_kl > self.target_kl:
                    print(f"Early stopping at epoch={epoch_i} due to KL={approx_kl:.4f} > {self.target_kl}")
                    break

            if final_approx_kl > self.target_kl:
                break

        return {
            "clip_fraction": float(np.mean(clipfracs)),
            "final_approx_kl": final_approx_kl,
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }
