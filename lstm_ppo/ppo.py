# lstm_ppo/ppo.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PPOTrainer:
    """
    PPO trainer that handles full-sequence LSTM forward passes for each episode (or chunk).
    Aligns more closely with the SNAIL paper approach (processing entire episodes).
    """

    def __init__(self,
                 policy_model,              # LSTMPolicy
                 lr=3e-4,
                 gamma=0.99,
                 gae_lambda=0.99,
                 clip_range=0.2,
                 n_epochs=10,
                 batch_size=8,             # how many sequences per mini-batch?
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
        self.batch_size = batch_size      # # of sequences in each mini-batch
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

    def compute_gae(self, rewards, dones, values, next_value):
        """
        rewards, dones, values: shape (T,)
        next_value: float
        Returns: advantages (T,) with GAE
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
        rollouts is a dictionary containing sequences in the shape:
            rollouts["obs"] -> (N, T, 3, 30, 40)   # N = #sequences, T = max steps
            rollouts["actions"] -> (N, T)
            rollouts["old_log_probs"] -> (N, T)
            rollouts["returns"] -> (N, T)
            rollouts["values"] -> (N, T)
            rollouts["advantages"] -> (N, T)
            rollouts["masks"] -> (N, T)  # e.g. (1 - done)
              or you can store "dones" -> (N, T) as well
        We'll do multi-epoch mini-batch updates across these sequences.
        """
        obs = rollouts["obs"]                 # shape (N, T, 3,30,40)
        actions = rollouts["actions"]         # shape (N, T)
        old_log_probs = rollouts["old_log_probs"]  # shape (N, T)
        returns_ = rollouts["returns"]        # shape (N, T)
        values_ = rollouts["values"]          # shape (N, T)
        advantages_ = rollouts["advantages"]  # shape (N, T)

        N, T = actions.shape  # number of sequences, length of each

        # Convert to torch
        device = next(self.policy_model.parameters()).device
        obs_t = torch.from_numpy(obs).float().to(device)  # (N, T, 3,30,40)
        actions_t = torch.from_numpy(actions).long().to(device)  # (N, T)
        old_log_probs_t = torch.from_numpy(old_log_probs).float().to(device)  # (N, T)
        returns_t = torch.from_numpy(returns_).float().to(device)   # (N, T)
        values_t = torch.from_numpy(values_).float().to(device)     # (N, T)
        advantages_t = torch.from_numpy(advantages_).float().to(device) # (N, T)

        # Optionally normalize advantages per-batch
        advantages_mean = advantages_t.mean()
        advantages_std = advantages_t.std() + 1e-8
        advantages_t = (advantages_t - advantages_mean) / advantages_std

        clipfracs = []
        final_approx_kl = 0.0

        # Flatten the sequences if we want to shuffle them
        # We'll produce an index list of size N. We'll do mini-batches of entire sequences (no partial seq).
        indices = np.arange(N)

        for epoch_i in range(self.n_epochs):
            np.random.shuffle(indices)
            start_idx = 0
            while start_idx < N:
                end_idx = start_idx + self.batch_size
                batch_idx = indices[start_idx:end_idx]
                start_idx = end_idx

                # Slice out the sequences
                batch_obs = obs_t[batch_idx]           # (B, T, 3,30,40)
                batch_actions = actions_t[batch_idx]   # (B, T)
                batch_old_logp = old_log_probs_t[batch_idx]  # (B, T)
                batch_adv = advantages_t[batch_idx]    # (B, T)
                batch_returns = returns_t[batch_idx]   # (B, T)

                # 1) Reset LSTM hidden state for this mini-batch
                self.policy_model.reset_lstm_states(batch_size=len(batch_idx))

                # 2) Forward pass the entire sequence (B,T,...)
                logits, v_pred = self.policy_model(batch_obs)  
                # logits -> (B, T, action_dim), v_pred -> (B, T)

                # Flatten to (B*T, ...)
                B_, seq_len, act_dim = logits.shape
                logits_2d = logits.view(B_ * seq_len, act_dim)         # (B*T, action_dim)
                v_pred_1d = v_pred.view(B_ * seq_len)                  # (B*T)

                # Same flatten for other arrays
                actions_2d = batch_actions.view(B_ * seq_len)           # (B*T,)
                old_logp_1d = batch_old_logp.view(B_ * seq_len)         # (B*T,)
                adv_1d = batch_adv.view(B_ * seq_len)                  # (B*T,)
                returns_1d = batch_returns.view(B_ * seq_len)           # (B*T,)

                dist = torch.distributions.Categorical(logits=logits_2d)
                new_log_probs_1d = dist.log_prob(actions_2d)            # (B*T,)

                ratio = torch.exp(new_log_probs_1d - old_logp_1d)       # (B*T,)

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

                # Some stats
                approx_kl = 0.5 * torch.mean((new_log_probs_1d - old_logp_1d)**2).cpu().item()
                final_approx_kl = approx_kl

                # clip fraction
                clip_frac = (ratio > (1 + self.clip_range)).float() + (ratio < (1 - self.clip_range)).float()
                clip_frac = torch.mean(clip_frac)
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
