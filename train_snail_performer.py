# train_snail_performer.py

import gymnasium as gym
import numpy as np
import torch
import json
from tqdm import tqdm

import env 
from snail_performer.snail_model import SNAILPolicyValueNet
from snail_performer.ppo import PPOTrainer
from env.maze_task import MazeTaskSampler


def main():
    env_id = "MetaMazeDiscrete3D-v0"
    env = gym.make(env_id, enable_render=False)

    # Load tasks configuration
    with open("mazes_data/train_tasks.json", "r") as f:
        tasks = json.load(f)
    num_tasks = len(tasks)
    print(f"Loaded {num_tasks} tasks.")

    # Hyperparameters
    total_timesteps = 1200000
    steps_per_update = 80000
    gamma = 0.99
    gae_lambda = 0.99
    clip_range = 0.1
    target_kl = 0.03
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize SNAIL model and PPO trainer
    snail = SNAILPolicyValueNet(
        action_dim=4,
        base_dim=256,
        policy_filters=32,
        policy_attn_dim=16,
        value_filters=16,
        seq_len=800,
        num_policy_attn=2
    ).to(device)

    ppo_trainer = PPOTrainer(
        policy_model=snail,
        lr=0.00005,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        n_epochs=3,
        target_kl=target_kl,
        max_grad_norm=0.7,
        entropy_coef=0.01,
        value_coef=0.5
    )

    total_steps = 0
    task_idx = 0
    pbar = tqdm(total=total_timesteps, desc="Training")

    # Buffers to store rollout data
    obs_buffer = []
    act_buffer = []
    logp_buffer = []
    val_buffer = []
    rew_buffer = []
    done_buffer = []

    while total_steps < total_timesteps:
        # Set new task for each episode/maze
        task_config = MazeTaskSampler(**tasks[task_idx])
        env.unwrapped.set_task(task_config)
        task_idx = (task_idx + 1) % num_tasks

        obs_raw, _ = env.reset()
        done = False
        truncated = False

        # Variables for the additional channels
        last_action = 0.0
        last_reward = 0.0
        boundary_bit = 1.0
        phase_boundary_signaled = False

        # Track phase steps and rewards
        phase1_steps = 0
        phase2_steps = 0
        phase1_rew = 0.0
        phase2_rew = 0.0

        # Episode-specific buffers
        ep_obs_seq = []
        ep_actions = []
        ep_logprobs = []
        ep_values = []
        ep_rewards = []
        ep_dones = []

        while not done and not truncated and total_steps < total_timesteps:
            # Construct 6-channel observation
            obs_img = np.transpose(obs_raw, (2, 0, 1))  # (3, H, W)
            H, W = obs_img.shape[1], obs_img.shape[2]
            c3 = np.full((1, H, W), last_action, dtype=np.float32)
            c4 = np.full((1, H, W), last_reward, dtype=np.float32)
            c5 = np.full((1, H, W), boundary_bit, dtype=np.float32)
            obs_6ch = np.concatenate([obs_img, c3, c4, c5], axis=0)  # (6, H, W)

            # Store the 6-channel observation
            ep_obs_seq.append(obs_6ch)
            t_len = len(ep_obs_seq)

            # Build partial sequence and forward through SNAIL
            obs_seq_np = np.stack(ep_obs_seq, axis=0)[None]  # (1, t_len, 6, H, W)
            obs_seq_torch = torch.from_numpy(obs_seq_np).float().to(device)
            with torch.no_grad():
                logits_seq, vals_seq = snail(obs_seq_torch)

            # Get the policy logits/value for just the current time step
            logits_t = logits_seq[:, t_len - 1, :]  # shape (1, action_dim)
            val_t = vals_seq[:, t_len - 1]          # shape (1,)

            # Sample action
            dist = torch.distributions.Categorical(logits=logits_t)
            action = dist.sample()
            logp = dist.log_prob(action)

            # Step environment
            obs_next, reward, done, truncated, info = env.step(action.item())
            total_steps += 1
            pbar.update(1)

            # Save transition
            ep_actions.append(action.item())
            ep_logprobs.append(logp.item())
            ep_values.append(val_t.item())
            ep_rewards.append(reward)
            ep_dones.append(float(done))

            # Update the last_action/last_reward for next step
            last_action = float(action.item())
            last_reward = float(reward)
            obs_raw = obs_next

            # Count steps/reward in whichever phase we are in
            if info["phase"] == 1:
                phase1_steps += 1
                phase1_rew += reward
            else:
                phase2_steps += 1
                phase2_rew += reward

            # Update boundary bit on phase change
            if env.unwrapped.maze_core.phase == 2 and not phase_boundary_signaled:
                boundary_bit = 1.0
                phase_boundary_signaled = True
            else:
                boundary_bit = 0.0

        # Episode done or truncated; log final stats
        print(f"== Maze finished: Phase1 Steps={phase1_steps}, Reward={phase1_rew:.3f}, "
              f"Phase2 Steps={phase2_steps}, Reward={phase2_rew:.3f} ==")

        # Append episode data to main buffers
        obs_buffer.extend(ep_obs_seq)
        act_buffer.extend(ep_actions)
        logp_buffer.extend(ep_logprobs)
        val_buffer.extend(ep_values)
        rew_buffer.extend(ep_rewards)
        done_buffer.extend(ep_dones)

        # Trigger PPO update if buffer is large enough
        if len(obs_buffer) >= steps_per_update:
            do_update(snail, ppo_trainer,
                      obs_buffer, act_buffer, logp_buffer,
                      val_buffer, rew_buffer, done_buffer, device)
            obs_buffer.clear()
            act_buffer.clear()
            logp_buffer.clear()
            val_buffer.clear()
            rew_buffer.clear()
            done_buffer.clear()

    pbar.close()
    env.close()

    # Final PPO update for any remaining data
    if len(obs_buffer) > 0:
        do_update(snail, ppo_trainer,
                  obs_buffer, act_buffer, logp_buffer,
                  val_buffer, rew_buffer, done_buffer, device)

    torch.save(snail.state_dict(), "models/snail_snaillike_policy_value_online.pt")
    print(f"Training finished after {total_steps} steps.")


def do_update(snail_net, trainer, obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf, device):
    """
    Prepare rollout data, compute GAE and returns, then perform a PPO update.
    Also print more detailed info about the PPO step.
    """
    T = len(obs_buf)
    if T < 2:
        return

    # Stack buffers into arrays
    obs_np = np.stack(obs_buf, axis=0)[None]    # (1, T, 6, H, W)
    acts_np = np.array(act_buf, dtype=np.int64)[None]        # (1, T)
    logp_np = np.array(logp_buf, dtype=np.float32)[None]     # (1, T)
    vals_np = np.array(val_buf, dtype=np.float32)[None]      # (1, T)
    rews_np = np.array(rew_buf, dtype=np.float32)[None]      # (1, T)
    done_np = np.array(done_buf, dtype=np.float32)[None]     # (1, T)

    B, T_ = acts_np.shape
    # We'll assume next_value = 0 if the final step was terminal
    next_value = 0.0

    rewards_ = rews_np.reshape(-1)
    dones_ = done_np.reshape(-1)
    values_ = vals_np.reshape(-1)

    # standardize rewards
    mean_r = np.mean(rewards_)
    std_r  = np.std(rewards_) + 1e-6
    rewards_ = (rewards_ - mean_r) / std_r

    # Compute advantages using GAE
    advantages_ = trainer.compute_gae(
        rewards=rewards_,
        dones=dones_,
        values=values_,
        next_value=next_value
    )
    returns_ = values_ + advantages_

    adv_2d = advantages_.reshape(B, T_)
    ret_2d = returns_.reshape(B, T_)

    rollouts = {
        "obs": obs_np,
        "actions": acts_np,
        "old_log_probs": logp_np,
        "returns": ret_2d,
        "values": vals_np,
        "advantages": adv_2d
    }

    stats = trainer.update(rollouts)
    print("[PPO Update] "
          f"policy_loss={stats['policy_loss']:.4f}, "
          f"value_loss={stats['value_loss']:.4f}, "
          f"entropy={stats['entropy']:.4f}, "
          f"approx_kl={stats['approx_kl']:.4f}")


if __name__ == "__main__":
    main()
