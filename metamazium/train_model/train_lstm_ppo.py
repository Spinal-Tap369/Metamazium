# train_lstm_ppo.py

import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import json
from tqdm import tqdm

import metamazium.env  # Importing custom Maze environment definitions.
from metamazium.lstm_ppo.lstm_model import StackedLSTMPolicy
from metamazium.lstm_ppo.ppo import PPOTrainer
from metamazium.env.maze_task import MazeTaskSampler

CHECKPOINT_DIR = "checkpoint_lstm"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "lstm_ckpt.pth")

def main():
    """
    Main training loop for the stacked LSTM PPO model in the MetaMaze environment.
    With:
      - Entropy schedule from 0.10 -> 0.01 over user-chosen range
      - Checkpoint saving/loading
    """
    env_id = "MetaMazeDiscrete3D-v0"
    env = gym.make(env_id, enable_render=False)

    tasks_file = "mazes_data/train_tasks.json"
    with open(tasks_file, "r") as f:
        tasks = json.load(f)
    num_tasks = len(tasks)
    print(f"Loaded {num_tasks} tasks.")

    # Hyperparameters
    total_timesteps = 1_200_000
    steps_per_update = 80_000
    gamma = 0.99
    gae_lambda = 0.99
    clip_range = 0.2
    target_kl = 0.03
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Entropy scheduling parameters
    entropy_coef_start = 0.10
    entropy_coef_end   = 0.01
    entropy_anneal_end = 400_000   # The step at which we reach 0.01

    # Initialize the stacked LSTM policy network.
    policy_net = StackedLSTMPolicy(
        action_dim=4,
        hidden_size=512,
        num_layers=2
    ).to(device)

    # Initialize PPO trainer with default entropy (it will be updated each iteration).
    ppo_trainer = PPOTrainer(
        policy_model=policy_net,
        lr=1e-4,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        n_epochs=3,
        target_kl=target_kl,
        max_grad_norm=0.5,
        entropy_coef=entropy_coef_start,  # Will be updated each PPO cycle
        value_coef=0.5
    )

    total_steps = 0
    task_idx = 0

    # Attempt to load checkpoint if it exists
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    else:
        if os.path.isfile(CHECKPOINT_FILE):
            print(f"Found checkpoint at {CHECKPOINT_FILE}, loading...")
            ckpt = torch.load(CHECKPOINT_FILE, map_location=device)
            policy_net.load_state_dict(ckpt["policy_state_dict"])
            ppo_trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            total_steps = ckpt["total_steps"]
            print(f"Resumed training from step {total_steps}")
        else:
            print("No checkpoint file found, starting from scratch.")

    pbar = tqdm(total=total_timesteps, initial=total_steps, desc="Training")

    # Buffers for storing rollout data.
    obs_buffer = []
    act_buffer = []
    logp_buffer = []
    val_buffer = []
    rew_buffer = []
    done_buffer = []

    while total_steps < total_timesteps:
        # Update the trainer's entropy coefficient according to linear schedule
        frac = min(1.0, total_steps / float(entropy_anneal_end))
        current_entropy_coef = entropy_coef_start + frac * (entropy_coef_end - entropy_coef_start)
        ppo_trainer.entropy_coef = current_entropy_coef

        # Select the next task and configure the environment.
        task_cfg = MazeTaskSampler(**tasks[task_idx])
        env.unwrapped.set_task(task_cfg)
        task_idx = (task_idx + 1) % num_tasks

        obs_raw, _ = env.reset()
        done = False
        truncated = False

        # Reset LSTM memory for new episode
        policy_net.reset_memory(batch_size=1, device=device)

        # Initialize variables for additional channels.
        last_action = 0.0
        last_reward = 0.0
        boundary_bit = 1.0
        phase_boundary_signaled = False

        ep_obs_seq = []
        ep_actions = []
        ep_logprobs = []
        ep_values = []
        ep_rewards = []
        ep_dones = []

        while not done and not truncated and total_steps < total_timesteps:
            # Construct a 6-channel observation
            obs_img = np.transpose(obs_raw, (2, 0, 1))  # (3, H, W)
            H, W = obs_img.shape[1], obs_img.shape[2]
            c3 = np.full((1, H, W), last_action, dtype=np.float32)
            c4 = np.full((1, H, W), last_reward, dtype=np.float32)
            c5 = np.full((1, H, W), boundary_bit, dtype=np.float32)
            obs_6ch = np.concatenate([obs_img, c3, c4, c5], axis=0)

            ep_obs_seq.append(obs_6ch)

            # Convert to shape (1,1,6,H,W) for a single step forward
            obs_t = torch.from_numpy(obs_6ch[None, None]).float().to(device)
            with torch.no_grad():
                logits_t, val_t = policy_net.act_single_step(obs_t)
            dist = torch.distributions.Categorical(logits=logits_t)
            action = dist.sample()
            logp = dist.log_prob(action)

            obs_next, reward, done, truncated, info = env.step(action.item())
            total_steps += 1
            pbar.update(1)

            # Store experience
            ep_actions.append(action.item())
            ep_logprobs.append(logp.item())
            ep_values.append(val_t.item())
            ep_rewards.append(reward)
            ep_dones.append(float(done))

            last_action = float(action.item())
            last_reward = float(reward)
            obs_raw = obs_next

            if env.unwrapped.maze_core.phase == 2 and not phase_boundary_signaled:
                boundary_bit = 1.0
                phase_boundary_signaled = True
            else:
                boundary_bit = 0.0

        # Accumulate into main buffers
        obs_buffer += ep_obs_seq
        act_buffer += ep_actions
        logp_buffer += ep_logprobs
        val_buffer += ep_values
        rew_buffer += ep_rewards
        done_buffer += ep_dones

        if len(obs_buffer) >= steps_per_update:
            do_update(policy_net, ppo_trainer,
                      obs_buffer, act_buffer, logp_buffer,
                      val_buffer, rew_buffer, done_buffer, device, total_steps)
            obs_buffer.clear()
            act_buffer.clear()
            logp_buffer.clear()
            val_buffer.clear()
            rew_buffer.clear()
            done_buffer.clear()

    pbar.close()
    env.close()

    # Final PPO update
    if len(obs_buffer) > 0:
        do_update(policy_net, ppo_trainer,
                  obs_buffer, act_buffer, logp_buffer,
                  val_buffer, rew_buffer, done_buffer, device, total_steps)

    # Save final model
    save_checkpoint(policy_net, ppo_trainer, total_steps)
    print(f"Training finished after {total_steps} steps.")


def do_update(policy_net, trainer,
              obs_buf, act_buf, logp_buf,
              val_buf, rew_buf, done_buf, device, total_steps):
    """
    Conducts a PPO update using collected rollout buffers,
    standardizes the rewards, and saves checkpoint after update.
    """
    T = len(obs_buf)
    if T < 2:
        return

    obs_np = np.stack(obs_buf, axis=0)[None]      # (1,T,6,H,W)
    acts_np = np.array(act_buf, dtype=np.int64)[None]
    logp_np = np.array(logp_buf, dtype=np.float32)[None]
    vals_np = np.array(val_buf, dtype=np.float32)[None]
    rews_np = np.array(rew_buf, dtype=np.float32)[None]
    done_np = np.array(done_buf, dtype=np.float32)[None]

    B, T_ = acts_np.shape
    next_value = 0.0

    rewards_ = rews_np.reshape(-1)
    dones_   = done_np.reshape(-1)
    values_  = vals_np.reshape(-1)

    # Standardize rewards
    mean_r = np.mean(rewards_)
    std_r  = np.std(rewards_) + 1e-6
    rewards_ = (rewards_ - mean_r) / std_r

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
    print("[PPO Update]", stats)

    # Save checkpoint each update
    save_checkpoint(policy_net, trainer, total_steps)


def save_checkpoint(policy_net, trainer, total_steps):
    """
    Saves the checkpoint to the disk, including policy + optimizer states + total steps.
    """
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    torch.save({
        "policy_state_dict": policy_net.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "total_steps": total_steps
    }, CHECKPOINT_FILE)
    print(f"Checkpoint saved at step {total_steps} -> {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()
