# train_model/train_snail_performer.py

# train_model/train_snail_performer.py

import os
import argparse
import json

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

import metamazium.env  # Import custom Maze environment definitions.
from metamazium.snail_performer.snail_model import SNAILPolicyValueNet
from metamazium.snail_performer.ppo import PPOTrainer
from metamazium.env.maze_task import MazeTaskSampler

# ---------------------------------------------------------------------------
# Standard Reward Normalizer using EMA for mean/variance.
# ---------------------------------------------------------------------------
class StandardRewardNormalizer:
    def __init__(self, alpha=0.01, epsilon=1e-6):
        self.alpha = alpha
        self.epsilon = epsilon
        self.mean = None
        self.var = None

    def update(self, rewards):
        current_mean = np.mean(rewards)
        current_var = np.var(rewards)
        if self.mean is None:
            self.mean = current_mean
            self.var = current_var
        else:
            self.mean = self.alpha * current_mean + (1 - self.alpha) * self.mean
            self.var = self.alpha * current_var + (1 - self.alpha) * self.var

    def normalize(self, rewards):
        std = np.sqrt(self.var) if self.var is not None else 1.0
        return (rewards - self.mean) / (std + self.epsilon)

# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------
def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Train SNAIL Performer on MetaMaze environment."
    )
    parser.add_argument("--total_timesteps", type=int, default=1200000, help="Total training steps.")
    parser.add_argument("--steps_per_update", type=int, default=60000, help="Steps between PPO updates.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae_lambda", type=float, default=0.99, help="GAE lambda.")
    parser.add_argument("--clip_range", type=float, default=0.1, help="Clipping range for PPO.")
    parser.add_argument("--target_kl", type=float, default=0.03, help="Target KL divergence.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for PPO.")
    parser.add_argument("--entropy_coef_start", type=float, default=0.04, help="Initial entropy coefficient.")
    parser.add_argument("--entropy_coef_end", type=float, default=0.01, help="Final entropy coefficient.")
    parser.add_argument("--entropy_anneal_end", type=int, default=500000, help="Timestep for entropy annealing.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint_snail", help="Checkpoint directory.")
    parser.add_argument("--checkpoint_file", type=str, default=None, help="Checkpoint file to load.")
    parser.add_argument("--model_save_path", type=str, default=None, help="Path to save the final model.")
    return parser.parse_args(args)

# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------
def main(args=None):
    parsed_args = parse_args(args)
    
    # Create the environment
    env_id = "MetaMazeDiscrete3D-v0"
    env = gym.make(env_id, enable_render=False)
    
    # Load tasks from the mazes_data folder
    tasks_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mazes_data", "train_tasks.json")
    if not os.path.isfile(tasks_file_path):
        print(f"Error: Tasks file not found at {tasks_file_path}")
        return
    with open(tasks_file_path, "r") as f:
        tasks = json.load(f)
    num_tasks = len(tasks)
    print(f"Loaded {num_tasks} tasks.")

    total_timesteps = parsed_args.total_timesteps
    steps_per_update = parsed_args.steps_per_update
    gamma = parsed_args.gamma
    gae_lambda = parsed_args.gae_lambda
    clip_range = parsed_args.clip_range
    target_kl = parsed_args.target_kl
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    entropy_coef_start = parsed_args.entropy_coef_start
    entropy_coef_end = parsed_args.entropy_coef_end
    entropy_anneal_end = parsed_args.entropy_anneal_end

    # Initialize the SNAIL model (replace hyperparameters as needed)
    snail = SNAILPolicyValueNet(
        action_dim=4,
        base_dim=256,
        policy_filters=32,
        policy_attn_dim=16,
        value_filters=16,
        seq_len=600,  # set appropriate sequence length for your setting
        num_policy_attn=2
    ).to(device)

    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(
        policy_model=snail,
        lr=parsed_args.lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        n_epochs=3,
        target_kl=target_kl,
        max_grad_norm=0.7,
        entropy_coef=entropy_coef_start,
        value_coef=0.5
    )

    total_steps = 0
    task_idx = 0
    # Determine checkpoint file path
    checkpoint_dir = parsed_args.checkpoint_dir
    if parsed_args.checkpoint_file:
        checkpoint_file = parsed_args.checkpoint_file
    else:
        checkpoint_file = os.path.join(checkpoint_dir, "snail_ckpt.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load checkpoint if it exists
    if os.path.isfile(checkpoint_file):
        tqdm.write(f"Found checkpoint at {checkpoint_file}, loading...")
        ckpt = torch.load(checkpoint_file, map_location=device)
        snail.load_state_dict(ckpt["policy_state_dict"])
        ppo_trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        total_steps = ckpt["total_steps"]
        tqdm.write(f"Resumed training from step {total_steps}")
    else:
        tqdm.write("No checkpoint file found, starting from scratch.")

    pbar = tqdm(total=total_timesteps, initial=total_steps, desc="Training Progress")
    reward_normalizer = StandardRewardNormalizer(alpha=0.01)

    # Rollout buffers
    obs_buffer = []
    act_buffer = []
    logp_buffer = []
    val_buffer = []
    rew_buffer = []
    done_buffer = []
    # (Phase metrics are assumed to be logged via the environment; here we do not need extra buffers.)

    while total_steps < total_timesteps:
        fraction = min(1.0, total_steps / float(entropy_anneal_end))
        ppo_trainer.entropy_coef = entropy_coef_start + fraction * (entropy_coef_end - entropy_coef_start)

        # Sample a new task and set it in the environment.
        task_config = MazeTaskSampler(**tasks[task_idx])
        env.unwrapped.set_task(task_config)
        task_idx = (task_idx + 1) % num_tasks

        obs_raw, _ = env.reset()
        done = False
        truncated = False
        snail.reset_memory(batch_size=1, device=device)

        # Initialize extra channels.
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
            # Build a 6-channel observation.
            obs_img = np.transpose(obs_raw, (2, 0, 1))  # (3, H, W)
            H, W = obs_img.shape[1], obs_img.shape[2]
            c3 = np.full((1, H, W), last_action, dtype=np.float32)
            c4 = np.full((1, H, W), last_reward, dtype=np.float32)
            c5 = np.full((1, H, W), boundary_bit, dtype=np.float32)
            obs_6ch = np.concatenate([obs_img, c3, c4, c5], axis=0)
            ep_obs_seq.append(obs_6ch)

            # Convert observation to tensor for a single step forward.
            obs_t = torch.from_numpy(obs_6ch[None, None]).float().to(device)  # Shape: (1, 1, 6, H, W)
            with torch.no_grad():
                logits_t, val_t = snail.act_single_step(obs_t)
            dist = torch.distributions.Categorical(logits=logits_t.squeeze(1))
            action = dist.sample()
            logp = dist.log_prob(action)

            # Step environment (use raw rewards, no clipping/masking)
            obs_next, reward, done, truncated, info = env.step(action.item())
            total_steps += 1
            pbar.update(1)

            ep_actions.append(action.item())
            ep_logprobs.append(logp.item())
            ep_values.append(val_t.item())
            ep_rewards.append(reward)
            ep_dones.append(float(done))

            last_action = float(action.item())
            last_reward = float(reward)
            obs_raw = obs_next

            # Update boundary bit for phase switch if applicable.
            if hasattr(env.unwrapped, 'maze_core') and env.unwrapped.maze_core.phase == 2 and not phase_boundary_signaled:
                boundary_bit = 1.0
                phase_boundary_signaled = True
            else:
                boundary_bit = 0.0

            pbar.set_postfix({
                "entropy_coef": f"{ppo_trainer.entropy_coef:.4f}",
                "last_reward": f"{reward:.2f}"
            })

            # Optionally, you can also update the progress bar with phase-specific stats
            # by reading info["phase_reports"] if your environment returns those.

        # End of episode: add rollout to buffers.
        obs_buffer.extend(ep_obs_seq)
        act_buffer.extend(ep_actions)
        logp_buffer.extend(ep_logprobs)
        val_buffer.extend(ep_values)
        rew_buffer.extend(ep_rewards)
        done_buffer.extend(ep_dones)

        # Update PPO when enough rollouts have been collected.
        if len(obs_buffer) >= steps_per_update:
            do_update(snail, ppo_trainer,
                      obs_buffer, act_buffer, logp_buffer,
                      val_buffer, rew_buffer, done_buffer,
                      device, total_steps, parsed_args, reward_normalizer)
            obs_buffer.clear()
            act_buffer.clear()
            logp_buffer.clear()
            val_buffer.clear()
            rew_buffer.clear()
            done_buffer.clear()

    pbar.close()
    env.close()

    # Final update on any remaining rollouts.
    if len(obs_buffer) > 0:
        do_update(snail, ppo_trainer,
                  obs_buffer, act_buffer, logp_buffer,
                  val_buffer, rew_buffer, done_buffer,
                  device, total_steps, parsed_args, reward_normalizer)

    save_checkpoint(snail, ppo_trainer, total_steps, parsed_args)
    print(f"Training finished after {total_steps} steps.")

# ---------------------------------------------------------------------------
# PPO Update Function (no extra advantage weighting)
# ---------------------------------------------------------------------------
def do_update(policy_net, trainer,
              obs_buf, act_buf, logp_buf,
              val_buf, rew_buf, done_buf,
              device, total_steps, args, reward_normalizer):
    if len(obs_buf) == 0:
        print("[Update] No data to update.")
        return
    T = len(obs_buf)
    if T < 2:
        return

    obs_np = np.stack(obs_buf, axis=0)[None]         # Shape: (1, T, 6, H, W)
    acts_np = np.array(act_buf, dtype=np.int64)[None]  # Shape: (1, T)
    logp_np = np.array(logp_buf, dtype=np.float32)[None]  # Shape: (1, T)
    vals_np = np.array(val_buf, dtype=np.float32)[None]  # Shape: (1, T)
    rew_np = np.array(rew_buf, dtype=np.float32)[None]   # Shape: (1, T)
    done_np = np.array(done_buf, dtype=np.float32)[None] # Shape: (1, T)

    B, T_ = acts_np.shape
    next_value = 0.0

    # Normalize rewards using the running normalizer
    rewards_reshaped = rew_np.reshape(-1)
    reward_normalizer.update(rewards_reshaped)
    normalized_rewards = reward_normalizer.normalize(rewards_reshaped)

    dones_ = done_np.reshape(-1)
    values_ = vals_np.reshape(-1)

    advantages_ = trainer.compute_gae(
        rewards=normalized_rewards,
        dones=dones_,
        values=values_,
        next_value=next_value
    )
    returns_ = values_ + advantages_

    adv_2d = advantages_.reshape(B, T_)
    ret_2d = returns_.reshape(B, T_)

    # Remove extra advantage weighting: use uniform weights.
    adv_weighted_2d = (adv_2d - adv_2d.mean()) / (adv_2d.std() + 1e-8)

    rollouts = {
        "obs": obs_np,
        "actions": acts_np,
        "old_log_probs": logp_np,
        "returns": ret_2d,
        "values": vals_np,
        "advantages": adv_weighted_2d
    }

    stats = trainer.update(rollouts)
    tqdm.write(f"[PPO Update] {stats}")
    save_checkpoint(policy_net, trainer, total_steps, args)

def save_checkpoint(policy_net, trainer, total_steps, args):
    checkpoint_file = (args.checkpoint_file if args.checkpoint_file 
                       else os.path.join(args.checkpoint_dir, "snail_ckpt.pth"))
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    torch.save({
        "policy_state_dict": policy_net.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "total_steps": total_steps
    }, checkpoint_file)
    tqdm.write(f"Checkpoint saved at step {total_steps} -> {checkpoint_file}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
