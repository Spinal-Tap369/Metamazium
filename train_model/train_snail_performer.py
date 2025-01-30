# train_model/train_snail_performer.py

import os
import argparse
import json

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

# Import from your metamazium package
from metamazium.env.maze_task import MazeTaskSampler
from metamazium.snail_performer.snail_model import SNAILPolicyValueNet
from metamazium.snail_performer.ppo import PPOTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train SNAIL Performer on MetaMaze environment.")

    # Training hyperparameters
    parser.add_argument("--total_timesteps", type=int, default=1200000, help="Total number of training steps.")
    parser.add_argument("--steps_per_update", type=int, default=60000, help="Number of steps between PPO updates.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae_lambda", type=float, default=0.99, help="GAE lambda.")
    parser.add_argument("--clip_range", type=float, default=0.1, help="Clip range for PPO.")
    parser.add_argument("--target_kl", type=float, default=0.03, help="Target KL divergence.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for PPO.")
    
    # Entropy settings
    parser.add_argument("--entropy_coef_start", type=float, default=0.04, help="Initial entropy coefficient.")
    parser.add_argument("--entropy_coef_end", type=float, default=0.01, help="Final entropy coefficient.")
    parser.add_argument("--entropy_anneal_end", type=int, default=500000, help="Timestep by which entropy coefficient is annealed.")
    
    # Checkpoints and model saving
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint_snail", help="Directory to save/load checkpoints.")
    parser.add_argument("--checkpoint_file", type=str, default=None, help="Path to a specific checkpoint file to load. Overrides checkpoint_dir if provided.")
    parser.add_argument("--model_save_path", type=str, default=None, help="Path to save the final model (optional).")

    return parser.parse_args()


def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine checkpoint file path
    if args.checkpoint_file is not None:
        checkpoint_file = args.checkpoint_file
        checkpoint_dir = os.path.dirname(checkpoint_file) if checkpoint_file else args.checkpoint_dir
    else:
        checkpoint_dir = args.checkpoint_dir
        checkpoint_file = os.path.join(checkpoint_dir, "snail_ckpt.pth")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create the environment
    env_id = "MetaMazeDiscrete3D-v0"
    env = gym.make(env_id, enable_render=False)

    # Load tasks configuration from mazes_data in the main directory
    # Assuming the current working directory is the main directory where setup.py is located
    tasks_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mazes_data", "train_tasks.json")
    tasks_file_path = os.path.normpath(tasks_file_path)  # Normalize path for different OS

    if not os.path.isfile(tasks_file_path):
        print(f"Error: Tasks file not found at {tasks_file_path}")
        return

    with open(tasks_file_path, "r") as f:
        tasks = json.load(f)
    num_tasks = len(tasks)

    # PPO hyperparameters
    total_timesteps = args.total_timesteps
    steps_per_update = args.steps_per_update
    gamma = args.gamma
    gae_lambda = args.gae_lambda
    clip_range = args.clip_range
    target_kl = args.target_kl

    # Entropy scheduling
    entropy_coef_start = args.entropy_coef_start
    entropy_coef_end   = args.entropy_coef_end
    entropy_anneal_end = args.entropy_anneal_end

    # Initialize SNAIL model
    snail = SNAILPolicyValueNet(
        action_dim=4,
        base_dim=256,
        policy_filters=32,
        policy_attn_dim=16,
        value_filters=16,
        seq_len=800,
        num_policy_attn=2
    ).to(device)

    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(
        policy_model=snail,
        lr=args.lr,
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

    # Check if checkpoint file exists
    if os.path.isfile(checkpoint_file):
        tqdm.write(f"Found checkpoint at {checkpoint_file}, loading...")
        ckpt = torch.load(checkpoint_file, map_location=device)
        snail.load_state_dict(ckpt["policy_state_dict"])
        ppo_trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        total_steps = ckpt["total_steps"]
        tqdm.write(f"Resumed training from step {total_steps}")
    else:
        tqdm.write("No valid checkpoint found; starting from scratch.")

    # Main progress bar
    pbar = tqdm(total=total_timesteps, initial=total_steps, desc="Training Progress")

    # Rollout buffers
    obs_buffer = []
    act_buffer = []
    logp_buffer = []
    val_buffer = []
    rew_buffer = []
    done_buffer = []

    # Phase stats buffers
    p1_steps_buffer = []
    p2_steps_buffer = []
    p1_goal_buffer = []
    p2_goal_buffer = []

    while total_steps < total_timesteps:
        # Update entropy schedule
        fraction = min(1.0, total_steps / float(entropy_anneal_end))
        current_entropy_coef = entropy_coef_start + fraction * (entropy_coef_end - entropy_coef_start)
        ppo_trainer.entropy_coef = current_entropy_coef

        # Choose next maze
        task_config = MazeTaskSampler(**tasks[task_idx])
        env.unwrapped.set_task(task_config)
        task_idx = (task_idx + 1) % num_tasks

        obs_raw, _ = env.reset()
        done = False
        truncated = False

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
            # Construct 6-channel observation
            obs_img = np.transpose(obs_raw, (2, 0, 1))  # (3,H,W)
            H, W = obs_img.shape[1], obs_img.shape[2]
            c3 = np.full((1, H, W), last_action, dtype=np.float32)
            c4 = np.full((1, H, W), last_reward, dtype=np.float32)
            c5 = np.full((1, H, W), boundary_bit, dtype=np.float32)
            obs_6ch = np.concatenate([obs_img, c3, c4, c5], axis=0)

            ep_obs_seq.append(obs_6ch)
            t_len = len(ep_obs_seq)

            obs_seq_np = np.stack(ep_obs_seq, axis=0)[None]
            obs_seq_torch = torch.from_numpy(obs_seq_np).float().to(device)

            with torch.no_grad():
                logits_seq, vals_seq = snail(obs_seq_torch)

            logits_t = logits_seq[:, t_len - 1, :]
            val_t = vals_seq[:, t_len - 1]

            dist = torch.distributions.Categorical(logits=logits_t)
            action = dist.sample()
            logp = dist.log_prob(action)

            obs_next, reward, done, truncated, info = env.step(action.item())
            total_steps += 1
            pbar.update(1)

            # Store transition
            ep_actions.append(action.item())
            ep_logprobs.append(logp.item())
            ep_values.append(val_t.item())
            ep_rewards.append(reward)
            ep_dones.append(float(done))

            last_action = float(action.item())
            last_reward = float(reward)
            obs_raw = obs_next

            # Boundary bit when phase changes from 1 -> 2
            if hasattr(env.unwrapped, 'maze_core') and env.unwrapped.maze_core.phase == 2 and not phase_boundary_signaled:
                boundary_bit = 1.0
                phase_boundary_signaled = True
            else:
                boundary_bit = 0.0

            pbar.set_postfix({
                "entropy_coef": f"{ppo_trainer.entropy_coef:.4f}",
                "last_reward":  f"{reward:.2f}"
            })

            # If episode ended, parse phase stats
            if done or truncated:
                p1_total_rew, p2_total_rew = 0.0, 0.0
                p1_steps, p2_steps = 0, 0
                p1_goal_reached, p2_goal_reached = False, False

                if "phase_reports" in info:
                    ph1 = info["phase_reports"].get("Phase1", {})
                    ph2 = info["phase_reports"].get("Phase2", {})

                    # Raw sums
                    p1_goal_rew  = ph1.get("Goal Rewards", 0.0)
                    p1_step_rew  = ph1.get("Total Step Rewards", 0.0)
                    p1_coll_rew  = ph1.get("Total Collision Rewards", 0.0)
                    p1_total_rew = p1_goal_rew + p1_step_rew + p1_coll_rew
                    p1_steps     = ph1.get("Total Steps", 0)
                    p1_goal_reached = ph1.get("Goal Reached", False)

                    p2_goal_rew  = ph2.get("Goal Rewards", 0.0)
                    p2_step_rew  = ph2.get("Total Step Rewards", 0.0)
                    p2_coll_rew  = ph2.get("Total Collision Rewards", 0.0)
                    p2_total_rew = p2_goal_rew + p2_step_rew + p2_coll_rew
                    p2_steps     = ph2.get("Total Steps", 0)
                    p2_goal_reached = ph2.get("Goal Reached", False)

                    tqdm.write(f"\n=== Detailed Episode Report ===")
                    tqdm.write(f"  Phase1:")
                    tqdm.write(f"    Steps:                {p1_steps}")
                    tqdm.write(f"    Step Rewards:         {p1_step_rew:.3f}")
                    tqdm.write(f"    Goal Rewards:         {p1_goal_rew:.3f}")
                    tqdm.write(f"    Collision Penalties:  {p1_coll_rew:.3f}")
                    tqdm.write(f"    => Phase1 Total:      {p1_total_rew:.3f}")

                    tqdm.write(f"  Phase2:")
                    tqdm.write(f"    Steps:                {p2_steps}")
                    tqdm.write(f"    Step Rewards:         {p2_step_rew:.3f}")
                    tqdm.write(f"    Goal Rewards:         {p2_goal_rew:.3f}")
                    tqdm.write(f"    Collision Penalties:  {p2_coll_rew:.3f}")
                    tqdm.write(f"    => Phase2 Total:      {p2_total_rew:.3f}\n")

                ep_len = len(ep_rewards)
                ep_p1_steps = [p1_steps] * ep_len
                ep_p2_steps = [p2_steps] * ep_len
                ep_p1_goals = [p1_goal_reached] * ep_len
                ep_p2_goals = [p2_goal_reached] * ep_len

                p1_steps_buffer.extend(ep_p1_steps)
                p2_steps_buffer.extend(ep_p2_steps)
                p1_goal_buffer.extend(ep_p1_goals)
                p2_goal_buffer.extend(ep_p2_goals)

        # Summarize the episode
        obs_buffer.extend(ep_obs_seq)
        act_buffer.extend(ep_actions)
        logp_buffer.extend(ep_logprobs)
        val_buffer.extend(ep_values)
        rew_buffer.extend(ep_rewards)
        done_buffer.extend(ep_dones)

        # Update PPO if needed
        if len(obs_buffer) >= steps_per_update:
            do_update(
                snail,
                ppo_trainer,
                obs_buffer,
                act_buffer,
                logp_buffer,
                val_buffer,
                rew_buffer,
                done_buffer,
                p1_steps_buffer,
                p2_steps_buffer,
                p1_goal_buffer,
                p2_goal_buffer,
                device,
                total_steps,
                checkpoint_file
            )
            # Clear buffers
            obs_buffer.clear()
            act_buffer.clear()
            logp_buffer.clear()
            val_buffer.clear()
            rew_buffer.clear()
            done_buffer.clear()
            p1_steps_buffer.clear()
            p2_steps_buffer.clear()
            p1_goal_buffer.clear()
            p2_goal_buffer.clear()

    # Final cleanup if leftover
    if len(obs_buffer) > 0:
        do_update(
            snail,
            ppo_trainer,
            obs_buffer,
            act_buffer,
            logp_buffer,
            val_buffer,
            rew_buffer,
            done_buffer,
            p1_steps_buffer,
            p2_steps_buffer,
            p1_goal_buffer,
            p2_goal_buffer,
            device,
            total_steps,
            checkpoint_file
        )

    # Save final model if specified
    if args.model_save_path:
        torch.save(snail.state_dict(), args.model_save_path)
        tqdm.write(f"Model saved at {args.model_save_path}")

    tqdm.write(f"Training finished after {total_steps} steps.")
    env.close()


def do_update(
    snail_net,
    trainer,
    obs_buf,
    act_buf,
    logp_buf,
    val_buf,
    rew_buf,
    done_buf,
    p1_steps_buf,
    p2_steps_buf,
    p1_goal_buf,
    p2_goal_buf,
    device,
    total_steps,
    checkpoint_file
):
    from tqdm import tqdm
    import numpy as np

    min_len = min(len(obs_buf), len(p1_steps_buf), len(p2_steps_buf), len(p1_goal_buf), len(p2_goal_buf))

    if min_len < len(obs_buf):
        tqdm.write(f"[Warning] Truncating buffers from {len(obs_buf)} to {min_len} to synchronize buffer lengths.")
        # Truncate all buffers to the minimum length to prevent broadcasting errors
        obs_buf = obs_buf[:min_len]
        act_buf = act_buf[:min_len]
        logp_buf = logp_buf[:min_len]
        val_buf = val_buf[:min_len]
        rew_buf = rew_buf[:min_len]
        done_buf = done_buf[:min_len]
        p1_steps_buf = p1_steps_buf[:min_len]
        p2_steps_buf = p2_steps_buf[:min_len]
        p1_goal_buf = p1_goal_buf[:min_len]
        p2_goal_buf = p2_goal_buf[:min_len]

    if min_len < 2:
        tqdm.write("[Info] Not enough data to perform PPO update.")
        return

    obs_np = np.stack(obs_buf, axis=0)[None]
    acts_np = np.array(act_buf, dtype=np.int64)[None]
    logp_np = np.array(logp_buf, dtype=np.float32)[None]
    vals_np = np.array(val_buf, dtype=np.float32)[None]
    rews_np = np.array(rew_buf, dtype=np.float32)[None]
    done_np = np.array(done_buf, dtype=np.float32)[None]

    p1_steps_np = np.array(p1_steps_buf, dtype=np.float32)[None]
    p2_steps_np = np.array(p2_steps_buf, dtype=np.float32)[None]
    p1_goals_np = np.array(p1_goal_buf, dtype=bool)[None]
    p2_goals_np = np.array(p2_goal_buf, dtype=bool)[None]

    B, T_ = acts_np.shape
    next_value = 0.0

    rewards_ = rews_np.reshape(-1)
    dones_ = done_np.reshape(-1)
    values_ = vals_np.reshape(-1)

    mean_r = np.mean(rewards_)
    std_r = np.std(rewards_) + 1e-6
    rewards_ = (rewards_ - mean_r) / std_r

    advantages_ = trainer.compute_gae(rewards_, dones_, values_, next_value)
    returns_ = values_ + advantages_

    adv_2d = advantages_.reshape(B, T_)
    ret_2d = returns_.reshape(B, T_)

    adv_1d = adv_2d.reshape(-1)

    p1_s = p1_steps_np.reshape(-1)
    p2_s = p2_steps_np.reshape(-1)
    p1_g = p1_goals_np.reshape(-1).astype(np.float32)
    p2_g = p2_goals_np.reshape(-1).astype(np.float32)

    mask_case1 = (p2_s < p1_s).astype(np.float32)
    mask_case3 = ((p1_g == 0) & (p2_g == 0)).astype(np.float32)

    weights = 1.0 + 0.2 * mask_case1 - 0.5 * mask_case3
    weights = np.clip(weights, 0.1, 2.0)
    adv_weighted = adv_1d * weights

    adv_weighted_2d = adv_weighted.reshape(B, T_)
    adv_weighted_2d = (adv_weighted_2d - adv_weighted_2d.mean()) / (adv_weighted_2d.std() + 1e-8)

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
    save_checkpoint(snail_net, trainer, total_steps, checkpoint_file)


def save_checkpoint(snail_net, trainer, total_steps, checkpoint_file):
    import os
    from tqdm import tqdm

    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    torch.save({
        "policy_state_dict": snail_net.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "total_steps": total_steps
    }, checkpoint_file)
    tqdm.write(f"Checkpoint saved at step {total_steps} -> {checkpoint_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
