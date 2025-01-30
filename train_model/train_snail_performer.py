# train_snail_performer.py

import os
import gymnasium as gym
import numpy as np
import torch
import json
from tqdm import tqdm

import metamazium.env
from metamazium.snail_performer.snail_model import SNAILPolicyValueNet
from metamazium.snail_performer.ppo import PPOTrainer
from metamazium.env.maze_task import MazeTaskSampler

CHECKPOINT_DIR = "checkpoint_snail"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "snail_ckpt.pth")

def main():
    env_id = "MetaMazeDiscrete3D-v0"
    env = gym.make(env_id, enable_render=False)

    # Load tasks configuration
    with open("mazes_data/train_tasks.json", "r") as f:
        tasks = json.load(f)
    num_tasks = len(tasks)

    # Hyperparameters
    total_timesteps = 1200000
    steps_per_update = 60000
    gamma = 0.99
    gae_lambda = 0.99
    clip_range = 0.1
    target_kl = 0.03
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rc = 1
    # Entropy scheduling
    entropy_coef_start = 0.04
    entropy_coef_end   = 0.01
    entropy_anneal_end = 500000

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

    # Initialize PPO
    ppo_trainer = PPOTrainer(
        policy_model=snail,
        lr=0.0001,
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

    # Check if checkpoint folder/file exist
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    else:
        if os.path.isfile(CHECKPOINT_FILE):
            tqdm.write(f"Found checkpoint at {CHECKPOINT_FILE}, loading...")
            ckpt = torch.load(CHECKPOINT_FILE, map_location=device)
            snail.load_state_dict(ckpt["policy_state_dict"])
            ppo_trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            total_steps = ckpt["total_steps"]
            tqdm.write(f"Resumed training from step {total_steps}")
        else:
            tqdm.write("No checkpoint found in snail folder; starting from scratch.")

    # Main progress bar for total training steps
    pbar = tqdm(total=total_timesteps, initial=total_steps, desc="Training Progress")

    # Rollout buffers
    obs_buffer = []
    act_buffer = []
    logp_buffer = []
    val_buffer = []
    rew_buffer = []
    done_buffer = []

    # Additional buffers for phase stats
    p1_steps_buffer = []
    p2_steps_buffer = []
    p1_goal_buffer = []
    p2_goal_buffer = []

    while total_steps < total_timesteps:
        # Update entropy schedule
        fraction = min(1.0, total_steps / float(entropy_anneal_end))
        current_entropy_coef = (
            entropy_coef_start + fraction * (entropy_coef_end - entropy_coef_start)
        )
        ppo_trainer.entropy_coef = current_entropy_coef

        # pick next maze
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
            c3 = np.full((1,H,W), last_action, dtype=np.float32)
            c4 = np.full((1,H,W), last_reward, dtype=np.float32)
            c5 = np.full((1,H,W), boundary_bit, dtype=np.float32)
            obs_6ch = np.concatenate([obs_img, c3, c4, c5], axis=0)

            ep_obs_seq.append(obs_6ch)
            t_len = len(ep_obs_seq)

            obs_seq_np = np.stack(ep_obs_seq, axis=0)[None]
            obs_seq_torch = torch.from_numpy(obs_seq_np).float().to(device)
            with torch.no_grad():
                logits_seq, vals_seq = snail(obs_seq_torch)

            logits_t = logits_seq[:, t_len-1, :]
            val_t = vals_seq[:, t_len-1]

            dist = torch.distributions.Categorical(logits=logits_t)
            action = dist.sample()
            logp = dist.log_prob(action)

            obs_next, reward, done, truncated, info = env.step(action.item())
            total_steps += 1
            pbar.update(1)  # update main progress bar by 1 step

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

            # Periodically update pbar's postfix with short info
            pbar.set_postfix({
                "entropy_coef": f"{ppo_trainer.entropy_coef:.4f}",
                "last_reward":  f"{reward:.2f}"
            })

            # If the episode ended, parse phase stats
            if done or truncated:
                p1_total_rew, p2_total_rew = 0.0, 0.0
                p1_steps, p2_steps = 0, 0
                p1_goal_reached, p2_goal_reached = False, False

                if "phase_reports" in info:
                    ph1 = info["phase_reports"].get("Phase1", {})
                    ph2 = info["phase_reports"].get("Phase2", {})

                    # raw sums
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
                
                    # Use tqdm.write to not break the bar
                    tqdm.write(f"=== Detailed Episode Report{rc} ===")
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
                    tqdm.write(f"    => Phase2 Total:      {p2_total_rew:.3f}")
                    rc +=1
                # pbar.set_description_str(
                #     f"Ph1= {p1_steps} steps, R= {p1_total_rew:.3f} | "
                #     f"Ph2= {p2_steps} steps, R= {p2_total_rew:.3f}"
                # )

                # Tag the entire episode with p1/p2 info
                ep_len = len(ep_rewards)
                ep_p1_steps = [p1_steps] * ep_len
                ep_p2_steps = [p2_steps] * ep_len
                ep_p1_goals = [p1_goal_reached] * ep_len
                ep_p2_goals = [p2_goal_reached] * ep_len

                # Extend global buffers
                p1_steps_buffer.extend(ep_p1_steps)
                p2_steps_buffer.extend(ep_p2_steps)
                p1_goal_buffer.extend(ep_p1_goals)
                p2_goal_buffer.extend(ep_p2_goals)

        # Summarize this episode in buffers
        obs_buffer.extend(ep_obs_seq)
        act_buffer.extend(ep_actions)
        logp_buffer.extend(ep_logprobs)
        val_buffer.extend(ep_values)
        rew_buffer.extend(ep_rewards)
        done_buffer.extend(ep_dones)

        # Check if it's time to update PPO
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
                total_steps
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

    pbar.close()
    env.close()

    # final update if leftover
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
            total_steps
        )

    # save final model
    save_checkpoint(snail, ppo_trainer, total_steps)
    tqdm.write(f"Training finished after {total_steps} steps.")


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
    total_steps
):
    # Determine the minimum buffer length across all relevant buffers
    min_len = min(len(obs_buf), len(p1_steps_buf), len(p2_steps_buf), len(p1_goal_buf), len(p2_goal_buf))

    # Check for buffer length mismatches
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

    # Proceed only if there's sufficient data to update
    if min_len < 2:
        tqdm.write("[Info] Not enough data to perform PPO update.")
        return {}

    # Convert lists to NumPy arrays
    obs_np = np.stack(obs_buf, axis=0)[None]  # Shape: (1, T, 6, H, W)
    acts_np = np.array(act_buf, dtype=np.int64)[None]  # Shape: (1, T)
    logp_np = np.array(logp_buf, dtype=np.float32)[None]  # Shape: (1, T)
    vals_np = np.array(val_buf, dtype=np.float32)[None]  # Shape: (1, T)
    rews_np = np.array(rew_buf, dtype=np.float32)[None]  # Shape: (1, T)
    done_np = np.array(done_buf, dtype=np.float32)[None]  # Shape: (1, T)

    # Phase stats
    p1_steps_np = np.array(p1_steps_buf, dtype=np.float32)[None]    # Shape: (1, T)
    p2_steps_np = np.array(p2_steps_buf, dtype=np.float32)[None]    # Shape: (1, T)
    p1_goals_np = np.array(p1_goal_buf,  dtype=bool)[None]          # Shape: (1, T)
    p2_goals_np = np.array(p2_goal_buf,  dtype=bool)[None]          # Shape: (1, T)

    B, T_ = acts_np.shape
    next_value = 0.0

    rewards_ = rews_np.reshape(-1)
    dones_   = done_np.reshape(-1)
    values_  = vals_np.reshape(-1)

    # Standardize rewards (for advantage calculations)
    mean_r = np.mean(rewards_)
    std_r  = np.std(rewards_) + 1e-6
    rewards_ = (rew_buf[:min_len] - mean_r) / std_r  # Adjusted to use truncated rew_buf

    # Compute GAE
    advantages_ = trainer.compute_gae(rewards_, dones_[:min_len], values_[:min_len], next_value)
    returns_ = values_[:min_len] + advantages_

    adv_2d = advantages_.reshape(B, T_)
    ret_2d = returns_.reshape(B, T_)

    # Convert to 1D for weighting
    adv_1d = adv_2d.reshape(-1)

    # Masks for the 3 Cases:
    p1_s = p1_steps_np.reshape(-1)  # Shape: (T,)
    p2_s = p2_steps_np.reshape(-1)
    p1_g = p1_goals_np.reshape(-1).astype(np.float32)
    p2_g = p2_goals_np.reshape(-1).astype(np.float32)

    # Case1: p2 < p1 -> Boost
    mask_case1 = (p2_s < p1_s).astype(np.float32)

    # Case3: no goals in either phase
    mask_case3 = ((p1_g == 0) & (p2_g == 0)).astype(np.float32)

    # Define weighting factors
    #  +0.2 for case1, -0.5 if case3 
    # Final factor = 1 + 0.2 * mask_case1 - 0.5 * mask_case3
    weights = 1.0 + 0.2 * mask_case1 - 0.5 * mask_case3
    # Keep them in [0.1, 2.0] to avoid extremes
    weights = np.clip(weights, 0.1, 2.0)

    # Apply weighting
    adv_weighted = adv_1d * weights

    # Reshape back to (B, T)
    adv_weighted_2d = adv_weighted.reshape(B, T_)

    # Normalize again after weighting
    adv_weighted_2d = (adv_weighted_2d - adv_weighted_2d.mean()) / (adv_weighted_2d.std() + 1e-8)

    # Construct rollouts with the newly weighted advantages
    rollouts = {
        "obs": obs_np,
        "actions": acts_np,
        "old_log_probs": logp_np,
        "returns": ret_2d,
        "values": vals_np,
        "advantages": adv_weighted_2d
    }

    # Perform PPO update
    stats = trainer.update(rollouts)
    tqdm.write(f"[PPO Update] {stats}")  # Log PPO update stats

    # Save checkpoint after each update
    save_checkpoint(snail_net, trainer, total_steps)

    return stats  # Return stats for potential further use

def save_checkpoint(snail_net, trainer, total_steps):
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "policy_state_dict": snail_net.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "total_steps": total_steps
    }, CHECKPOINT_FILE)
    tqdm.write(f"Checkpoint saved at step {total_steps} -> {CHECKPOINT_FILE}")

if __name__ == "__main__":
    main()
