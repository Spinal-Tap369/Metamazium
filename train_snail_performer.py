# train_snail_performer.py

import gymnasium as gym
import numpy as np
import torch
import json
from tqdm import tqdm  # For progress bar

import env  # Ensure your custom env is registered
from snail_performer.performer_model import SNAILPerformerPolicy
from snail_performer.ppo import PPOTrainer
from env.maze_task import MazeTaskSampler

def main():
    env_id = "MetaMazeDiscrete3D-v0"
    env = gym.make(env_id, enable_render=False)

    # Load tasks
    tasks_file = "mazes_data/train_tasks.json"
    with open(tasks_file, "r") as f:
        tasks = json.load(f)
    num_tasks = len(tasks)
    tqdm.write(f"Loaded {num_tasks} tasks from {tasks_file}")

    # Hyperparameters
    total_timesteps = 800000
    timesteps_per_update = 50000
    gamma = 0.99
    gae_lambda = 0.99
    clip_range = 0.2
    target_kl = 0.03
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize policy and PPO trainer
    policy = SNAILPerformerPolicy(
        action_dim=4,
        base_dim=256,
        num_tc_blocks=2,
        tc_filters=32,
        attn_heads=8,
        attn_dim_head=256,
        nb_features=256,
        causal=False
    ).to(device)

    ppo_trainer = PPOTrainer(
        policy_model=policy,
        lr=1e-5,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        n_epochs=5,
        batch_size=8,
        target_kl=target_kl,
        max_grad_norm=0.3,
        entropy_coef=0.01,
        value_coef=0.5
    )

    total_steps = 0
    episode_count = 0
    task_idx = 0

    # Global buffers for collecting transitions.
    seq_obs = []
    seq_actions = []
    seq_log_probs = []
    seq_values = []
    seq_rewards = []
    seq_dones = []

    # Initialize step counter since last PPO update
    steps_since_update = 0

    # Initialize tqdm progress bar
    pbar = tqdm(total=total_timesteps, desc="Training Progress", dynamic_ncols=True)

    while total_steps < total_timesteps:
        # Pick next maze task
        this_task_params = tasks[task_idx]
        task_config = MazeTaskSampler(**this_task_params)
        env.unwrapped.set_task(task_config)
        tqdm.write(f"*** Starting new maze task {task_idx+1}/{num_tasks} ***")

        # Reset boundaries and previous action/reward at start of new episode
        last_action = 0.0  
        last_reward = 0.0
        boundary_flag = 1.0  
        phase_boundary_signaled = False

        # Run exactly ONE full episode in this maze
        obs_raw, info = env.reset()
        done = False
        truncated = False

        episode_obs_list = []

        while not done and not truncated:
            # Check for phase 2 boundary transition
            if env.unwrapped.maze_core.phase == 2 and not phase_boundary_signaled:
                boundary_flag = 1.0
                phase_boundary_signaled = True
            else:
                boundary_flag = 0.0

            # Build 6-channel input:
            obs_image = np.transpose(obs_raw, (2, 0, 1))  # shape (3,H,W)
            H, W = obs_image.shape[1], obs_image.shape[2]
            channel3 = np.full((1, H, W), last_action, dtype=np.float32)
            channel4 = np.full((1, H, W), last_reward, dtype=np.float32)
            channel5 = np.full((1, H, W), boundary_flag, dtype=np.float32)

            # Concatenate channels to form 6-channel observation
            obs_6ch = np.concatenate((obs_image, channel3, channel4, channel5), axis=0)  # shape (6,H,W)
            episode_obs_list.append(obs_6ch)

            # Build sequence so far: shape (1, t, 6, H, W)
            t = len(episode_obs_list)
            full_obs_seq = np.array(episode_obs_list, dtype=np.float32)[None]  # (1,t,6,H,W)
            full_obs_seq_torch = torch.from_numpy(full_obs_seq).float().to(device)

            # Get action logits for the last timestep
            with torch.no_grad():
                logits_last = policy.act_online_sequence(full_obs_seq_torch)

            # Sample action from the distribution
            dist = torch.distributions.Categorical(logits=logits_last)
            action = dist.sample()
            logp = dist.log_prob(action)

            # Step the environment with the sampled action
            obs_next, reward, done, truncated, info = env.step(action.item())
            total_steps += 1
            steps_since_update += 1

            # Save transition data
            seq_obs.append(obs_6ch)
            seq_actions.append(action.item())
            seq_log_probs.append(logp.item())
            seq_rewards.append(reward)
            seq_dones.append(float(done))
            # We will fill seq_values later

            # Update last_action and last_reward for next timestep
            last_action = float(action.item())
            last_reward = float(reward)

            obs_raw = obs_next

            # Update tqdm progress bar dynamically
            current_phase = env.unwrapped.maze_core.phase
            phase_steps = env.unwrapped.maze_core.current_phase_steps
            pbar.update(1)
            pbar.set_postfix({
                "Phase": current_phase,
                "Phase Steps": phase_steps,
                "Total Steps": total_steps,
            })

            if steps_since_update >= timesteps_per_update:
                tqdm.write(f"Collected {steps_since_update} transitions so far. Doing PPO update...")

                # Next value estimation
                next_value = 0.0
                if not done:
                    obs_image = np.transpose(obs_raw, (2, 0, 1))
                    H, W = obs_image.shape[1], obs_image.shape[2]
                    channel3 = np.full((1, H, W), last_action, dtype=np.float32)
                    channel4 = np.full((1, H, W), last_reward, dtype=np.float32)
                    channel5 = np.full((1, H, W), boundary_flag, dtype=np.float32)
                    obs_6ch_final = np.concatenate((obs_image, channel3, channel4, channel5), axis=0)
                    full_obs_seq_final = torch.from_numpy(obs_6ch_final).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,6,H,W)
                    with torch.no_grad():
                        logits, final_val = policy.forward(full_obs_seq_final)
                        next_value = final_val[0,0].item()
                else:
                    next_value = 0.0

                # Build observation array for value estimation
                T_ = len(seq_obs)
                full_obs_array = np.array(seq_obs, dtype=np.float32)[None]  # shape (1,T_,6,H,W)

                with torch.no_grad():
                    full_obs_t = torch.from_numpy(full_obs_array).float().to(device)
                    _, all_values_t = policy.forward(full_obs_t)  # shape (1, T_)
                    all_values_np = all_values_t.cpu().numpy().squeeze(0)

                seq_values = all_values_np.tolist()

                rewards_np = np.array(seq_rewards, dtype=np.float32)
                dones_np   = np.array(seq_dones, dtype=np.float32)
                values_np  = np.array(seq_values, dtype=np.float32)

                advantages_np = ppo_trainer.compute_gae(
                    rewards=rewards_np,
                    dones=dones_np,
                    values=values_np,
                    next_value=next_value
                )
                returns_np = values_np + advantages_np

                final_rollouts = {
                    "obs": full_obs_array,  # shape (1,T_,6,H,W)
                    "actions": np.array(seq_actions, dtype=np.int64)[None],
                    "old_log_probs": np.array(seq_log_probs, dtype=np.float32)[None],
                    "values": values_np[None],
                    "returns": returns_np[None],
                    "advantages": advantages_np[None]
                }

                stats = ppo_trainer.update(final_rollouts)
                tqdm.write(f"[Update] Steps={total_steps}, Episode={episode_count}, Stats={stats}")

                seq_obs.clear()
                seq_actions.clear()
                seq_log_probs.clear()
                seq_values.clear()
                seq_rewards.clear()
                seq_dones.clear()
                steps_since_update = 0

        episode_count += 1
        task_idx = (task_idx + 1) % num_tasks

    pbar.close()
    env.close()
    torch.save(policy.state_dict(), "models/snail_performer_policy_sequence.pt")
    tqdm.write("Model saved.")
    tqdm.write(f"Training completed after {total_steps} steps and {episode_count} episodes.")

if __name__ == "__main__":
    main()
