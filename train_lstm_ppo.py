# train_lstm_ppo.py

import gymnasium as gym
import numpy as np
import torch
import random
import time
import json
import sys

import env  # Custom environment module
from lstm_ppo.lstm_model import LSTMPolicy
from lstm_ppo.ppo import PPOTrainer
from env.maze_task import MazeTaskSampler

def main():
    # Initialize environment
    env_id = "MetaMazeDiscrete3D-v0"
    env = gym.make(env_id, enable_render=False)

    # Load tasks from JSON file
    tasks_file = "mazes_data/train_tasks.json"
    with open(tasks_file, "r") as f:
        tasks = json.load(f)
    num_tasks = len(tasks)
    print(f"Loaded {num_tasks} tasks from {tasks_file}")

    # Hyperparameters
    total_timesteps = 5000000
    timesteps_per_update = 50000
    gamma = 0.99
    gae_lambda = 0.99
    clip_range = 0.2
    target_kl = 0.03
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize policy and PPO trainer
    policy = LSTMPolicy(action_dim=4, hidden_size=512).to(device)
    ppo_trainer = PPOTrainer(
        policy_model=policy,
        lr=3e-4,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        n_epochs=5,
        batch_size=1,  # Single sequence per batch
        target_kl=target_kl,
        max_grad_norm=0.5,
        entropy_coef=0.01,
        value_coef=0.5
    )

    # Training loop variables
    total_steps = 0
    episode_count = 0
    task_idx = 0

    # Buffers for a full sequence across episodes
    seq_obs, seq_actions, seq_log_probs = [], [], []
    seq_values, seq_rewards, seq_dones = [], [], []

    while total_steps < total_timesteps:
        # Load and configure the next task
        this_task_params = tasks[task_idx]
        task_config = MazeTaskSampler(**this_task_params)
        env.unwrapped.set_task(task_config)
        print(f"Starting task {task_idx + 1}/{num_tasks}")

        # Reset LSTM memory for the new task
        policy.reset_memory(batch_size=1, device=device)

        # Collect data across multiple episodes for a single task
        steps_in_this_task = 0
        while steps_in_this_task < timesteps_per_update:
            obs_raw, info = env.reset()  # Start a new episode
            done = False
            truncated = False

            while not done and not truncated:
                # Prepare observation for the model
                obs_for_model = np.transpose(obs_raw, (2, 0, 1))  # Convert to (3, 30, 40)
                obs_torch = torch.from_numpy(obs_for_model).float().to(device)

                # Get action and value prediction
                with torch.no_grad():
                    logits, value = policy.act_single_step(obs_torch)

                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

                # Step in the environment
                obs_next, reward, done, truncated, info = env.step(action.item())
                total_steps += 1
                steps_in_this_task += 1

                # Store rollout data
                seq_obs.append(obs_for_model)
                seq_actions.append(action.item())
                seq_log_probs.append(logp.item())
                seq_values.append(value.item())
                seq_rewards.append(reward)
                seq_dones.append(float(done))

                obs_raw = obs_next

                # Exit early if timestep limit for the task is reached
                if steps_in_this_task >= timesteps_per_update:
                    break

            episode_count += 1
            if steps_in_this_task >= timesteps_per_update:
                break

        # Process and compute advantages for the collected sequence
        T = len(seq_obs)
        next_value = 0.0
        if not done:
            # Estimate value for the last state if the episode wasn't completed
            obs_for_model = np.transpose(obs_raw, (2, 0, 1))
            obs_torch = torch.from_numpy(obs_for_model).float().to(device)
            with torch.no_grad():
                _, val_ = policy.act_single_step(obs_torch)
            next_value = val_.item()

        # Compute Generalized Advantage Estimation (GAE)
        rewards_np = np.array(seq_rewards, dtype=np.float32)
        dones_np = np.array(seq_dones, dtype=np.float32)
        values_np = np.array(seq_values, dtype=np.float32)

        advantages_np = ppo_trainer.compute_gae(
            rewards=rewards_np, dones=dones_np, values=values_np,
            next_value=next_value
        )
        returns_np = values_np + advantages_np

        # Prepare rollout for PPO update
        final_rollouts = {
            "obs": np.array(seq_obs, dtype=np.float32)[None],
            "actions": np.array(seq_actions, dtype=np.int64)[None],
            "old_log_probs": np.array(seq_log_probs, dtype=np.float32)[None],
            "values": values_np[None],
            "returns": returns_np[None],
            "advantages": advantages_np[None]
        }

        # Update the policy using PPO
        stats = ppo_trainer.update(final_rollouts)
        print(f"[Update] Steps={total_steps}, Episodes={episode_count}, Stats={stats}")

        # Clear sequence buffers
        seq_obs.clear()
        seq_actions.clear()
        seq_log_probs.clear()
        seq_values.clear()
        seq_rewards.clear()
        seq_dones.clear()

        # Move to the next task
        task_idx = (task_idx + 1) % num_tasks
        if total_steps >= total_timesteps:
            break

    # Save the trained model
    env.close()
    torch.save(policy.state_dict(), "models/lstm_policy.pt")
    print("Model saved.")
    print(f"Training completed after {total_steps} steps and {episode_count} episodes.")


if __name__ == "__main__":
    main()
