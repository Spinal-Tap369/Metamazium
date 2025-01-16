# train_snail_performer.py

import gymnasium as gym
import numpy as np
import torch
import json
import sys

import env  # Custom environment module
from snail_performer.performer_model import SNAILPerformerPolicy
from snail_performer.ppo import PPOTrainer
from env.maze_task import MazeTaskSampler

def main():
    # Initialize the environment
    env_id = "MetaMazeDiscrete3D-v0"
    env = gym.make(env_id, enable_render=False)

    # Load maze tasks from the JSON file
    tasks_file = "mazes_data/train_tasks.json"
    with open(tasks_file, "r") as f:
        tasks = json.load(f)
    num_tasks = len(tasks)
    print(f"Loaded {num_tasks} tasks from {tasks_file}")

    # Hyperparameters for training
    total_timesteps = 5000000 
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
        lr=0.00001,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        n_epochs=5,
        batch_size=8,        # Mini-batch size
        target_kl=target_kl,
        max_grad_norm=0.3,
        entropy_coef=0.01,
        value_coef=0.5
    )

    # Training loop variables
    total_steps = 0
    episode_count = 0
    task_idx = 0

    # Buffers for collecting sequence data
    seq_obs, seq_actions, seq_log_probs = [], [], []
    seq_values, seq_rewards, seq_dones = [], [], []

    while total_steps < total_timesteps:
        # Load and set the next maze task
        this_task_params = tasks[task_idx]
        task_config = MazeTaskSampler(**this_task_params)
        env.unwrapped.set_task(task_config)
        print(f"Starting task {task_idx+1}/{num_tasks}")

        steps_collected = 0
        while steps_collected < timesteps_per_update:
            # Reset environment for a new episode
            obs_raw, info = env.reset()
            done = False
            truncated = False

            while not done and not truncated:
                # Preprocess observation for the policy
                obs_for_model = np.transpose(obs_raw, (2, 0, 1))  # (3,30,40)
                obs_torch = torch.from_numpy(obs_for_model).float().to(device)

                # Get action and value prediction from the policy
                with torch.no_grad():
                    logits, value = policy.act_single_step(obs_torch)

                # Sample action from the policy distribution
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

                # Step in the environment
                obs_next_raw, reward, done, truncated, info = env.step(action.item())

                # Update counters and store rollout data
                total_steps += 1
                steps_collected += 1

                seq_obs.append(obs_for_model)
                seq_actions.append(action.item())
                seq_log_probs.append(logp.item())
                seq_values.append(value.item())
                seq_rewards.append(reward)
                seq_dones.append(float(done))

                obs_raw = obs_next_raw

                # Exit if task collection limit is reached
                if steps_collected >= timesteps_per_update:
                    break

            episode_count += 1
            if steps_collected >= timesteps_per_update:
                break

        # Calculate sequence length and prepare for PPO update
        T = len(seq_obs)
        print(f"Collected {T} transitions. Updating PPO...")

        # Estimate next value for incomplete episodes
        next_value = 0.0
        if not done:
            obs_for_model = np.transpose(obs_raw, (2, 0, 1))
            obs_torch = torch.from_numpy(obs_for_model).float().to(device)
            with torch.no_grad():
                _, val_ = policy.act_single_step(obs_torch)
            next_value = val_.item()

        # Convert collected data to numpy arrays
        rewards_np = np.array(seq_rewards, dtype=np.float32)
        dones_np = np.array(seq_dones, dtype=np.float32)
        values_np = np.array(seq_values, dtype=np.float32)

        # Compute advantages and returns using GAE
        advantages_np = ppo_trainer.compute_gae(
            rewards=rewards_np,
            dones=dones_np,
            values=values_np,
            next_value=next_value
        )
        returns_np = values_np + advantages_np

        # Prepare rollouts for PPO
        final_rollouts = {
            "obs": np.array(seq_obs, dtype=np.float32)[None],           # (1, T, 3, 30, 40)
            "actions": np.array(seq_actions, dtype=np.int64)[None],     # (1, T)
            "old_log_probs": np.array(seq_log_probs, dtype=np.float32)[None],
            "values": values_np[None],
            "returns": returns_np[None],
            "advantages": advantages_np[None]
        }

        # Update policy using PPO
        stats = ppo_trainer.update(final_rollouts)
        print(f"[Update] Steps={total_steps}, Episodes={episode_count}, Stats={stats}")

        # Clear collected sequences
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

    # Save trained model and cleanup
    env.close()
    torch.save(policy.state_dict(), "models/snail_performer_policy_sequence.pt")
    print("Model saved.")
    print(f"Training completed after {total_steps} steps and {episode_count} episodes.")


if __name__ == "__main__":
    main()
