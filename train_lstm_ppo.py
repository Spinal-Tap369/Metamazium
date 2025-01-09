# train_lstm_ppo.py

import gymnasium as gym
import numpy as np
import torch
import random
import time

import json
import sys

import env  # Make sure your env package is imported so "MetaMazeDiscrete3D-v0" is registered
from lstm_ppo.lstm_model import LSTMPolicy
from lstm_ppo import PPOTrainer

# Import MazeTaskSampler to generate MazeTask configs
from env.maze_task import MazeTaskSampler


def main():
    env_id = "MetaMazeDiscrete3D-v0"
    env = gym.make(env_id, enable_render=False)

    # 1) Load tasks from JSON (which has 1000 mazes). We'll iterate them in order.
    tasks_file = "mazes_data/train_tasks.json"
    with open(tasks_file, "r") as f:
        tasks = json.load(f)
    num_tasks = len(tasks)
    print(f"Loaded {num_tasks} tasks from {tasks_file}")

    # We'll maintain a pointer, task_idx, to cycle through tasks one by one each episode.
    task_idx = 0

    # Hyperparameters
    total_timesteps = 500000
    timesteps_per_update = 50000
    gamma = 0.99
    gae_lambda = 0.99
    clip_range = 0.2
    target_kl = 0.01

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create policy + PPO
    policy = LSTMPolicy(action_dim=4, hidden_size=512).to(device)
    ppo_trainer = PPOTrainer(
        policy_model=policy,
        lr=3e-4,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        n_epochs=10,
        batch_size=4,
        target_kl=target_kl,
        max_grad_norm=0.5,
        entropy_coef=0.0,
        value_coef=0.5
    )

    total_steps = 0
    episode_count = 0

    # We'll store multiple episodes to do a PPO update every timesteps_per_update
    all_obs = []
    all_actions = []
    all_log_probs = []
    all_values = []
    all_rewards = []
    all_dones = []

    while total_steps < total_timesteps:
        # ---------------------------------------------
        # 2) Set a new maze each episode
        # Convert the next task's dictionary into a TaskConfig
        this_task_params = tasks[task_idx]
        task_config = MazeTaskSampler(**this_task_params)
        # Now call set_task before reset
        env.unwrapped.set_task(task_config)

        # Move to next task (wrap around if we pass the end)
        task_idx = (task_idx + 1) % num_tasks
        # ---------------------------------------------

        # Reset environment
        obs_raw, info = env.reset()  # Raw shape e.g. (40,30,3)
        policy.reset_lstm_states(batch_size=1)

        done = False
        truncated = False
        print("Initial obs shape from env:", obs_raw.shape)

        # Episode-level storage
        step_list_obs = []
        step_list_actions = []
        step_list_logp = []
        step_list_values = []
        step_list_rewards = []
        step_list_dones = []

        while not done and not truncated:
            # 3) DO NOT overwrite obs_raw. Instead, create a transposed copy for the model
            # raw shape: (40,30,3) => model shape: (3,40,30)
            obs_for_model = np.transpose(obs_raw, (2, 0, 1))
            # print("obs_for_model shape:", obs_for_model.shape)

            obs_torch = torch.from_numpy(obs_for_model).float().unsqueeze(0).to(device)
            with torch.no_grad():
                logits, value = policy.act_single_step(obs_torch[0])

            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)

            # Step environment with the chosen action
            obs_next_raw, reward, done, truncated, info = env.step(action.item())
            total_steps += 1

            # Collect transitions
            step_list_obs.append(obs_for_model)  # store the transposed version for training
            step_list_actions.append(action.item())
            step_list_logp.append(logp.item())
            step_list_values.append(value.item())
            step_list_rewards.append(reward)
            step_list_dones.append(float(done))

            # Now get the next raw observation
            obs_raw = obs_next_raw  # shape is e.g. (40,30,3)

        episode_count += 1

        # If not done, we can estimate next_value for GAE
        next_value = 0.0
        if not done:
            obs_for_model = np.transpose(obs_raw, (2, 0, 1))
            obs_torch = torch.from_numpy(obs_for_model).float().unsqueeze(0).to(device)
            with torch.no_grad():
                _, val_ = policy.act_single_step(obs_torch[0])
            next_value = val_.item()

        ep_len = len(step_list_rewards)
        rewards_np = np.array(step_list_rewards, dtype=np.float32)
        dones_np = np.array(step_list_dones, dtype=np.float32)
        values_np = np.array(step_list_values, dtype=np.float32)

        # Compute GAE
        advantages_np = ppo_trainer.compute_gae(
            rewards=rewards_np, dones=dones_np, values=values_np, next_value=next_value
        )
        returns_np = values_np + advantages_np

        all_obs.append(np.array(step_list_obs, dtype=np.float32))            # shape (T,3,40,30)
        all_actions.append(np.array(step_list_actions, dtype=np.int64))      # shape (T,)
        all_log_probs.append(np.array(step_list_logp, dtype=np.float32))     # shape (T,)
        all_values.append(values_np)                                         # shape (T,)
        all_rewards.append(rewards_np)                                       # shape (T,)
        all_dones.append(dones_np)                                           # shape (T,)

        # If we've accumulated enough timesteps, do a PPO update
        if total_steps >= timesteps_per_update:
            max_len = max(o.shape[0] for o in all_obs)
            N = len(all_obs)

            # Build a rollout structure
            sequences = []
            for i in range(N):
                seq_dict = {
                    "obs": all_obs[i],          # shape (T_i, 3, 40, 30)
                    "actions": all_actions[i],  
                    "old_log_probs": all_log_probs[i],
                    "values": all_values[i],
                    "returns": None,
                    "advantages": None
                }
                T_i = seq_dict["actions"].shape[0]

                # If truncated, next_value = 0 or re-estimate. For simplicity, we skip it here.
                next_val = 0.0
                if all_dones[i][-1] < 1e-6:
                    # partial
                    pass

                adv_i = ppo_trainer.compute_gae(all_rewards[i], all_dones[i], all_values[i], next_val)
                ret_i = all_values[i] + adv_i
                seq_dict["returns"] = ret_i
                seq_dict["advantages"] = adv_i
                sequences.append(seq_dict)

            # Now we unify these sequences into a single batch by padding
            obs_list = []
            actions_list = []
            old_logp_list = []
            returns_list = []
            values_list = []
            adv_list = []
            max_len = max(seq["obs"].shape[0] for seq in sequences)

            for seq in sequences:
                T_i = seq["obs"].shape[0]
                pad_len = max_len - T_i

                # pad obs from shape (T_i,3,40,30) to (max_len,3,40,30)
                obs_pad = np.pad(seq["obs"], ((0,pad_len),(0,0),(0,0),(0,0)), mode="constant")
                actions_pad = np.pad(seq["actions"], (0,pad_len), mode="constant")
                old_logp_pad = np.pad(seq["old_log_probs"], (0,pad_len), mode="constant")
                returns_pad = np.pad(seq["returns"], (0,pad_len), mode="constant")
                values_pad = np.pad(seq["values"], (0,pad_len), mode="constant")
                adv_pad = np.pad(seq["advantages"], (0,pad_len), mode="constant")

                obs_list.append(obs_pad)
                actions_list.append(actions_pad)
                old_logp_list.append(old_logp_pad)
                returns_list.append(returns_pad)
                values_list.append(values_pad)
                adv_list.append(adv_pad)

            obs_arr = np.stack(obs_list, axis=0)
            actions_arr = np.stack(actions_list, axis=0)
            old_logp_arr = np.stack(old_logp_list, axis=0)
            returns_arr = np.stack(returns_list, axis=0)
            values_arr = np.stack(values_list, axis=0)
            adv_arr = np.stack(adv_list, axis=0)

            final_rollouts = {
                "obs": obs_arr,
                "actions": actions_arr,
                "old_log_probs": old_logp_arr,
                "returns": returns_arr,
                "values": values_arr,
                "advantages": adv_arr
            }

            # PPO Update
            stats = ppo_trainer.update(final_rollouts)
            print(f"[Update] Steps={total_steps}, Episode={episode_count}, Stats={stats}")

            # Clear buffers
            all_obs.clear()
            all_actions.clear()
            all_log_probs.clear()
            all_values.clear()
            all_rewards.clear()
            all_dones.clear()

            # Increase threshold for next update
            timesteps_per_update += 20000

    env.close()
    print(f"Finished training after {total_steps} steps and {episode_count} episodes.")


if __name__ == "__main__":
    main()
