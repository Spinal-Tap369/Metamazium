import gymnasium as gym
import numpy as np
import torch
import json
import sys

import env  # custom env is registered in this module
from lstm_ppo.lstm_model import LSTMPolicy
from lstm_ppo import PPOTrainer
from env.maze_task import MazeTaskSampler  # For generating MazeTask configurations

def main():
    env_id = "MetaMazeDiscrete3D-v0"
    env = gym.make(env_id, enable_render=False)

    # Load tasks from JSON file
    tasks_file = "mazes_data/train_tasks.json"
    with open(tasks_file, "r") as f:
        tasks = json.load(f)
    num_tasks = len(tasks)
    print(f"Loaded {num_tasks} tasks from {tasks_file}")

    # Hyperparameters
    total_timesteps = 500000
    timesteps_per_update = 100000
    gamma = 0.99
    gae_lambda = 0.99
    clip_range = 0.2
    target_kl = 0.03
    episodes_per_task = 5  # Number of episodes to run on a single maze

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize policy and PPO trainer
    policy = LSTMPolicy(action_dim=4, hidden_size=512).to(device)
    ppo_trainer = PPOTrainer(
        policy_model=policy,
        lr=1e-5,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        n_epochs=5,
        batch_size=64,
        target_kl=target_kl,
        max_grad_norm=0.3,
        entropy_coef=0.01,
        value_coef=0.5
    )

    total_steps = 0
    episode_count = 0

    # Buffers for PPO updates
    all_obs = []
    all_actions = []
    all_log_probs = []
    all_values = []
    all_rewards = []
    all_dones = []

    task_idx = 0

    while total_steps < total_timesteps:
        # Select a new maze (task)
        this_task_params = tasks[task_idx]
        task_config = MazeTaskSampler(**this_task_params)
        env.unwrapped.set_task(task_config)
        print(f"Starting new task {task_idx+1}/{num_tasks}")

        # Reset acting LSTM state once per new maze
        policy.reset_lstm_states(batch_size=1, mode='act')

        # Run multiple episodes on the same maze
        for ep in range(episodes_per_task):
            obs_raw, info = env.reset()  # Reset environment for new episode on same maze
            # Do NOT reset LSTM state here to preserve memory across episodes

            done = False
            truncated = False
            print(f"Episode {episode_count+1} on task {task_idx+1}")

            step_list_obs = []
            step_list_actions = []
            step_list_logp = []
            step_list_values = []
            step_list_rewards = []
            step_list_dones = []

            while not done and not truncated:
                obs_for_model = np.transpose(obs_raw, (2, 0, 1))  # (3,30,40)
                obs_torch = torch.from_numpy(obs_for_model).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    logits, value = policy.act_single_step(obs_torch[0])

                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

                obs_next_raw, reward, done, truncated, info = env.step(action.item())
                total_steps += 1

                step_list_obs.append(obs_for_model)
                step_list_actions.append(action.item())
                step_list_logp.append(logp.item())
                step_list_values.append(value.item())
                step_list_rewards.append(reward)
                step_list_dones.append(float(done))

                obs_raw = obs_next_raw

            episode_count += 1

            # Estimate next value at episode end
            next_value = 0.0
            if not done:
                obs_for_model = np.transpose(obs_raw, (2, 0, 1))
                obs_torch = torch.from_numpy(obs_for_model).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    _, val_ = policy.act_single_step(obs_torch[0])
                next_value = val_.item()

            rewards_np = np.array(step_list_rewards, dtype=np.float32)
            dones_np = np.array(step_list_dones, dtype=np.float32)
            values_np = np.array(step_list_values, dtype=np.float32)
            advantages_np = ppo_trainer.compute_gae(
                rewards=rewards_np, dones=dones_np, values=values_np, next_value=next_value
            )
            returns_np = values_np + advantages_np

            all_obs.append(np.array(step_list_obs, dtype=np.float32))
            all_actions.append(np.array(step_list_actions, dtype=np.int64))
            all_log_probs.append(np.array(step_list_logp, dtype=np.float32))
            all_values.append(values_np)
            all_rewards.append(rewards_np)
            all_dones.append(dones_np)

            # Check if it's time for a PPO update
            if total_steps >= timesteps_per_update:
                sequences = []
                for i in range(len(all_obs)):
                    seq_dict = {
                        "obs": all_obs[i],
                        "actions": all_actions[i],
                        "old_log_probs": all_log_probs[i],
                        "values": all_values[i],
                        "returns": None,
                        "advantages": None
                    }
                    next_val = 0.0
                    adv_i = ppo_trainer.compute_gae(all_rewards[i], all_dones[i], all_values[i], next_val)
                    ret_i = all_values[i] + adv_i
                    seq_dict["returns"] = ret_i
                    seq_dict["advantages"] = adv_i
                    sequences.append(seq_dict)

                max_len = max(seq["obs"].shape[0] for seq in sequences)
                obs_list, actions_list, old_logp_list = [], [], []
                returns_list, values_list, adv_list = [], [], []

                for seq in sequences:
                    T_i = seq["obs"].shape[0]
                    pad_len = max_len - T_i

                    obs_pad = np.pad(seq["obs"], ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode="constant")
                    actions_pad = np.pad(seq["actions"], (0, pad_len), mode="constant")
                    old_logp_pad = np.pad(seq["old_log_probs"], (0, pad_len), mode="constant")
                    returns_pad = np.pad(seq["returns"], (0, pad_len), mode="constant")
                    values_pad = np.pad(seq["values"], (0, pad_len), mode="constant")
                    adv_pad = np.pad(seq["advantages"], (0, pad_len), mode="constant")

                    obs_list.append(obs_pad)
                    actions_list.append(actions_pad)
                    old_logp_list.append(old_logp_pad)
                    returns_list.append(returns_pad)
                    values_list.append(values_pad)
                    adv_list.append(adv_pad)

                final_rollouts = {
                    "obs": np.stack(obs_list, axis=0),
                    "actions": np.stack(actions_list, axis=0),
                    "old_log_probs": np.stack(old_logp_list, axis=0),
                    "returns": np.stack(returns_list, axis=0),
                    "values": np.stack(values_list, axis=0),
                    "advantages": np.stack(adv_list, axis=0)
                }

                stats = ppo_trainer.update(final_rollouts)
                print(f"[Update] Steps={total_steps}, Episode={episode_count}, Stats={stats}")

                all_obs.clear()
                all_actions.clear()
                all_log_probs.clear()
                all_values.clear()
                all_rewards.clear()
                all_dones.clear()

                policy.reset_lstm_states(batch_size=64, mode='update')
                policy.reset_lstm_states(batch_size=1, mode='act')
                timesteps_per_update += 20000

        # After completing episodes_per_task episodes on the current maze,
        # move to the next maze.
        task_idx = (task_idx + 1) % num_tasks

        if total_steps >= total_timesteps:
            break

    env.close()
    torch.save(policy.state_dict(), "models/lstm_policy.pt")
    print("Model saved")
    print(f"Finished training after {total_steps} steps and {episode_count} episodes.")

if __name__ == "__main__":
    main()
