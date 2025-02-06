# test_lstm.py

import gymnasium as gym
import numpy as np
import torch
import json
import os
from metamazium.env.maze_task import MazeTaskSampler
from metamazium.lstm_trpo.lstm_model import StackedLSTMPolicy
import metamazium.env  


def main():
    # 1) Load test tasks
    test_tasks_file = "metamazium./mazes_data/test_small_tasks.json"
    with open(test_tasks_file, "r") as f:
        test_tasks = json.load(f)
    num_test_tasks = len(test_tasks)
    print(f"Loaded {num_test_tasks} test tasks from {test_tasks_file}")

    # 2) Load environment
    env_id = "MetaMazeDiscrete3D-v0"
    env = gym.make(env_id, enable_render=False)

    # 3) Load your trained LSTM model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model = StackedLSTMPolicy(
        action_dim=4,
        hidden_size=512,
        num_layers=2
    ).to(device)

    # Load state dict from checkpoint
    checkpoint_path = "checkpoint_lstm/custom_lstm_ckpt.pth"
    if not os.path.isfile(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    lstm_model.load_state_dict(checkpoint["policy_state_dict"])
    lstm_model.eval()
    print(f"Loaded model from {checkpoint_path}")

    # Keep track of total steps in phase1 & phase2
    phase1_steps_list = []
    phase2_steps_list = []

    for idx, task_params in enumerate(test_tasks):
        # 4) Set up the Maze with the given test task
        task_config = MazeTaskSampler(**task_params)
        env.unwrapped.set_task(task_config)

        # 5) Reset environment
        obs_raw, _ = env.reset()
        done = False
        truncated = False

        # Reset LSTM hidden states
        lstm_model.reset_memory(batch_size=1, device=device)

        # Initialize variables for additional channels.
        last_action = 0.0
        last_reward = 0.0
        boundary_bit = 1.0
        phase_boundary_signaled = False

        ep_obs_seq = []

        # We'll measure steps in each phase
        phase1_steps = 0
        phase2_steps = 0

        while not done and not truncated:
            # Build 6-channel observation
            obs_img = np.transpose(obs_raw, (2, 0, 1))  # => (3,H,W)
            H, W = obs_img.shape[1], obs_img.shape[2]
            c3 = np.full((1, H, W), last_action, dtype=np.float32)
            c4 = np.full((1, H, W), last_reward, dtype=np.float32)
            c5 = np.full((1, H, W), boundary_bit, dtype=np.float32)
            obs_6ch = np.concatenate([obs_img, c3, c4, c5], axis=0)  # => (6,H,W)

            # Append to partial sequence
            ep_obs_seq.append(obs_6ch)

            # Convert to shape (1,1,6,H,W) for a single step forward
            obs_t = torch.from_numpy(obs_6ch[None, None]).float().to(device)  # Shape: (1,1,6,H,W)
            with torch.no_grad():
                logits_t, val_t = lstm_model.act_single_step(obs_t)  # Shapes: (1,1,action_dim), (1,1)
            dist = torch.distributions.Categorical(logits=logits_t.squeeze(1))  # Remove timestep dim
            action = dist.sample()
            # step environment
            obs_next, reward, done, truncated, info = env.step(action.item())

            # Update
            obs_raw = obs_next
            last_action = float(action.item())
            last_reward = float(reward)

            # Check which phase
            if "phase" in info:
                current_phase = info["phase"]
                if current_phase == 1:
                    phase1_steps += 1
                elif current_phase == 2:
                    phase2_steps += 1

            # If we just switched to phase 2, set boundary bit once
            if hasattr(env.unwrapped, 'maze_core') and env.unwrapped.maze_core.phase == 2 and not phase_boundary_signaled:
                boundary_bit = 1.0
                phase_boundary_signaled = True
            else:
                boundary_bit = 0.0

        # End of this test run
        phase1_steps_list.append(phase1_steps)
        phase2_steps_list.append(phase2_steps)
        print(f"Test Task {idx+1}/{num_test_tasks}: Phase1 steps={phase1_steps}, Phase2 steps={phase2_steps}")

    # Done testing all tasks
    env.close()

    # 6) Compute average steps in phase 1 & 2
    avg_phase1 = np.mean(phase1_steps_list)
    avg_phase2 = np.mean(phase2_steps_list)
    print("==== Test Results ====")
    print(f"Average Phase1 steps: {avg_phase1:.2f}")
    print(f"Average Phase2 steps: {avg_phase2:.2f}")


if __name__ == "__main__":
    main()
