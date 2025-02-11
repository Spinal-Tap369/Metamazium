# test_lstm.py

import gymnasium as gym
import numpy as np
import torch
import json
import random
import pygame  # needed for saving trajectories if desired

from metamazium.env.maze_task import MazeTaskManager
from metamazium.lstm_trpo.lstm_model import StackedLSTMPolicyValueNet
import metamazium.env  

def save_trajectory_from_list(traj, file_name, maze_core):
    """
    Save a trajectory image from a list of grid positions.
    This function uses the render parameters of maze_core.
    """
    maze_core.render_init(480)  # use 480 as view size (adjust as needed)
    view_size = maze_core._view_size
    render_cell_size = maze_core._render_cell_size
    traj_screen = pygame.Surface((view_size, view_size))
    traj_screen.fill(pygame.Color("white"))
    for i in range(len(traj) - 1):
        p = traj[i]
        n = traj[i + 1]
        p_pos = ((p[0] + 0.5) * render_cell_size, view_size - (p[1] + 0.5) * render_cell_size)
        n_pos = ((n[0] + 0.5) * render_cell_size, view_size - (n[1] + 0.5) * render_cell_size)
        pygame.draw.line(traj_screen, pygame.Color("red"), p_pos, n_pos, 3)
    pygame.image.save(traj_screen, file_name)
    print(f"Saved trajectory to {file_name}")

def main():
    # 1) Load 100 unique mazes.
    test_tasks_file = "/content/test_tasks_small.json"
    with open(test_tasks_file, "r") as f:
        unique_tasks = json.load(f)
    num_unique = len(unique_tasks)
    print(f"Loaded {num_unique} unique test mazes from {test_tasks_file}")

    # 2) For each unique maze, sample 5 runs.
    trials = []
    for task in unique_tasks:
        for _ in range(5):
            trials.append(task)
    print(f"Total runs: {len(trials)} (each unique maze repeated 5 times)")

    # 3) Load environment.
    env_id = "MetaMazeDiscrete3D-v0"
    env = gym.make(env_id, enable_render=False)

    # 4) Load your trained LSTM model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model = StackedLSTMPolicyValueNet(
        action_dim=4,
        hidden_size=512,
        num_layers=2
    ).to(device)

    checkpoint_path = "/content/trpo_chkpt.pth"
    if not torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path)
    lstm_model.load_state_dict(checkpoint["policy_state_dict"])
    lstm_model.eval()
    print(f"Loaded model from {checkpoint_path}")

    # List to store details of selected runs.
    selected_runs = []  # Each element: (trial_index, run_index, phase1_steps, phase2_steps, phase_change_idx)

    trial_idx = 0
    for idx, task_params in enumerate(trials):
        trial_idx += 1
        print(f"\nTrial {trial_idx}/{len(trials)}")
        # Use TaskConfig to build the task.
        task_config = MazeTaskManager.TaskConfig(**task_params)
        env.unwrapped.set_task(task_config)

        # Run one trial (run) independently.
        # Reset LSTM hidden states before each run.
        lstm_model.reset_memory(batch_size=1, device=device)
        obs_raw, _ = env.reset()
        env.unwrapped.maze_core.randomize_start()
        try:
            env.unwrapped.maze_core.randomize_goal(min_distance=3.0)
        except Exception as e:
            print(f"  Warning: randomize_goal failed: {e}. Using current goal.")

        done = False
        truncated = False

        ep_obs_seq = []
        last_action = 0.0
        last_reward = 0.0
        boundary_bit = 1.0
        phase_boundary_signaled = False
        phase_change_idx = None

        phase1_steps = 0
        phase2_steps = 0

        while not done and not truncated:
            obs_img = np.transpose(obs_raw, (2, 0, 1))  # (3, H, W)
            H, W = obs_img.shape[1], obs_img.shape[2]
            c3 = np.full((1, H, W), last_action, dtype=np.float32)
            c4 = np.full((1, H, W), last_reward, dtype=np.float32)
            c5 = np.full((1, H, W), boundary_bit, dtype=np.float32)
            obs_6ch = np.concatenate([obs_img, c3, c4, c5], axis=0)
            ep_obs_seq.append(obs_6ch)
            t_len = len(ep_obs_seq)
            obs_seq_np = np.stack(ep_obs_seq, axis=0)[None]  # Shape: (1, t_len, 6, H, W)
            obs_seq_torch = torch.from_numpy(obs_seq_np).float().to(device)

            with torch.no_grad():
                logits_t, _ = lstm_model.act_single_step(obs_seq_torch)  # (1,1,action_dim)
            dist = torch.distributions.Categorical(logits=logits_t.squeeze(1))
            action = dist.sample()
            obs_next, reward, done, truncated, info = env.step(action.item())

            obs_raw = obs_next
            last_action = float(action.item())
            last_reward = float(reward)

            # Update phase metrics.
            if "phase" in info:
                current_phase = info["phase"]
                if current_phase == 1:
                    phase1_steps += 1
                elif current_phase == 2:
                    phase2_steps += 1

            # Record phase change index if not yet set and phase becomes 2.
            current_phase = env.unwrapped.maze_core.phase
            if phase_change_idx is None and current_phase == 2:
                phase_change_idx = len(ep_obs_seq)
            if current_phase == 2 and not phase_boundary_signaled:
                boundary_bit = 1.0
                phase_boundary_signaled = True
            else:
                boundary_bit = 0.0

        print(f"  Run {trial_idx}: Phase1 steps = {phase1_steps}, Phase2 steps = {phase2_steps}, Phase change index = {phase_change_idx}")
        if phase2_steps < phase1_steps:
            selected_runs.append((trial_idx, phase1_steps, phase2_steps, phase_change_idx))
            print("  Run selected.")
        else:
            print("  Run not selected.")

    if selected_runs:
        all_phase1 = [r[1] for r in selected_runs]
        all_phase2 = [r[2] for r in selected_runs]
        print("\n==== Selected Runs (Phase2 steps < Phase1 steps) ====")
        for r in selected_runs:
            print(f"Trial {r[0]}: Phase1 steps = {r[1]}, Phase2 steps = {r[2]}, Phase change index = {r[3]}")
        print("==== Final Averages for Selected Runs ====")
        print(f"Average Phase1 steps: {np.mean(all_phase1):.2f}")
        print(f"Average Phase2 steps: {np.mean(all_phase2):.2f}")
    else:
        print("No runs met the condition: Phase2 steps < Phase1 steps.")

    env.close()

if __name__ == "__main__":
    main()
