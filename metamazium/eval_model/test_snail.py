# test_snail.py

import gymnasium as gym
import numpy as np
import torch
import json
import random
import pygame  # needed for saving trajectories

from metamazium.env.maze_task import MazeTaskSampler
from metamazium.snail_performer.snail_model import SNAILPolicyValueNet
import metamazium.env  

def save_trajectory_from_list(traj, file_name, maze_core):
    """
    Save a trajectory image from a list of grid positions.
    This function uses the render parameters of maze_core.
    """
    # Ensure render parameters are set.
    maze_core.render_init(480)  # use 480 as view size (adjust as needed)
    view_size = maze_core._view_size
    render_cell_size = maze_core._render_cell_size
    traj_screen = pygame.Surface((view_size, view_size))
    traj_screen.fill(pygame.Color("white"))
    # Draw trajectory lines.
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
    test_tasks_file = "metamazium/mazes_data/test_tasks_small.json"
    with open(test_tasks_file, "r") as f:
        unique_tasks = json.load(f)
    num_unique = len(unique_tasks)
    print(f"Loaded {num_unique} unique test mazes from {test_tasks_file}")

    # 2) For each unique maze, sample 5 trials.
    trials = []
    for task in unique_tasks:
        for _ in range(5):
            trials.append(task)
    print(f"Total trials: {len(trials)} (each unique maze repeated 5 times)")

    # 3) Load environment.
    env_id = "MetaMazeDiscrete3D-v0"
    env = gym.make(env_id, enable_render=False)

    # 4) Load the trained SNAIL model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snail_model = SNAILPolicyValueNet(
        action_dim=4,  # adjust if needed
        base_dim=256,
        policy_filters=32,
        policy_attn_dim=16,
        value_filters=16,
        seq_len=800,  # must match or exceed max steps used in training
        num_policy_attn=2
    ).to(device)
    model_path = "models/snail_chkload95.pth"
    ckpt = torch.load(model_path, map_location=device)
    snail_model.load_state_dict(ckpt["policy_state_dict"])
    snail_model.eval()
    print(f"Loaded model from {model_path}")

    # Lists to store selected trial details.
    selected_trials = []  # Each element: (trial_index, avg_phase1, avg_phase2)

    # For each trial (each maze repeated 5 times)
    trial_idx = 0
    for idx, task_params in enumerate(trials):
        trial_idx += 1
        print(f"\nTrial {trial_idx}/{len(trials)}")
        # 5) Set up the maze for the given trial.
        task_config = MazeTaskSampler(**task_params)
        env.unwrapped.set_task(task_config)

        # For this trial, run 5 independent runs.
        run_phase1_steps = []
        run_phase2_steps = []
        phase_change_indices = []  # to store the index when phase transition occurs per run

        for run in range(5):
            obs_raw, _ = env.reset()
            # Randomize start and goal.
            env.unwrapped.maze_core.randomize_start()
            try:
                env.unwrapped.maze_core.randomize_goal(min_distance=3.0)
            except Exception as e:
                print(f"Warning (run {run+1}): randomize_goal failed: {e}. Using current goal.")

            done = False
            truncated = False

            ep_obs_seq = []  # to build the input sequence (if needed for the model)
            last_action = 0.0
            last_reward = 0.0
            boundary_bit = 1.0
            phase_boundary_signaled = False
            phase_change_idx = None  # record when phase changes

            phase1_steps = 0
            phase2_steps = 0

            while not done and not truncated:
                obs_img = np.transpose(obs_raw, (2, 0, 1))  # (3, H, W)
                H, W = obs_img.shape[1], obs_img.shape[2]
                c3 = np.full((1, H, W), last_action, dtype=np.float32)
                c4 = np.full((1, H, W), last_reward, dtype=np.float32)
                c5 = np.full((1, H, W), boundary_bit, dtype=np.float32)
                obs_6ch = np.concatenate([obs_img, c3, c4, c5], axis=0)  # (6, H, W)

                ep_obs_seq.append(obs_6ch)
                t_len = len(ep_obs_seq)
                obs_seq_np = np.stack(ep_obs_seq, axis=0)[None]  # shape: (1, t_len, 6, H, W)
                obs_seq_torch = torch.from_numpy(obs_seq_np).float().to(device)

                with torch.no_grad():
                    logits_seq, _ = snail_model(obs_seq_torch)
                logits_t = logits_seq[:, t_len - 1, :]
                dist = torch.distributions.Categorical(logits=logits_t)
                action = dist.sample()
                obs_next, reward, done, truncated, info = env.step(action.item())

                obs_raw = obs_next
                last_action = float(action.item())
                last_reward = float(reward)

                # Update phase counts.
                if info["phase"] == 1:
                    phase1_steps += 1
                else:
                    phase2_steps += 1

                # Record phase change index (only first occurrence)
                current_phase = env.unwrapped.maze_core.phase
                if phase_change_idx is None and current_phase == 2:
                    phase_change_idx = len(ep_obs_seq)
                # Set boundary flag.
                if current_phase == 2 and not phase_boundary_signaled:
                    boundary_bit = 1.0
                    phase_boundary_signaled = True
                else:
                    boundary_bit = 0.0

            run_phase1_steps.append(phase1_steps)
            run_phase2_steps.append(phase2_steps)
            phase_change_indices.append(phase_change_idx)
            print(f"  Run {run+1}: Phase1 steps = {phase1_steps}, Phase2 steps = {phase2_steps}, Phase change index = {phase_change_idx}")

        # Compute average steps over the 5 runs.
        avg_phase1 = np.mean(run_phase1_steps)
        avg_phase2 = np.mean(run_phase2_steps)
        print(f"Trial {trial_idx}: Avg Phase1 steps = {avg_phase1:.2f}, Avg Phase2 steps = {avg_phase2:.2f}")

        # Select trial if average Phase2 steps are lower than average Phase1 steps.
        if avg_phase2 < avg_phase1:
            selected_trials.append((trial_idx, avg_phase1, avg_phase2, phase_change_indices))
    
    # After testing all trials, compute overall averages among selected trials.
    if selected_trials:
        sel_phase1_avg = np.mean([trial[1] for trial in selected_trials])
        sel_phase2_avg = np.mean([trial[2] for trial in selected_trials])
        print("\n==== Selected Trials (Avg Phase2 steps < Avg Phase1 steps) ====")
        for trial in selected_trials:
            print(f"Trial {trial[0]}: Avg Phase1 = {trial[1]:.2f}, Avg Phase2 = {trial[2]:.2f}")
        print("==== Final Averages for Selected Trials ====")
        print(f"Average Phase1 steps: {sel_phase1_avg:.2f}")
        print(f"Average Phase2 steps: {sel_phase2_avg:.2f}")

        # For each selected trial, save trajectories for both phases.
        # Here, we assume that the maze_core stores the full trajectory in _agent_trajectory.
        # We'll split this trajectory using the phase change index from the first run of that trial.
        # (For simplicity, using the phase change index from the first run.)
        for trial in selected_trials:
            trial_index, _, _, phase_change_indices = trial
            # Use the first non-None phase change index from the 5 runs.
            phase_change_idx = next((idx for idx in phase_change_indices if idx is not None), None)
            if phase_change_idx is None:
                print(f"Trial {trial_index}: No phase change detected; skipping trajectory saving.")
                continue
            full_traj = env.unwrapped.maze_core._agent_trajectory  # list of grid positions
            # Split the trajectory.
            phase1_traj = full_traj[:phase_change_idx]
            phase2_traj = full_traj[phase_change_idx:]
            # Save the trajectories.
            save_trajectory_from_list(phase1_traj, f"trial_{trial_index}_phase1.png", env.unwrapped.maze_core)
            save_trajectory_from_list(phase2_traj, f"trial_{trial_index}_phase2.png", env.unwrapped.maze_core)
    else:
        print("No trials met the condition: Avg Phase2 steps < Avg Phase1 steps.")

    env.close()

if __name__ == "__main__":
    main()
