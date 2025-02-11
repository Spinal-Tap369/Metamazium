import gymnasium as gym
import numpy as np
import torch
import json
import random
import pygame  # needed for saving trajectories, if desired

from metamazium.env.maze_task import MazeTaskManager
from metamazium.snail_performer.snail_model import SNAILPolicyValueNet
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
        action_dim=3,  
        base_dim=256,
        policy_filters=32,
        policy_attn_dim=16,
        value_filters=16,
        seq_len=500,
        num_policy_attn=2
    ).to(device)
    model_path = "/content/snail_final_model.pth"
    ckpt = torch.load(model_path, map_location=device)
    snail_model.load_state_dict(ckpt)
    snail_model.eval()
    print(f"Loaded model from {model_path}")

    # List to store details of selected runs (each run individually).
    selected_runs = []  # Each element: (trial_index, run_index, phase1_steps, phase2_steps, phase_change_idx)

    trial_idx = 0
    # For each trial (each maze repeated 5 times)
    for idx, task_params in enumerate(trials):
        trial_idx += 1
        print(f"\nTrial {trial_idx}/{len(trials)}")
        task_config = MazeTaskManager.TaskConfig(**task_params)
        env.unwrapped.set_task(task_config)

        # For this trial, run 5 independent runs.
        for run in range(5):
            print(f"  Run {run+1}:")
            obs_raw, _ = env.reset()
            env.unwrapped.maze_core.randomize_start()
            try:
                env.unwrapped.maze_core.randomize_goal(min_distance=3.0)
            except Exception as e:
                print(f"    Warning: randomize_goal failed: {e}. Using current goal.")

            done = False
            truncated = False

            # For SNAIL input (if needed)
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
                obs_seq_np = np.stack(ep_obs_seq, axis=0)[None]
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

                if info["phase"] == 1:
                    phase1_steps += 1
                else:
                    phase2_steps += 1

                # Record the phase change index (first time phase becomes 2)
                current_phase = env.unwrapped.maze_core.phase
                if phase_change_idx is None and current_phase == 2:
                    phase_change_idx = len(ep_obs_seq)
                if current_phase == 2 and not phase_boundary_signaled:
                    boundary_bit = 1.0
                    phase_boundary_signaled = True
                else:
                    boundary_bit = 0.0

            print(f"    Phase1 steps = {phase1_steps}, Phase2 steps = {phase2_steps}, Phase change index = {phase_change_idx}")
            if phase2_steps < phase1_steps:
                selected_runs.append((trial_idx, run+1, phase1_steps, phase2_steps, phase_change_idx))
                print("    Run selected.")
            else:
                print("    Run not selected.")

        # End trial loop (if needed, you can add a break or continue)

    # Report overall selected runs.
    if selected_runs:
        all_phase1 = [r[2] for r in selected_runs]
        all_phase2 = [r[3] for r in selected_runs]
        print("\n==== Selected Runs (Phase2 steps < Phase1 steps) ====")
        for r in selected_runs:
            print(f"Trial {r[0]} Run {r[1]}: Phase1 steps = {r[2]}, Phase2 steps = {r[3]}, Phase change index = {r[4]}")
        print("==== Final Averages for Selected Runs ====")
        print(f"Average Phase1 steps: {np.mean(all_phase1):.2f}")
        print(f"Average Phase2 steps: {np.mean(all_phase2):.2f}")
    else:
        print("No runs met the condition: Phase2 steps < Phase1 steps.")

    env.close()

if __name__ == "__main__":
    main()
