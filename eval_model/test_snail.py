# test_snail.py

import gymnasium as gym
import numpy as np
import torch
import json
from metamazium.env.maze_task import MazeTaskSampler
from metamazium.snail_performer.snail_model import SNAILPolicyValueNet
from metamazium.snail_performer.ppo import PPOTrainer  
import metamazium.env  

def main():
    # 1) Load test tasks
    test_tasks_file = "mazes_data/test_small_tasks.json"
    with open(test_tasks_file, "r") as f:
        test_tasks = json.load(f)
    num_test_tasks = len(test_tasks)
    print(f"Loaded {num_test_tasks} test tasks from {test_tasks_file}")

    # 2) Load environment
    env_id = "MetaMazeDiscrete3D-v0"
    env = gym.make(env_id, enable_render=False)

    # 3) Load your trained SNAIL model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snail_model = SNAILPolicyValueNet(
        action_dim=4,
        base_dim=256,
        policy_filters=32,
        policy_attn_dim=16,
        value_filters=16,
        seq_len=800,  # must match or exceed max steps used in training
        num_policy_attn=2
    ).to(device)

    # load state dict
    model_path = "models/snail_snaillike_policy_value_online.pt"
    snail_model.load_state_dict(torch.load(model_path, map_location=device))
    snail_model.eval()
    print(f"Loaded model from {model_path}")

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

        # Also track partial-sequence data
        ep_obs_seq = []

        # Additional channels
        last_action = 0.0
        last_reward = 0.0
        boundary_bit = 1.0
        phase_boundary_signaled = False

        # We'll measure steps in each phase
        phase1_steps = 0
        phase2_steps = 0

        while not done and not truncated:
            # Build 6-channel
            obs_img = np.transpose(obs_raw, (2,0,1))  # => (3,H,W)
            H, W = obs_img.shape[1], obs_img.shape[2]
            c3 = np.full((1,H,W), last_action, dtype=np.float32)
            c4 = np.full((1,H,W), last_reward, dtype=np.float32)
            c5 = np.full((1,H,W), boundary_bit, dtype=np.float32)
            obs_6ch = np.concatenate([obs_img, c3, c4, c5], axis=0)  # => (6,H,W)

            # Append to partial sequence
            ep_obs_seq.append(obs_6ch)
            t_len = len(ep_obs_seq)

            # Shape => (1,t_len,6,H,W)
            obs_seq_np = np.stack(ep_obs_seq, axis=0)[None]
            obs_seq_torch = torch.from_numpy(obs_seq_np).float().to(device)

            # Forward pass => last index => current step's logits
            with torch.no_grad():
                logits_seq, vals_seq = snail_model(obs_seq_torch)
            logits_t = logits_seq[:, t_len - 1, :]   # => (1, action_dim)
            # We don't necessarily need the value here, but we could:
            # val_t = vals_seq[:, t_len - 1]

            dist = torch.distributions.Categorical(logits=logits_t)
            action = dist.sample()
            # step environment
            obs_next, reward, done, truncated, info = env.step(action.item())

            # Update
            obs_raw = obs_next
            last_action = float(action.item())
            last_reward = float(reward)

            # Check which phase
            if info["phase"] == 1:
                phase1_steps += 1
            else:
                phase2_steps += 1

            # If we just switched to phase 2, set boundary bit once
            if env.unwrapped.maze_core.phase == 2 and not phase_boundary_signaled:
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
