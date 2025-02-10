# metamazium/train_model/train_snail_trpo_fo.py

import os
import json
import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
import argparse
import random

torch.autograd.set_detect_anomaly(False)

import metamazium.env
from metamazium.snail_performer.snail_model import SNAILPolicyValueNet
from metamazium.snail_performer.trpo_fo import TRPO_FO
from metamazium.env.maze_task import MazeTaskManager  # for task reconstruction

# Default hyperparameters
DEFAULT_TOTAL_TIMESTEPS = 500000
DEFAULT_STEPS_PER_UPDATE = 50000  
DEFAULT_DISCOUNT = 0.99
DEFAULT_GAE_LAMBDA = 0.99
DEFAULT_MAX_KL_DIV = 0.01
DEFAULT_VF_LR = 0.01
DEFAULT_VF_ITERS = 5
DEFAULT_CHECKPOINT_DIR = "checkpoint_snail_trpo_fo"
DEFAULT_CHECKPOINT_LOAD_DIR = "checkpoint_snail_trpo_fo_load"
DEFAULT_CHECKPOINT_FILE = None

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SNAIL-TRPO (First-Order) on MetaMaze environment."
    )
    parser.add_argument("--total_timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS,
                        help="Total training timesteps (default: 500000)")
    parser.add_argument("--steps_per_update", type=int, default=DEFAULT_STEPS_PER_UPDATE,
                        help="Timesteps collected per TRPO update (default: 50000)")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help="Directory to save checkpoints (default: checkpoint_snail_trpo_fo)")
    parser.add_argument("--checkpoint_load_dir", type=str, default=DEFAULT_CHECKPOINT_LOAD_DIR,
                        help="Directory to load checkpoints from (default: checkpoint_snail_trpo_fo_load)")
    parser.add_argument("--checkpoint_file", type=str, default=DEFAULT_CHECKPOINT_FILE,
                        help="File to load checkpoint from (if provided, this overrides the load directory)")
    return parser.parse_args()

def save_checkpoint(policy_net, total_steps, checkpoint_dir, batch_count, load_dir, task_idx):
    filename = f"snail_trpo_ckpt_batch{batch_count}.pth"
    checkpoint_file = os.path.join(checkpoint_dir, filename)
    chkload_file = os.path.join(load_dir, "chkload.pth")
    checkpoint = {
        "policy_state_dict": policy_net.state_dict(),
        "total_steps": total_steps,
        "batch_count": batch_count,
        "task_idx": task_idx  # Save the current trial index.
    }
    torch.save(checkpoint, checkpoint_file)
    torch.save(checkpoint, chkload_file)
    print(f"Checkpoint saved at step {total_steps} as {filename} (also updated chkload.pth in load dir)")

def load_checkpoint(policy_net, load_dir, checkpoint_file=None):
    if not checkpoint_file:
        checkpoint_file = os.path.join(load_dir, "chkload.pth")
    if os.path.isfile(checkpoint_file):
        print(f"Loading checkpoint: {checkpoint_file}")
        ckpt = torch.load(checkpoint_file, map_location=torch.device("cpu"))
        policy_net.load_state_dict(ckpt["policy_state_dict"])
        total_steps = ckpt.get("total_steps", 0)
        batch_count = ckpt.get("batch_count", 0)
        task_idx = ckpt.get("task_idx", 0)  # Retrieve the saved trial index.
        print(f"Resumed training from step {total_steps}, batch count {batch_count}, starting at trial index {task_idx}")
        return total_steps, batch_count, task_idx
    else:
        print("No checkpoint found, starting from scratch.")
        return 0, 0, 0

def pad_trials(trial_list, pad_value=0.0):
    """
    Pads a list of image trajectories so that each trial (shape: (L, 6, H, W))
    becomes shape (T, 6, H, W), where T is the maximum length.
    Returns a numpy array of shape (N, T, 6, H, W)
    """
    if not trial_list:
        return np.array([])
    max_T = max(trial.shape[0] for trial in trial_list)
    padded_trials = []
    for trial in trial_list:
        T = trial.shape[0]
        if T < max_T:
            pad_width = [(0, max_T - T), (0, 0), (0, 0), (0, 0)]
            trial_padded = np.pad(trial, pad_width, mode='constant', constant_values=pad_value)
        else:
            trial_padded = trial
        padded_trials.append(trial_padded)
    return np.stack(padded_trials, axis=0)

def pad_trials_1d(trial_list, pad_value=0):
    """
    Pads a list of 1D arrays (e.g. actions) so that each trial has the same length.
    Returns a numpy array of shape (N, T)
    """
    if not trial_list:
        return np.array([])
    max_T = max(trial.shape[0] for trial in trial_list)
    padded_trials = []
    for trial in trial_list:
        if trial.shape[0] < max_T:
            trial_padded = np.pad(trial, (0, max_T - trial.shape[0]), mode='constant', constant_values=pad_value)
        else:
            trial_padded = trial
        padded_trials.append(trial_padded)
    return np.stack(padded_trials, axis=0)

def main(args=None):
    if args is None:
        args = parse_args()

    TOTAL_TIMESTEPS = args.total_timesteps
    STEPS_PER_UPDATE = args.steps_per_update
    CHECKPOINT_DIR = args.checkpoint_dir
    LOAD_DIR = args.checkpoint_load_dir
    CHECKPOINT_FILE = args.checkpoint_file

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOAD_DIR, exist_ok=True)

    # Create environment.
    env = gym.make("MetaMazeDiscrete3D-v0", enable_render=False)
    tasks_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mazes_data", "train_tasks.json")
    with open(tasks_file, "r") as f:
        tasks_all = json.load(f)
    print(f"Loaded {len(tasks_all)} unique tasks.")

    # From the available tasks sample 200 tasks evenly.
    sampled_tasks = random.sample(tasks_all, 200)
    trial_tasks = []
    for task in sampled_tasks:
        for _ in range(50):
            trial_tasks.append(task)
    print(f"Using {len(trial_tasks)} trial tasks (each of 200 tasks repeated 50 times).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize SNAIL policy network and TRPO trainer.
    policy_net = SNAILPolicyValueNet(
        action_dim=3,  
        base_dim=256,
        policy_filters=32,
        policy_attn_dim=16,
        value_filters=16,
        seq_len=500,
        num_policy_attn=2,
        nb_features=64,
        num_heads=1
    ).to(device)

    trpo_trainer = TRPO_FO(
        policy=policy_net,
        value_fun=policy_net,
        simulator=None,
        max_kl_div=DEFAULT_MAX_KL_DIV,
        discount=DEFAULT_DISCOUNT,
        lam=DEFAULT_GAE_LAMBDA,
        vf_iters=DEFAULT_VF_ITERS,
        max_value_step=DEFAULT_VF_LR
    )

    total_steps = 0
    steps_since_update = 0
    batch_update_count = 0
    task_idx = 0
    replay_buffer = []

    # Load checkpoint (including saved trial index).
    total_steps, batch_update_count, task_idx = load_checkpoint(policy_net, LOAD_DIR, CHECKPOINT_FILE)

    pbar = tqdm(total=TOTAL_TIMESTEPS, initial=total_steps, desc="Training")

    # Main training loop.
    while total_steps < TOTAL_TIMESTEPS:
        # Reconstruct task configuration using the saved parameters.
        task_cfg = MazeTaskManager.TaskConfig(**trial_tasks[task_idx])
        env.unwrapped.set_task(task_cfg)
        task_idx = (task_idx + 1) % len(trial_tasks)

        obs_raw, _ = env.reset()

        # Randomize start and goal for this trial.
        env.unwrapped.maze_core.randomize_start()
        try:
            env.unwrapped.maze_core.randomize_goal(min_distance=3.0)
        except Exception as e:
            print(f"Warning: randomize_goal failed: {e}. Using current goal.")

        done = False
        truncated = False
        last_action = 0.0
        last_reward = 0.0
        boundary_bit = 0.0  # Initialize boundary bit to 0.

        prev_phase = env.unwrapped.maze_core.phase  # Get initial phase.

        trial_states = []    # List of (6, H, W) observations.
        trial_actions = []   # 1D array of actions.
        trial_rewards = []
        trial_values = []
        trial_log_probs = []

        # --- Initialize phase metrics for this trial ---
        phase1_steps = 0
        phase2_steps = 0
        phase1_reward = 0.0
        phase2_reward = 0.0
        # -----------------------------------------------------

        # Run one trial (episode).
        while not done and not truncated and total_steps < TOTAL_TIMESTEPS:
            # Check current phase; if transitioning from phase 1 to phase 2, set boundary_bit to 1.
            current_phase = env.unwrapped.maze_core.phase
            if prev_phase == 1 and current_phase == 2:
                boundary_bit = 1.0
            else:
                boundary_bit = 0.0
            prev_phase = current_phase

            obs_img = np.transpose(obs_raw, (2, 0, 1))  # (3, H, W)
            H, W = obs_img.shape[1], obs_img.shape[2]
            c3 = np.full((1, H, W), last_action, dtype=np.float32)
            c4 = np.full((1, H, W), last_reward, dtype=np.float32)
            c5 = np.full((1, H, W), boundary_bit, dtype=np.float32)
            obs_6ch = np.concatenate([obs_img, c3, c4, c5], axis=0)
            trial_states.append(obs_6ch)

            with torch.no_grad():
                obs_t = torch.from_numpy(obs_6ch[None, None]).float().to(device)
                logits, val = policy_net.act_single_step(obs_t)
                # Use the logits to sample an action.
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            obs_next, reward, done, truncated, info = env.step(action.item())

            trial_actions.append(action.item())
            trial_values.append(val.item())
            trial_log_probs.append(log_prob.item())
            trial_rewards.append(reward)

            # Update phase metrics based on current phase 
            if info["phase"] == 1:
                phase1_steps += 1
                phase1_reward += reward
            else:
                phase2_steps += 1
                phase2_reward += reward

            total_steps += 1
            steps_since_update += 1
            pbar.update(1)
            last_action = float(action.item())
            last_reward = float(reward)
            obs_raw = obs_next

        # End of trial: convert trial data to numpy arrays.
        trial_states_np = np.array(trial_states, dtype=np.float32)
        trial_actions_np = np.array(trial_actions, dtype=np.int64)
        trial_rewards_np = np.array(trial_rewards, dtype=np.float32)
        trial_values_np = np.array(trial_values, dtype=np.float32)
        trial_log_probs_np = np.array(trial_log_probs, dtype=np.float32)

        # Store trial data along with phase metrics in the replay buffer.
        replay_buffer.append({
            'states': trial_states_np,
            'actions': trial_actions_np,
            'rewards': trial_rewards_np,
            'values': trial_values_np,
            'log_probs': trial_log_probs_np,
            'phase1_steps': phase1_steps,
            'phase2_steps': phase2_steps,
            'phase1_reward': phase1_reward,
            'phase2_reward': phase2_reward
        })

        torch.cuda.empty_cache()

        if steps_since_update >= STEPS_PER_UPDATE:
            # Prepare rollouts and compute advantages.
            rollouts = []
            for data in replay_buffer:
                rollouts.append({
                    'states': data['states'],
                    'actions': data['actions'],
                    'rewards': data['rewards'],
                    'values': data['values']
                })
            all_advantages = trpo_trainer.get_advantages(rollouts)

            combined_states = []
            combined_actions = []
            combined_advantages = []
            combined_old_log_probs = []
            combined_values = []

            for buffer_item, advs in zip(replay_buffer, all_advantages):
                combined_states.append(buffer_item['states'])
                combined_actions.append(buffer_item['actions'])
                combined_advantages.append(advs)
                combined_old_log_probs.append(buffer_item['log_probs'])
                combined_values.append(buffer_item['values'])

            combined_states = np.concatenate(combined_states, axis=0)
            combined_actions = np.concatenate(combined_actions, axis=0)
            combined_advantages = np.concatenate(combined_advantages, axis=0)
            combined_old_log_probs = np.concatenate(combined_old_log_probs, axis=0)
            combined_values = np.concatenate(combined_values, axis=0)

            # Aggregate phase metrics over the batch.
            total_phase1_steps = 0
            total_phase2_steps = 0
            total_phase1_reward = 0.0
            total_phase2_reward = 0.0
            num_trials = len(replay_buffer)
            for data in replay_buffer:
                total_phase1_steps += data.get('phase1_steps', 0)
                total_phase2_steps += data.get('phase2_steps', 0)
                total_phase1_reward += data.get('phase1_reward', 0.0)
                total_phase2_reward += data.get('phase2_reward', 0.0)
            avg_phase1_steps = total_phase1_steps / num_trials if num_trials > 0 else 0
            avg_phase2_steps = total_phase2_steps / num_trials if num_trials > 0 else 0
            avg_phase1_reward = total_phase1_reward / num_trials if num_trials > 0 else 0.0
            avg_phase2_reward = total_phase2_reward / num_trials if num_trials > 0 else 0.0

            combined_states_t = torch.FloatTensor(combined_states).to(device)
            combined_actions_t = torch.LongTensor(combined_actions).to(device)
            combined_advantages_t = torch.FloatTensor(combined_advantages).to(device)
            combined_old_log_probs_t = torch.FloatTensor(combined_old_log_probs).to(device)
            combined_values_t = torch.FloatTensor(combined_values).to(device)

            policy_loss, _ = trpo_trainer.update_policy(
                combined_states_t,
                combined_actions_t,
                combined_advantages_t,
                combined_old_log_probs_t
            )
            final_vloss = trpo_trainer.update_value_fun(
                combined_states_t,
                combined_values_t
            )
            print(f"[TRPO UPDATE] Steps: {total_steps}, KL: {trpo_trainer.current_kl:.4f}, "
                  f"Policy Loss: {policy_loss:.4f}, Value Loss: {final_vloss:.4f}, "
                  f"Avg Phase1 Steps: {avg_phase1_steps:.2f}, Avg Phase2 Steps: {avg_phase2_steps:.2f}, "
                  f"Avg Phase1 Reward: {avg_phase1_reward:.4f}, Avg Phase2 Reward: {avg_phase2_reward:.4f}")

            batch_update_count += 1
            save_checkpoint(policy_net, total_steps, CHECKPOINT_DIR, batch_update_count, LOAD_DIR, task_idx)
            replay_buffer.clear()
            steps_since_update = 0
            torch.cuda.empty_cache()

    if steps_since_update >= STEPS_PER_UPDATE:
        rollouts = []
        for data in replay_buffer:
            rollouts.append({
                'states': data['states'],
                'actions': data['actions'],
                'rewards': data['rewards'],
                'values': data['values']
            })
        all_advantages = trpo_trainer.get_advantages(rollouts)
        combined_states = pad_trials([data['states'] for data in replay_buffer])
        combined_actions = pad_trials_1d([data['actions'] for data in replay_buffer])
        combined_advantages = pad_trials_1d(all_advantages)
        combined_old_log_probs = pad_trials_1d([data['log_probs'] for data in replay_buffer])
        combined_values = pad_trials_1d([data['values'] for data in replay_buffer])

        combined_states_t = torch.FloatTensor(combined_states).to(device)
        combined_actions_t = torch.LongTensor(combined_actions).to(device)
        combined_advantages_t = torch.FloatTensor(combined_advantages).to(device)
        combined_old_log_probs_t = torch.FloatTensor(combined_old_log_probs).to(device)
        combined_values_t = torch.FloatTensor(combined_values).to(device)

        policy_loss, _ = trpo_trainer.update_policy(
            combined_states_t,
            combined_actions_t,
            combined_advantages_t,
            combined_old_log_probs_t
        )
        final_vloss = trpo_trainer.update_value_fun(
            combined_states_t,
            combined_values_t
        )
        print(f"[FINAL TRPO UPDATE] Steps: {total_steps}, KL: {trpo_trainer.current_kl:.4f}, "
              f"Policy Loss: {policy_loss:.4f}, Value Loss: {final_vloss:.4f}, "
              f"Avg Phase1 Steps: {avg_phase1_steps:.2f}, Avg Phase2 Steps: {avg_phase2_steps:.2f}, "
              f"Avg Phase1 Reward: {avg_phase1_reward:.4f}, Avg Phase2 Reward: {avg_phase2_reward:.4f}")
        batch_update_count += 1
        save_checkpoint(policy_net, total_steps, CHECKPOINT_DIR, batch_update_count, LOAD_DIR, task_idx)

    pbar.close()
    env.close()
    save_checkpoint(policy_net, total_steps, CHECKPOINT_DIR, batch_update_count, LOAD_DIR, task_idx)
    print(f"Training completed. Total steps: {total_steps}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
