# metamazium/train_model/train_lstm_trpo.py

"""
Train an LSTM-TRPO agent on the MetaMaze environment.

This script uses a recurrent policy (StackedLSTMPolicyValueNet) and the TRPO algorithm
to learn to navigate in the MetaMaze environment. Hyperparameters may be adjusted via
command-line arguments.
"""

import os
import json
import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
import copy
import time
import argparse

torch.autograd.set_detect_anomaly(False)

import metamazium.env
from metamazium.lstm_trpo.lstm_model import StackedLSTMPolicyValueNet
from metamazium.lstm_trpo.trpo import TRPO
from metamazium.env.maze_task import MazeTaskSampler

# Default hyperparameters
DEFAULT_TOTAL_TIMESTEPS = 100000
DEFAULT_STEPS_PER_UPDATE = 30000  
DEFAULT_GAMMA = 0.99
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_MAX_KL_DIV = 0.01
DEFAULT_CG_ITERS = 10
DEFAULT_DAMPING = 0.1
DEFAULT_VF_LR = 0.01
DEFAULT_VF_ITERS = 5
DEFAULT_ENTROPY_COEF = 0.01
DEFAULT_CHECKPOINT_DIR = "checkpoint_trpo"
DEFAULT_CHECKPOINT_FILE = None


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train LSTM-TRPO on MetaMaze environment."
    )
    parser.add_argument("--total_timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS,
                        help="Total training timesteps (default: 100000)")
    parser.add_argument("--steps_per_update", type=int, default=DEFAULT_STEPS_PER_UPDATE,
                        help="Timesteps collected per TRPO update (default: 30000)")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help="Directory to save checkpoints (default: checkpoint_trpo)")
    parser.add_argument("--checkpoint_file", type=str, default=DEFAULT_CHECKPOINT_FILE,
                        help="File to load checkpoint from (default: None)")
    return parser.parse_args()


def pad_trials(trial_list, pad_value=0.0):
    """
    Pad each trial in the list to the same length and stack them.
    
    Args:
        trial_list (list[np.ndarray]): List of trial arrays.
        pad_value (float): Value to pad with (default: 0.0).
    
    Returns:
        np.ndarray: Array with shape (num_trials, max_trial_length, ...).
    """
    if not trial_list:
        return np.array([])
    max_T = max(trial.shape[0] for trial in trial_list)
    padded_trials = []
    for trial in trial_list:
        T = trial.shape[0]
        if T < max_T:
            pad_width = [(0, max_T - T)] + [(0, 0)] * (trial.ndim - 1)
            trial_padded = np.pad(trial, pad_width, mode='constant', constant_values=pad_value)
        else:
            trial_padded = trial
        padded_trials.append(trial_padded)
    return np.stack(padded_trials, axis=0)


def _batch_rnn_state(rnn_state_list):
    """
    Batch a list of RNN state tuples (hidden, cell) into a single tuple.
    
    Args:
        rnn_state_list (list[tuple[torch.Tensor, torch.Tensor]]): List of (hidden, cell) states.
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Batched (hidden, cell) states.
    """
    if not rnn_state_list:
        return None
    h_list = [state[0] for state in rnn_state_list]
    c_list = [state[1] for state in rnn_state_list]
    batched_hidden = torch.stack(h_list, dim=1).squeeze(2)
    batched_cell = torch.stack(c_list, dim=1).squeeze(2)
    return batched_hidden, batched_cell


def save_checkpoint(policy_net, trainer, total_steps, checkpoint_dir):
    """
    Save a checkpoint of the model and training progress.
    
    Args:
        policy_net (torch.nn.Module): The policy network.
        trainer (TRPO): The TRPO trainer instance.
        total_steps (int): Total training steps completed.
        checkpoint_dir (str): Directory in which to save the checkpoint.
    """
    checkpoint_file = os.path.join(checkpoint_dir, "trpo_ckpt.pth")
    torch.save({
        "policy_state_dict": policy_net.state_dict(),
        "total_steps": total_steps,
    }, checkpoint_file)
    print(f"Checkpoint saved at step {total_steps} -> {checkpoint_file}")


def main(args=None):
    """
    Main training function for the LSTM-TRPO agent.
    
    Args:
        args (argparse.Namespace, optional): Parsed command-line arguments.
    """
    if args is None:
        args = parse_args()

    TOTAL_TIMESTEPS = args.total_timesteps
    STEPS_PER_UPDATE = args.steps_per_update
    CHECKPOINT_DIR = args.checkpoint_dir
    CHECKPOINT_FILE = args.checkpoint_file

    env = gym.make("MetaMazeDiscrete3D-v0", enable_render=False)
    tasks_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "mazes_data", "train_tasks.json")
    with open(tasks_file, "r") as f:
        tasks = json.load(f)
    print(f"Loaded {len(tasks)} tasks.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize policy network and TRPO trainer
    policy_net = StackedLSTMPolicyValueNet(action_dim=4, hidden_size=512, num_layers=2).to(device)
    trpo_trainer = TRPO(
        policy=policy_net,
        value_fun=policy_net,
        simulator=None,
        max_kl_div=DEFAULT_MAX_KL_DIV,
        discount=DEFAULT_GAMMA,
        lam=DEFAULT_GAE_LAMBDA,
        cg_damping=DEFAULT_DAMPING,
        cg_max_iters=DEFAULT_CG_ITERS,
        vf_iters=DEFAULT_VF_ITERS,
        max_value_step=DEFAULT_VF_LR
    )

    total_steps = 0
    steps_since_update = 0
    task_idx = 0
    replay_buffer = []

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_file = os.path.join(CHECKPOINT_DIR, "trpo_ckpt.pth") if CHECKPOINT_FILE is None else CHECKPOINT_FILE

    if os.path.isfile(checkpoint_file):
        print(f"Loading checkpoint: {checkpoint_file}")
        ckpt = torch.load(checkpoint_file, map_location=device)
        policy_net.load_state_dict(ckpt["policy_state_dict"])
        total_steps = ckpt.get("total_steps", 0)
        print(f"Resumed training from step {total_steps}")

    pbar = tqdm(total=TOTAL_TIMESTEPS, initial=total_steps, desc="Training")

    while total_steps < TOTAL_TIMESTEPS:
        # Set task
        task_cfg = MazeTaskSampler(**tasks[task_idx])
        env.unwrapped.set_task(task_cfg)
        task_idx = (task_idx + 1) % len(tasks)

        # Reset environment and RNN memory
        policy_net.reset_memory(batch_size=1, device=device)
        obs_raw, _ = env.reset()
        env.unwrapped.maze_core.randomize_start()

        done = False
        truncated = False
        last_action = 0.0
        last_reward = 0.0
        boundary_bit = 1.0

        trial_states = []
        trial_actions = []
        trial_rewards = []
        trial_values = []
        trial_log_probs = []

        # Run one trial
        while not done and not truncated and total_steps < TOTAL_TIMESTEPS:
            obs_img = np.transpose(obs_raw, (2, 0, 1))
            H, W = obs_img.shape[1], obs_img.shape[2]
            c3 = np.full((1, H, W), last_action, dtype=np.float32)
            c4 = np.full((1, H, W), last_reward, dtype=np.float32)
            c5 = np.full((1, H, W), boundary_bit, dtype=np.float32)
            obs_6ch = np.concatenate([obs_img, c3, c4, c5], axis=0)

            with torch.no_grad():
                obs_t = torch.from_numpy(obs_6ch[None, None]).float().to(device)
                logits_t, val_t = policy_net.act_single_step(obs_t)
                dist = torch.distributions.Categorical(logits=logits_t.squeeze(1))
                action = dist.sample()
                log_prob = dist.log_prob(action)

            obs_next, reward, done, truncated, info = env.step(action.item())

            trial_states.append(obs_6ch)
            trial_actions.append(action.item())
            trial_rewards.append(reward)
            trial_values.append(val_t.item())
            trial_log_probs.append(log_prob.item())

            total_steps += 1
            steps_since_update += 1
            pbar.update(1)
            last_action = float(action.item())
            last_reward = float(reward)
            obs_raw = obs_next

        trial_states = np.array(trial_states, dtype=np.float32)
        trial_actions = np.array(trial_actions, dtype=np.int64)
        trial_rewards = np.array(trial_rewards, dtype=np.float32)
        trial_values = np.array(trial_values, dtype=np.float32)
        trial_log_probs = np.array(trial_log_probs, dtype=np.float32)

        replay_buffer.append({
            'states': trial_states,
            'actions': trial_actions,
            'rewards': trial_rewards,
            'values': trial_values,
            'log_probs': trial_log_probs
        })

        # Update policy using TRPO if enough steps have been collected
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

            combined_states_t = torch.FloatTensor(combined_states).to(device)
            combined_actions_t = torch.LongTensor(combined_actions).to(device)
            combined_advantages_t = torch.FloatTensor(combined_advantages).to(device)
            combined_old_log_probs_t = torch.FloatTensor(combined_old_log_probs).to(device)
            combined_values_t = torch.FloatTensor(combined_values).to(device)

            policy_loss, value_loss = trpo_trainer.update_policy(
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
                  f"Policy Loss: {policy_loss:.4f}, Value Loss: {final_vloss:.4f}")

            save_checkpoint(policy_net, trpo_trainer, total_steps, CHECKPOINT_DIR)
            replay_buffer.clear()
            steps_since_update = 0

        if total_steps % (STEPS_PER_UPDATE * 2) == 0:
            save_checkpoint(policy_net, trpo_trainer, total_steps, CHECKPOINT_DIR)

    pbar.close()
    env.close()
    save_checkpoint(policy_net, trpo_trainer, total_steps, CHECKPOINT_DIR)
    print(f"Training completed. Total steps: {total_steps}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
