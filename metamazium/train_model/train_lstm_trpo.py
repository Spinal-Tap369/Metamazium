# train_lstm_trpo.py

import os
import json
import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
import copy
import time
torch.autograd.set_detect_anomaly(True)
import metamazium.env
from metamazium.lstm_trpo.lstm_model import StackedLSTMPolicyValueNet
from metamazium.lstm_trpo.trpo import TRPO
from metamazium.env.maze_task import MazeTaskSampler

# ---------------------------------------------------------------------------
# Hyperparameters (Hard-coded)
# ---------------------------------------------------------------------------
TOTAL_TIMESTEPS = 100000
STEPS_PER_UPDATE = 30000  
GAMMA = 0.99
GAE_LAMBDA = 0.95
MAX_KL_DIV = 0.01
CG_ITERS = 10
DAMPING = 0.1
VF_LR = 0.01
VF_ITERS = 5
ENTROPY_COEF = 0.01
CHECKPOINT_DIR = "checkpoint_trpo"
CHECKPOINT_FILE = None  

# ---------------------------------------------------------------------------
# Utility Functions (Same as PPO)
# ---------------------------------------------------------------------------
def pad_trials(trial_list, pad_value=0.0):
    if len(trial_list) == 0:
        return np.array([])
    max_T = max(trial.shape[0] for trial in trial_list)
    padded_trials = []
    for trial in trial_list:
        T = trial.shape[0]
        if T < max_T:
            pad_width = [(0, max_T - T)] + [(0,0)]*(trial.ndim-1)
            trial_padded = np.pad(trial, pad_width, mode='constant', constant_values=pad_value)
        else:
            trial_padded = trial
        padded_trials.append(trial_padded)
    return np.stack(padded_trials, axis=0)

def _batch_rnn_state(rnn_state_list):
    """
    Utility to batch up a list of (hidden, cell) tuples. Not strictly needed 
    in this new approach, but kept here if you want to do any RNN batching.
    """
    if len(rnn_state_list) == 0:
        return None
    h_list = [state[0] for state in rnn_state_list]
    c_list = [state[1] for state in rnn_state_list]
    batched_hidden = torch.stack(h_list, dim=1).squeeze(2)
    batched_cell = torch.stack(c_list, dim=1).squeeze(2)
    return (batched_hidden, batched_cell)

# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------
def main():
    env = gym.make("MetaMazeDiscrete3D-v0", enable_render=False)
    # Example tasks file
    tasks_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "..", "mazes_data", "train_tasks.json")

    with open(tasks_file, "r") as f:
        tasks = json.load(f)
    print(f"Loaded {len(tasks)} tasks.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize network
    policy_net = StackedLSTMPolicyValueNet(action_dim=4, hidden_size=512, num_layers=2).to(device)

    # Initialize TRPO (using value function from the policy network)
    trpo_trainer = TRPO(
        policy=policy_net,
        value_fun=policy_net,  # Using policy network's value head
        simulator=None,        # We'll handle our own sampling
        max_kl_div=MAX_KL_DIV,
        discount=GAMMA,
        lam=GAE_LAMBDA,
        cg_damping=DAMPING,
        cg_max_iters=CG_ITERS,
        vf_iters=VF_ITERS,
        max_value_step=VF_LR
    )

    total_steps = 0
    steps_since_update = 0  # counts how many steps we've gathered since last TRPO update

    # We'll cycle through tasks in a round-robin style
    task_idx = 0

    # This buffer will hold data from multiple trials
    # until steps_since_update >= STEPS_PER_UPDATE
    replay_buffer = []

    # Checkpoint setup
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_file = CHECKPOINT_FILE or os.path.join(CHECKPOINT_DIR, "trpo_ckpt.pth")

    # Possibly load from checkpoint
    if os.path.isfile(checkpoint_file):
        print(f"Loading checkpoint: {checkpoint_file}")
        ckpt = torch.load(checkpoint_file, map_location=device)
        policy_net.load_state_dict(ckpt["policy_state_dict"])
        total_steps = ckpt.get("total_steps", 0)
        print(f"Resumed training from step {total_steps}")

    pbar = tqdm(total=TOTAL_TIMESTEPS, initial=total_steps, desc="Training")

    # Training loop
    while total_steps < TOTAL_TIMESTEPS:
        # -----------------------------
        # 1) SET THE TASK (Maze)
        # -----------------------------
        task_cfg = MazeTaskSampler(**tasks[task_idx])
        env.unwrapped.set_task(task_cfg)
        task_idx = (task_idx + 1) % len(tasks)

        # -----------------------------
        # 2) RESET environment + LSTM
        # -----------------------------
        policy_net.reset_memory(batch_size=1, device=device)
        obs_raw, _ = env.reset()
        env.unwrapped.maze_core.randomize_start()

        done = False
        truncated = False

        # Keep track of last_action, last_reward for 6-channel input
        last_action = 0.0
        last_reward = 0.0
        boundary_bit = 1.0  # 1.0 means new trial boundary (some designs use it)

        # Single trial data (two-phase environment internally).
        trial_states = []
        trial_actions = []
        trial_rewards = []
        trial_values = []
        trial_log_probs = []

        # -----------------------------
        # 3) Run the TRIAL to completion
        #    (including exploration & test phases)
        # -----------------------------
        while not done and not truncated and total_steps < TOTAL_TIMESTEPS:
            # Build 6-channel observation
            obs_img = np.transpose(obs_raw, (2, 0, 1))  # (C, H, W)
            H, W = obs_img.shape[1], obs_img.shape[2]
            c3 = np.full((1, H, W), last_action, dtype=np.float32)
            c4 = np.full((1, H, W), last_reward, dtype=np.float32)
            c5 = np.full((1, H, W), boundary_bit, dtype=np.float32)
            obs_6ch = np.concatenate([obs_img, c3, c4, c5], axis=0)

            # Step policy forward
            with torch.no_grad():
                obs_t = torch.from_numpy(obs_6ch[None, None]).float().to(device)
                logits_t, val_t = policy_net.act_single_step(obs_t)
                dist = torch.distributions.Categorical(logits=logits_t.squeeze(1))
                action = dist.sample()
                log_prob = dist.log_prob(action)

            # Step environment
            obs_next, reward, done, truncated, info = env.step(action.item())

            # Store step in trial
            trial_states.append(obs_6ch)
            trial_actions.append(action.item())
            trial_rewards.append(reward)
            trial_values.append(val_t.item())
            trial_log_probs.append(log_prob.item())

            # Update counters
            total_steps += 1
            steps_since_update += 1
            pbar.update(1)
            last_action = float(action.item())
            last_reward = float(reward)
            obs_raw = obs_next

        # One full trial is done (both phases). We store it in the replay buffer.
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

        # -----------------------------------------
        # 4) Check if we have enough steps to do a TRPO update
        #    and only do it after a trial ends
        # -----------------------------------------
        if steps_since_update >= STEPS_PER_UPDATE:
            # a) Construct rollouts for advantage calc
            rollouts = []
            for data in replay_buffer:
                rollouts.append({
                    'states': data['states'],
                    'actions': data['actions'],
                    'rewards': data['rewards'],
                    'values': data['values']
                })

            # b) Compute advantages
            all_advantages = trpo_trainer.get_advantages(rollouts)

            # c) Combine all transitions from replay_buffer
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

            # d) Convert to torch
            combined_states_t = torch.FloatTensor(combined_states).to(device)
            combined_actions_t = torch.LongTensor(combined_actions).to(device)
            combined_advantages_t = torch.FloatTensor(combined_advantages).to(device)
            combined_old_log_probs_t = torch.FloatTensor(combined_old_log_probs).to(device)
            combined_values_t = torch.FloatTensor(combined_values).to(device)

            # e) TRPO policy update
            policy_loss, value_loss = trpo_trainer.update_policy(
                combined_states_t,
                combined_actions_t,
                combined_advantages_t,
                combined_old_log_probs_t
            )

            # f) TRPO value update
            final_vloss = trpo_trainer.update_value_fun(
                combined_states_t,
                combined_values_t
            )

            print(f"[TRPO UPDATE] Steps: {total_steps}, KL: {trpo_trainer.current_kl:.4f}, "
                  f"Policy Loss: {policy_loss:.4f}, Value Loss: {final_vloss:.4f}")

            # g) Clear the replay buffer and reset steps_since_update
            replay_buffer.clear()
            steps_since_update = 0

        # Periodically save checkpoint
        # (We'll do it every STEPS_PER_UPDATE * 2 as in your code, or any interval you like)
        if total_steps % (STEPS_PER_UPDATE * 2) == 0:
            save_checkpoint(policy_net, trpo_trainer, total_steps)

    # Done
    pbar.close()
    env.close()
    save_checkpoint(policy_net, trpo_trainer, total_steps)
    print(f"Training completed. Total steps: {total_steps}")

# ---------------------------------------------------------------------------
# Checkpoint Saving
# ---------------------------------------------------------------------------
def save_checkpoint(policy_net, trainer, total_steps):
    checkpoint_file = os.path.join(CHECKPOINT_DIR, "trpo_ckpt.pth")
    torch.save({
        "policy_state_dict": policy_net.state_dict(),
        "total_steps": total_steps,
        # If needed, you can store other info like hyperparameters here
    }, checkpoint_file)
    print(f"Checkpoint saved at step {total_steps} -> {checkpoint_file}")

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
