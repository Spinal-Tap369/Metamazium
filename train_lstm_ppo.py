import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import json
from tqdm import tqdm

import env  # Importing custom Maze environment definitions.
from lstm_ppo.lstm_model import StackedLSTMPolicy
from lstm_ppo.ppo import PPOTrainer
from env.maze_task import MazeTaskSampler

def main():
    """
    Main training loop for the stacked LSTM PPO model in the MetaMaze environment.
    Loads tasks, initializes the environment and models, collects rollouts,
    and performs PPO updates until a specified number of timesteps is reached.
    """
    env_id = "MetaMazeDiscrete3D-v0"
    env = gym.make(env_id, enable_render=False)

    # Load training tasks from a JSON file.
    tasks_file = "mazes_data/train_tasks.json"
    with open(tasks_file, "r") as f:
        tasks = json.load(f)
    num_tasks = len(tasks)
    print(f"Loaded {num_tasks} tasks.")

    # Hyperparameters.
    total_timesteps = 4000
    steps_per_update = 2000
    gamma = 0.99
    gae_lambda = 0.99
    clip_range = 0.2
    target_kl = 0.03
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the stacked LSTM policy network.
    policy_net = StackedLSTMPolicy(
        action_dim=4,
        hidden_size=512,
        num_layers=2
    ).to(device)

    # Initialize PPO trainer with specified hyperparameters.
    ppo_trainer = PPOTrainer(
        policy_model=policy_net,
        lr=1e-4,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        n_epochs=3,
        target_kl=target_kl,
        max_grad_norm=0.5,
        entropy_coef=0.01,
        value_coef=0.5
    )

    total_steps = 0
    task_idx = 0
    pbar = tqdm(total=total_timesteps, desc="Training")

    # Buffers for storing rollout data.
    obs_buffer = []
    act_buffer = []
    logp_buffer = []
    val_buffer = []
    rew_buffer = []
    done_buffer = []

    while total_steps < total_timesteps:
        # Select the next task and configure the environment.
        task_cfg = MazeTaskSampler(**tasks[task_idx])
        env.unwrapped.set_task(task_cfg)
        task_idx = (task_idx + 1) % num_tasks

        obs_raw, _ = env.reset()
        done = False
        truncated = False

        # Initialize variables for additional channels.
        last_action = 0.0
        last_reward = 0.0
        boundary_bit = 1.0
        phase_boundary_signaled = False

        # Temporary storage for episode data.
        ep_obs_seq = []  # List to store sequence of observations with 6 channels.
        ep_actions = []
        ep_logprobs = []
        ep_values = []
        ep_rewards = []
        ep_dones = []

        while not done and not truncated and total_steps < total_timesteps:
            # Construct a 6-channel observation by combining image data with additional information.
            obs_img = np.transpose(obs_raw, (2, 0, 1))  # Convert to shape (3, H, W).
            H, W = obs_img.shape[1], obs_img.shape[2]
            c3 = np.full((1, H, W), last_action, dtype=np.float32)
            c4 = np.full((1, H, W), last_reward, dtype=np.float32)
            c5 = np.full((1, H, W), boundary_bit, dtype=np.float32)
            obs_6ch = np.concatenate([obs_img, c3, c4, c5], axis=0)  # Shape: (6, H, W)

            # Append the 6-channel observation to the episode sequence.
            ep_obs_seq.append(obs_6ch)
            t_len = len(ep_obs_seq)

            # Prepare input tensor for the current sequence.
            obs_seq_np = np.stack(ep_obs_seq, axis=0)  # Shape: (t_len, 6, H, W)
            obs_seq_np = obs_seq_np[None]             # Add batch dimension: (1, t_len, 6, H, W)
            obs_seq_torch = torch.from_numpy(obs_seq_np).float().to(device)

            # Obtain policy logits and value estimates for the current sequence.
            with torch.no_grad():
                logits_seq, vals_seq = policy_net(obs_seq_torch)

            # Select the logits and value corresponding to the latest timestep.
            logits_t = logits_seq[:, t_len-1, :]  # Shape: (1, action_dim)
            val_t = vals_seq[:, t_len-1]         # Shape: (1,)

            # Sample an action from the policy distribution.
            dist = torch.distributions.Categorical(logits=logits_t)
            action = dist.sample()
            logp = dist.log_prob(action)

            # Execute the action in the environment.
            obs_next, reward, done, truncated, info = env.step(action.item())
            total_steps += 1
            pbar.update(1)

            # Store experience from the current timestep.
            ep_actions.append(action.item())
            ep_logprobs.append(logp.item())
            ep_values.append(val_t.item())
            ep_rewards.append(reward)
            ep_dones.append(float(done))

            # Update variables for the next iteration.
            last_action = float(action.item())
            last_reward = float(reward)
            obs_raw = obs_next

            # Signal phase boundary if transitioning from phase 1 to phase 2.
            if env.unwrapped.maze_core.phase == 2 and not phase_boundary_signaled:
                boundary_bit = 1.0
                phase_boundary_signaled = True
            else:
                boundary_bit = 0.0

        # After episode ends, accumulate episode data into main buffers.
        obs_buffer += ep_obs_seq
        act_buffer += ep_actions
        logp_buffer += ep_logprobs
        val_buffer += ep_values
        rew_buffer += ep_rewards
        done_buffer += ep_dones

        # Perform a PPO update once enough data is collected.
        if len(obs_buffer) >= steps_per_update:
            do_update(policy_net, ppo_trainer,
                      obs_buffer, act_buffer, logp_buffer,
                      val_buffer, rew_buffer, done_buffer, device)
            obs_buffer.clear()
            act_buffer.clear()
            logp_buffer.clear()
            val_buffer.clear()
            rew_buffer.clear()
            done_buffer.clear()

    pbar.close()
    env.close()

    # Perform a final update if there is any remaining data.
    if len(obs_buffer) > 0:
        do_update(policy_net, ppo_trainer,
                  obs_buffer, act_buffer, logp_buffer,
                  val_buffer, rew_buffer, done_buffer, device)

    # Save the trained policy model.
    torch.save(policy_net.state_dict(), "models/lstm_stacked_online.pt")
    print(f"Training finished after {total_steps} steps.")

def do_update(policy_net, trainer,
              obs_buf, act_buf, logp_buf,
              val_buf, rew_buf, done_buf, device):
    """
    Conducts a PPO update using collected rollout buffers.

    Steps:
      1. Reshape observation data to shape (1, T, 6, H, W).
      2. Compute Generalized Advantage Estimation (GAE).
      3. Invoke the trainer's update method with the prepared rollout data.

    Args:
        policy_net (nn.Module): The policy network.
        trainer (PPOTrainer): The PPO trainer instance.
        obs_buf (list): List of observation arrays.
        act_buf (list): List of actions taken.
        logp_buf (list): List of log probabilities of actions.
        val_buf (list): List of value estimates.
        rew_buf (list): List of rewards received.
        done_buf (list): List of done flags.
        device (torch.device): The device to run computations on.
    """
    T = len(obs_buf)
    if T < 2:
        return

    # Convert buffers into NumPy arrays with appropriate shapes.
    obs_np = np.stack(obs_buf, axis=0)  # Shape: (T, 6, H, W)
    obs_np = obs_np[None]               # Shape: (1, T, 6, H, W)
    acts_np = np.array(act_buf, dtype=np.int64)[None]     # Shape: (1, T)
    logp_np = np.array(logp_buf, dtype=np.float32)[None]  # Shape: (1, T)
    vals_np = np.array(val_buf, dtype=np.float32)[None]   # Shape: (1, T)
    rews_np = np.array(rew_buf, dtype=np.float32)[None]   # Shape: (1, T)
    done_np = np.array(done_buf, dtype=np.float32)[None]  # Shape: (1, T)

    B, T_ = acts_np.shape
    next_value = 0.0  # No next value if the final step was terminal.

    # Flatten rewards, dones, and values for GAE computation.
    rewards_ = rews_np.reshape(-1)
    dones_ = done_np.reshape(-1)
    values_ = vals_np.reshape(-1)

    # Compute advantages and returns.
    advantages_ = trainer.compute_gae(rewards_, dones_, values_, next_value)
    returns_ = values_ + advantages_

    # Reshape advantages and returns to match expected dimensions.
    adv_2d = advantages_.reshape(B, T_)
    ret_2d = returns_.reshape(B, T_)

    # Prepare the rollout dictionary required by the PPO update.
    rollouts = {
        "obs": obs_np,             # Shape: (1, T, 6, H, W)
        "actions": acts_np,        # Shape: (1, T)
        "old_log_probs": logp_np,  # Shape: (1, T)
        "returns": ret_2d,         # Shape: (1, T)
        "values": vals_np,         # Shape: (1, T)
        "advantages": adv_2d       # Shape: (1, T)
    }

    # Execute the PPO update and display statistics.
    stats = trainer.update(rollouts)
    print("[PPO Update]", stats)

if __name__ == "__main__":
    main()
