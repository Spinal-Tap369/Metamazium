"""
Test script for sampling 2 unique maze tasks (from unique_mazes.json) and running 3 trials per task.
For each trial, the environment will randomize the start and goal positions (with a minimum distance)
so that the same maze layout is used but the agent’s starting location and goal are different.
Keyboard control is preserved so you control the agent using the arrow keys.
Each trial runs until the user terminates (by pressing ESC or closing the window) or a maximum number of steps is reached.
"""

import gymnasium as gym
import json
import random
import sys
import time
import pygame
import argparse
import numpy as np
import copy

from metamazium.env.maze_task import MazeTaskManager  # Use TaskConfig for reconstruction

# Discrete actions: Left, Right, Down, Up – used only for displaying instructions.
DISCRETE_ACTIONS = [(-1, 0), (1, 0), (0, 1)]

def load_tasks(file_path):
    """Load tasks from a JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{file_path}': {e}")
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Maze environment with keyboard control and randomized start/goal for unique tasks."
    )
    parser.add_argument("--task_file", type=str, default="mazes_data_try/train_tasks.json",
                        help="Path to the JSON file with unique task definitions")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum steps to run each trial")
    return parser.parse_args()

def run_trial(env, max_steps):
    """
    Runs a single trial (episode) using keyboard control.
    The trial runs until the user presses ESC or the maximum steps is reached.
    Returns the number of steps taken.
    """
    try:
        obs, info = env.reset()
    except Exception as e:
        print(f"Error during reset: {e}")
        sys.exit(1)

    steps = 0
    terminated = False
    truncated = False

    print("Trial started. Use arrow keys to control the agent. Press ESC to end the trial.")
    while not terminated and not truncated and steps < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                break
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                terminated = True
                break

        if terminated:
            break

        keys = pygame.key.get_pressed()
        # Use keyboard control: if no key is pressed, do nothing.
        if keys[pygame.K_LEFT]:
            action = 0
        elif keys[pygame.K_RIGHT]:
            action = 1
        elif keys[pygame.K_UP]:
            action = 2
        else:
            # If no key is pressed, continue waiting.
            time.sleep(0.05)
            continue

        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            print(f"Error during step: {e}")
            break
        steps += 1
        print(f"Step {steps}: Action {action}, Reward {reward:.3f}")
        try:
            env.render()
        except Exception as e:
            print(f"Error rendering: {e}")
        time.sleep(0.05)
    return steps, info

def main(args=None):
    if args is None:
        args = parse_args()

    pygame.init()
    # Set up a minimal display for rendering.
    pygame.display.set_mode((1, 1))

    tasks_all = load_tasks(args.task_file)
    print(f"Loaded {len(tasks_all)} unique tasks.")

    # Randomly sample 2 unique tasks.
    if len(tasks_all) < 2:
        print("Not enough tasks to sample 2 unique tasks.")
        sys.exit(1)
    selected_tasks = random.sample(tasks_all, 2)
    print("Selected 2 unique tasks for testing.")

    total_trials = 0
    trial_results = []

    # Create the environment once.
    env_id = "MetaMazeDiscrete3D-v0"
    try:
        env = gym.make(env_id, enable_render=True)
    except Exception as e:
        print(f"Error creating environment '{env_id}': {e}")
        sys.exit(1)

    for task_dict in selected_tasks:
        # Reconstruct the task configuration using MazeTaskManager.TaskConfig.
        task_config = MazeTaskManager.TaskConfig(**task_dict)
        print(f"\nUsing task with base goal at: {task_config.goal}")

        # Set the task into the environment.
        if hasattr(env, "set_task"):
            env.set_task(task_config)
        elif hasattr(env.unwrapped, "set_task"):
            env.unwrapped.set_task(task_config)
        else:
            print("Error: Environment does not support 'set_task'.")
            sys.exit(1)

        # Run 3 trials for this task.
        for trial in range(3):
            # Randomize the start and goal for this trial.
            env.unwrapped.maze_core.randomize_start()
            try:
                env.unwrapped.maze_core.randomize_goal(min_distance=3.0)
            except Exception as e:
                print(f"Warning: randomize_goal failed: {e}. Using current goal.")
            print(f"Trial {trial+1}: randomized start = {env.unwrapped.maze_core._start}, goal = {env.unwrapped.maze_core._goal}")

            steps, info = run_trial(env, args.max_steps)
            print(f"Trial completed in {steps} steps.")
            trial_results.append({
                "task_goal": env.unwrapped.maze_core._goal,
                "steps": steps,
                "info": info
            })
            total_trials += 1

    env.close()
    pygame.quit()

    print(f"\nCompleted {total_trials} trials (2 tasks x 3 trials each).")
    # Optionally, save the trial results to a JSON file.
    # with open("mazes_data_try/test_trial_results.json", "w") as f:
    #     json.dump(trial_results, f, indent=2)
    # print("Trial results saved to mazes_data_try/test_trial_results.json")

if __name__ == "__main__":
    main()
