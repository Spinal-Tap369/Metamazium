# test_fsc.py

import gymnasium as gym
import json
import random
import sys
import time
import pygame

from env.maze_task import MazeTaskManager, MazeTaskSampler
from gymnasium.wrappers import FrameStackObservation


DISCRETE_ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Left, Right, Down, Up


def load_tasks(file_path):
    """
    Load tasks from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing tasks.

    Returns:
        list: A list of task configurations.
    """
    try:
        with open(file_path, "r") as f:
            tasks = json.load(f)
        return tasks
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from the file '{file_path}': {e}")
        sys.exit(1)


def main():
    # Initialize pygame
    pygame.init()
    pygame.display.set_mode((1, 1))  # Dummy display just to capture events

    # Paths to the tasks JSON files
    train_tasks_path = "mazes_data/train_tasks.json"
    test_small_tasks_path = "mazes_data/test_small_tasks.json"
    test_large_tasks_path = "mazes_data/test_large_tasks.json"

    # We choose which tasks to load:
    task_set = "train"

    # Load tasks from JSON
    if task_set == "train":
        tasks = load_tasks(train_tasks_path)
    elif task_set == "test_small":
        tasks = load_tasks(test_small_tasks_path)
    elif task_set == "test_large":
        tasks = load_tasks(test_large_tasks_path)
    else:
        raise ValueError("Invalid task_set specified. Choose from 'train', 'test_small', 'test_large'.")

    # Select a random task from the chosen set
    random_task_params = random.choice(tasks)
    print(f"Selected Task Parameters: {random_task_params}")

    # Use MazeTaskSampler to generate a TaskConfig
    try:
        task_config = MazeTaskSampler(**random_task_params)
    except TypeError as e:
        print(f"Error sampling TaskConfig: {e}")
        print("Available parameters in the task configuration:", list(random_task_params.keys()))
        print("Expected parameters: n, allow_loops, crowd_ratio, cell_size, wall_height, "
              "agent_height, step_reward, goal_reward, initial_life, max_life, food_density, food_interval")
        sys.exit(1)

    # Initialize the environment
    env_id = "MetaMazeDiscrete3D-v0"

    try:
        env = gym.make(env_id)
    except gym.error.UnregisteredEnv:
        print(f"Error: The environment '{env_id}' is not registered.")
        print("Check your environment registration in 'env/__init__.py'.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error while creating the environment '{env_id}': {e}")
        sys.exit(1)

    # Check the base observation space
    base_obs_shape = env.observation_space.shape
    print(f"Base observation space shape: {base_obs_shape}")

    # Wrap with FrameStack
    sequence_length = 4
    env = FrameStackObservation(env, stack_size=sequence_length)
    print(f"Environment wrapped with FrameStackObservation (sequence_length={sequence_length}).")

    # The shape we expect now
    expected_shape = (sequence_length,) + base_obs_shape
    print(f"Expected observation shape after stacking: {expected_shape}")

    # Set the task in the environment
    try:
        env.unwrapped.set_task(task_config)
        print(f"Goal Position: {task_config.goal}")  # Print the goal position
    except AttributeError:
        print("Error: The environment does not support the 'set_task' method.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error while setting the task: {e}")
        sys.exit(1)

    # Reset the environment
    try:
        observation, info = env.reset()
    except Exception as e:
        print(f"Error resetting the environment: {e}")
        sys.exit(1)

    # Attempt to render the initial state
    try:
        env.render()
    except Exception as e:
        print(f"Error rendering the environment: {e}")
        # Continue even if rendering fails

    # Initialize tracking variables
    phase_steps = {1: 0, 2: 0}
    phase_rewards = {1: 0.0, 2: 0.0}
    phase_collisions = {1: 0, 2: 0}
    phase_collision_penalties = {1: 0.0, 2: 0.0}
    goal_reached_phase1 = False
    goal_reached_phase2 = False  # [ADDED HERE, track if goal is reached in phase2]

    # We'll track the previous_phase for transitions
    previous_phase = None

    # We'll run a test loop with two-phase logic
    terminated = False
    truncated = False
    steps = 0
    max_steps = 500  # total steps across both phases

    print("\nStarting the two-phase episode. Use arrow keys to control the agent. Press ESC to exit.\n")

    while not terminated and not truncated and steps < max_steps:
        try:
            # Process any pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Render Window Closed by User.")
                    terminated = True
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("ESC pressed. Exiting the episode.")
                        terminated = True
                        break

            if terminated:
                break

            # Check keyboard input
            keys = pygame.key.get_pressed()
            action = None
            if keys[pygame.K_LEFT]:
                action = 0  # Left
            elif keys[pygame.K_RIGHT]:
                action = 1  # Right
            elif keys[pygame.K_UP]:
                action = 3  # Up
            elif keys[pygame.K_DOWN]:
                action = 2  # Down

            if action is not None:
                print(f"Keyboard Action: {DISCRETE_ACTIONS[action]}")

                # Step in the environment
                observation, reward, terminated, truncated, info = env.step(action)

                # Check which phase we're in
                current_phase = info.get("phase", None)
                if current_phase not in [1, 2]:
                    print(f"Warning: Unknown phase '{current_phase}' detected.")
                    current_phase = previous_phase if previous_phase else 1

                # If we just switched from phase1 to phase2 => implies agent reached the goal in phase1
                if previous_phase == 1 and current_phase == 2:
                    goal_reached_phase1 = True
                    print("Goal reached in Phase 1. Transitioning to Phase 2 and teleporting back to start.")

                # Update the previous_phase
                previous_phase = current_phase

                # Phase-specific counters
                phase_steps[current_phase] += 1
                phase_rewards[current_phase] += reward

                # Check collisions
                collision = info.get("collision", False)
                if collision:
                    collision_penalty = env.unwrapped.collision_penalty
                    phase_collisions[current_phase] += 1
                    phase_collision_penalties[current_phase] += abs(collision_penalty)
                    print(f"Collision Detected at Step {steps + 1}: Penalty Applied = {collision_penalty}")

                # Logging
                agent_grid = info.get("agent_grid", "Unknown")
                print(f"Step {steps + 1}: Reward = {reward:.3f}, Terminated = {terminated}, "
                      f"Truncated = {truncated}, Collision = {collision}, Phase = {current_phase}, "
                      f"Agent Position = {agent_grid}")

                # Verify the observation shape
                if observation.shape != expected_shape:
                    print(f"Error: Observation shape {observation.shape} does not match expected {expected_shape}.")
                    sys.exit(1)
                else:
                    print(f"Observation Shape Verified: {observation.shape}")

                # [ADDED HERE] If we see that the environment terminated in phase2,
                # check if "phase_reports" indicates a goal reward in phase2:
                if terminated and current_phase == 2:
                    phase_reports = info.get("phase_reports", {})
                    if "Phase2" in phase_reports:
                        if phase_reports["Phase2"].get("Goal Rewards", 0.0) > 0:
                            goal_reached_phase2 = True

                # Render
                env.render()

                steps += 1
                time.sleep(0.05)
            else:
                # If no key is pressed, just render
                env.render()
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\nEpisode interrupted by user.")
            break
        except Exception as e:
            print(f"Error during environment step: {e}")
            break

    print("\nTwo-phase episode finished.\n")

    # Summaries for each phase
    for phase in [1, 2]:
        print(f"--- Phase {phase} Report ---")
        print(f"Total Steps: {phase_steps[phase]}")
        print(f"Total Step Reward: {phase_rewards[phase]:.3f}")
        print(f"Total Collisions: {phase_collisions[phase]}")
        print(f"Total Collision Penalties: {phase_collision_penalties[phase]:.3f}")
        if phase == 1:
            print(f"Goal Reached in Phase 1: {'Yes' if goal_reached_phase1 else 'No'}")
        elif phase == 2:
            print(f"Goal Reached in Phase 2: {'Yes' if goal_reached_phase2 else 'No'}")
        print("--------------------------\n")

    print(f"Overall Steps Taken: {steps}")
    overall_penalty = phase_collision_penalties[1] + phase_collision_penalties[2]
    print(f"Overall Collision Penalties: {overall_penalty:.3f}")
    print(f"Overall Collisions: {phase_collisions[1] + phase_collisions[2]}")

    # Close environment gracefully
    try:
        env.close()
    except Exception as e:
        print(f"Error closing the environment: {e}")

    pygame.quit()


if __name__ == "__main__":
    main()
