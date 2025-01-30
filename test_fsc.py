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

def main():
    pygame.init()
    pygame.display.set_mode((1, 1))  

    train_tasks_path = "mazes_data_try/train_tasks.json"
    task_set = "train"
    
    if task_set == "train":
        tasks = load_tasks(train_tasks_path)
    else:
        raise ValueError("Invalid task_set specified.")

    random_task_params = random.choice(tasks)
    print(f"Selected Task Parameters: {random_task_params}")

    try:
        task_config = MazeTaskSampler(**random_task_params)
    except TypeError as e:
        print(f"Error sampling TaskConfig: {e}")
        sys.exit(1)

    env_id = "MetaMazeDiscrete3D-v0"
    
    try:
        env = gym.make(env_id)
    except gym.error.UnregisteredEnv:
        print(f"Error: Environment '{env_id}' not registered.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error creating environment '{env_id}': {e}")
        sys.exit(1)

    base_obs_shape = env.observation_space.shape
    sequence_length = 4
    env = FrameStackObservation(env, stack_size=sequence_length)
    
    expected_shape = (sequence_length,) + base_obs_shape

    try:
        env.unwrapped.set_task(task_config)
        print(f"Goal Position: {task_config.goal}")
    except AttributeError:
        print("Error: Environment does not support 'set_task'.")
        sys.exit(1)

    try:
        observation, info = env.reset()
    except Exception as e:
        print(f"Error resetting environment: {e}")
        sys.exit(1)

    try:
        env.render()
    except Exception as e:
        print(f"Error rendering environment: {e}")

    phase_steps = {1: 0, 2: 0}
    phase_rewards = {1: 0.0, 2: 0.0}
    phase_collisions = {1: 0, 2: 0}
    phase_collision_penalties = {1: 0.0, 2: 0.0}
    goal_reached_phase1 = False
    goal_reached_phase2 = False  

    previous_phase = None
    terminated, truncated = False, False
    steps, max_steps = 0, 500  

    print("\nStarting the two-phase episode. Use arrow keys to control the agent. Press ESC to exit.\n")

    while not terminated and not truncated and steps < max_steps:
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Render Window Closed by User.")
                    terminated = True
                    break
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    print("ESC pressed. Exiting.")
                    terminated = True
                    break

            if terminated:
                break

            keys = pygame.key.get_pressed()
            action = None
            if keys[pygame.K_LEFT]:
                action = 0  
            elif keys[pygame.K_RIGHT]:
                action = 1  
            elif keys[pygame.K_UP]:
                action = 3  
            elif keys[pygame.K_DOWN]:
                action = 2  

            if action is not None:
                print(f"Action: {DISCRETE_ACTIONS[action]}")

                observation, reward, terminated, truncated, info = env.step(action)

                current_phase = info.get("phase", None) or previous_phase or 1

                if previous_phase == 1 and current_phase == 2:
                    goal_reached_phase1 = True
                    print("Goal reached in Phase 1. Transitioning to Phase 2.")

                previous_phase = current_phase

                phase_steps[current_phase] += 1
                phase_rewards[current_phase] += reward

                if info.get("collision", False):
                    collision_penalty = env.unwrapped.collision_penalty
                    phase_collisions[current_phase] += 1
                    phase_collision_penalties[current_phase] += abs(collision_penalty)
                    print(f"Collision at Step {steps + 1}: Penalty = {collision_penalty}")

                agent_grid = info.get("agent_grid", "Unknown")
                print(f"Step {steps + 1}: Reward = {reward:.3f}, Terminated = {terminated}, "
                      f"Truncated = {truncated}, Collision = {info.get('collision', False)}, "
                      f"Phase = {current_phase}, Agent Position = {agent_grid}")

                if observation.shape != expected_shape:
                    print(f"Error: Observation shape {observation.shape} mismatch.")
                    sys.exit(1)

                if terminated and current_phase == 2:
                    phase_reports = info.get("phase_reports", {})
                    if phase_reports.get("Phase2", {}).get("Goal Rewards", 0.0) > 0:
                        goal_reached_phase2 = True

                env.render()
                steps += 1
                time.sleep(0.05)
            else:
                env.render()
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
        except Exception as e:
            print(f"Error during step: {e}")
            break

    print("\nTwo-phase episode finished.\n")

    for phase in [1, 2]:
        print(f"--- Phase {phase} Report ---")
        print(f"Steps: {phase_steps[phase]}")
        print(f"Step Reward: {phase_rewards[phase]:.3f}")
        print(f"Collisions: {phase_collisions[phase]}")
        print(f"Collision Penalties: {phase_collision_penalties[phase]:.3f}")
        if phase == 1:
            print(f"Goal Reached: {'Yes' if goal_reached_phase1 else 'No'}")
        else:
            print(f"Goal Reached: {'Yes' if goal_reached_phase2 else 'No'}")
        print("--------------------------\n")

    print(f"Total Steps: {steps}")
    print(f"Total Collision Penalties: {sum(phase_collision_penalties.values()):.3f}")
    print(f"Total Collisions: {sum(phase_collisions.values())}")

    try:
        env.close()
    except Exception as e:
        print(f"Error closing environment: {e}")

    pygame.quit()

if __name__ == "__main__":
    main()
