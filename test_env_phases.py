# test_env_phases.py

import gymnasium as gym
import json
import random
import sys
import time  


from env.maze_task import MazeTaskManager, MazeTaskSampler

class RandomAgent:
    """
    A simple agent that selects actions randomly from the environment's action space.
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self):
        return self.action_space.sample()

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
    train_tasks_path = "mazes_data/train_tasks.json"
    test_small_tasks_path = "mazes_data/test_small_tasks.json"
    test_large_tasks_path = "mazes_data/test_large_tasks.json"
    
   
    task_set = "train" 
    
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
    
    # Use MazeTaskSampler to generate a TaskConfig from the parameters
    try:
        task_config = MazeTaskSampler(**random_task_params)
    except TypeError as e:
        print(f"Error sampling TaskConfig: {e}")
        print("Available parameters in the task configuration:", list(random_task_params.keys()))
        # List expected parameters for clarity
        print("Expected parameters: n, allow_loops, crowd_ratio, cell_size, wall_height, agent_height, "
              "step_reward, goal_reward, initial_life, max_life, food_density, food_interval")
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
    
    # Set the task
    try:
        env.unwrapped.set_task(task_config)
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
    
    # Initialize the Random Agent
    agent = RandomAgent(env.action_space)
    
    # Render the initial state
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
    goal_reached_phase1 = False  # To track if the goal was reached in Phase 1
    goal_reached_phase2 = False  # To track if the goal was reached in Phase 2
    
    # Initialize previous_phase to None
    previous_phase = None
    
    # Run a test loop
    terminated = False
    truncated = False
    steps = 0
    max_steps = 500  # Total steps for two phases (e.g., 250 + 250)
    
    print("\nStarting the two-phase episode...\n")
    
    while not terminated and not truncated and steps < max_steps:
        try:
            # Agent selects a random action
            action = agent.select_action()

            # Take a step in the environment
            observation, reward, terminated, truncated, info = env.step(action)

            # Extract current phase information
            current_phase = info.get("phase", None)
            if current_phase not in [1, 2]:
                print(f"Warning: Unknown phase '{current_phase}' detected.")
                current_phase = previous_phase if previous_phase else 1  # Default to previous or Phase 1

            # Detect phase transition from Phase 1 to Phase 2
            if previous_phase == 1 and current_phase == 2:
                goal_reached_phase1 = True
                print("Goal reached in Phase 1. Transitioning to Phase 2 and teleporting back to start.")
            
            # Update previous_phase for the next iteration
            previous_phase = current_phase

            # Update phase-specific counters
            phase_steps[current_phase] += 1
            phase_rewards[current_phase] += reward

            # Check for collision and apply collision penalty
            collision = info.get("collision", False)
            if collision:
                # Access collision_penalty via env.unwrapped
                collision_penalty = env.unwrapped.collision_penalty
                phase_collisions[current_phase] += 1
                phase_collision_penalties[current_phase] += abs(collision_penalty)
                print(f"Collision Detected at Step {steps + 1}: Penalty Applied = {collision_penalty}")
            
            # Detect if goal was reached in Phase 2
            if terminated and current_phase == 2:
                # Assuming that termination in Phase 2 implies goal achievement
                goal_reached_phase2 = True

            # Log the step information, including agent's position
            agent_grid = info.get("agent_grid", "Unknown")
            print(f"Step {steps + 1}: Reward = {reward:.3f}, Terminated = {terminated}, "
                  f"Truncated = {truncated}, Collision = {collision}, Phase = {current_phase}, "
                  f"Agent Position = {agent_grid}")
            
            # Render the environment
            env.render()

            # Increment step count after taking the step
            steps += 1

            # Optional: Control rendering frequency to prevent high CPU usage
            time.sleep(0.05)  # Sleep for 50 milliseconds (~20 FPS)
        except KeyboardInterrupt:
            print("\nEpisode interrupted by user.")
            break
        except Exception as e:
            print(f"Error during environment step: {e}")
            break

    print("\nTwo-phase episode finished.\n")

    # Generate reports for each phase
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
    print(f"Overall Collision Penalties: {phase_collision_penalties[1] + phase_collision_penalties[2]:.3f}")
    print(f"Overall Collisions: {phase_collisions[1] + phase_collisions[2]}")

  
    try:
        env.close()
    except Exception as e:
        print(f"Error closing the environment: {e}")

if __name__ == "__main__":
    main()
