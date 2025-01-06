# env/maze_env.py

import numpy
import gymnasium as gym  # Changed from gym to gymnasium
import pygame

from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
from env.maze_2d import MazeCore2D
from env.maze_continuous_3d import MazeCoreContinuous3D
from env.maze_discrete_3d import MazeCoreDiscrete3D  

DISCRETE_ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Left, Right, Down, Up

class MetaMazeDiscrete3D(gym.Env):
    def __init__(self, 
            enable_render=True,
            render_scale=480,
            resolution=(320, 320),
            max_steps=5000,
            task_type="SURVIVAL",
            collision_penalty=-0.001,  # New parameter
            phase_step_limit=250  
            ):

        super(MetaMazeDiscrete3D, self).__init__()

        self.enable_render = enable_render
        self.render_viewsize = render_scale
        self.collision_penalty = collision_penalty  # Store collision penalty
        self.maze_core = MazeCoreDiscrete3D(
                resolution_horizon=resolution[0],
                resolution_vertical=resolution[1],
                max_steps=max_steps,
                task_type=task_type,
                phase_step_limit=phase_step_limit,    # Pass phase_step_limit
                collision_penalty=collision_penalty   # Pass collision_penalty
                )

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # Left, Right, Down, Up

        # Updated observation_space to match the actual observation shape and dtype
        self.observation_space = spaces.Box(
            low=0.0, 
            high=255.0, 
            shape=(resolution[0], resolution[1], 3), 
            dtype=numpy.float32
        )

        self.need_reset = True
        self.need_set_task = True

        # Initialize RNG
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_task(self, task_config):
        self.maze_core.set_task(task_config)
        self.need_set_task = False
        print("Task has been set successfully.")

    def reset(self, seed=None, options=None):
        """
        Reset the environment and return the initial observation and info.

        Returns:
            observation (numpy.ndarray): The initial observation of the environment.
            info (dict): Additional information about the reset.
        """
        if self.need_set_task:
            raise Exception('Must call "set_task" before reset')
        
        # Reset the maze core
        observation = self.maze_core.reset()
        info = {}
        
        if self.enable_render:
            self.maze_core.render_init(self.render_viewsize)
            self.keyboard_press = pygame.key.get_pressed()
            print("Environment has been reset and rendering initialized.")
        
        self.need_reset = False
        self.key_done = False
        return observation, info  # Return a tuple as per Gymnasium's API

    def step(self, action=None):
        """
        Take an action in the environment.

        Args:
            action (int or tuple): The action to take. If None, uses keyboard control.

        Returns:
            observation (numpy.ndarray): The next observation.
            reward (float): The reward received.
            terminated (bool): Whether the episode has terminated.
            truncated (bool): Whether the episode was truncated.
            info (dict): Additional information about the step.
        """
        if self.need_reset:
            raise Exception('Must "reset" before doing any actions')
        
        if action is None:  # Keyboard control
            pygame.time.delay(100)  # 10 FPS
            action = self.maze_core.movement_control(self.keyboard_press)
            print(f"Keyboard Action: {action}")
        else:
            action = DISCRETE_ACTIONS[action]
            print(f"Discrete Action: {action}")
            
        try:
            # Take a step in the environment
            reward, done, collision = self.maze_core.do_action(action)

            # Collision penalty is already handled in do_action
            # No need to apply it again here

            # Update termination flags based on phase transitions
            if done:
                self.need_reset = True
                terminated = True
                truncated = False
                print("Episode Terminated.")

                # Gather per-phase reports
                phase1 = self.maze_core.phase_metrics[1]
                phase2 = self.maze_core.phase_metrics[2]
                reports = {
                    "Phase1": {
                        "Goal Rewards": phase1["goal_rewards"],
                        "Total Steps": phase1["steps"],
                        "Total Step Rewards": phase1["step_rewards"],
                        "Total Collisions": phase1["collisions"],
                        "Total Collision Rewards": phase1["collision_rewards"]
                    },
                    "Phase2": {
                        "Goal Rewards": phase2["goal_rewards"],
                        "Total Steps": phase2["steps"],
                        "Total Step Rewards": phase2["step_rewards"],
                        "Total Collisions": phase2["collisions"],
                        "Total Collision Rewards": phase2["collision_rewards"]
                    }
                }
            else:
                terminated = False
                truncated = False  # Modify based on your logic
                reports = {}

            # Add 'phase' and 'agent_grid' to info
            info = {
                "steps": self.maze_core.steps, 
                "collision": collision, 
                "phase": self.maze_core.phase,
                "agent_grid": self.maze_core._agent_grid  # Added agent position
            }

            # Include reports if episode is done
            if done:
                info["phase_reports"] = reports

            return self.maze_core.get_observation(), reward, terminated, truncated, info  # Return five values
        except Exception as e:
            print(f"Error during environment step: {e}")
            raise  # Re-raise the exception to be caught in the testing script

    def render(self, mode="human"):
        """
        Render the environment.

        Args:
            mode (str): The mode to render with. Only "human" is supported.
        """
        if mode != "human":
            raise NotImplementedError("Only 'human' mode is supported")
        done, keys = self.maze_core.render_update()
        if done:
            print("Render Window Closed by User.")

    def save_trajectory(self, file_name):
        """
        Save the trajectory of the agent.

        Args:
            file_name (str): The file path to save the trajectory image.
        """
        self.maze_core.render_trajectory(file_name)
        print(f"Trajectory saved to {file_name}.")

class MetaMazeContinuous3D(gym.Env):
    def __init__(self, 
            enable_render=True,
            render_scale=480,
            resolution=(320, 320),
            max_steps=5000,
            task_type="SURVIVAL",
            collision_penalty=-0.001  # New parameter
            ):

        super(MetaMazeContinuous3D, self).__init__()

        self.enable_render = enable_render
        self.render_viewsize = render_scale
        self.collision_penalty = collision_penalty  # Store collision penalty
        self.maze_core = MazeCoreContinuous3D(
                resolution_horizon=resolution[0],
                resolution_vertical=resolution[1],
                max_steps=max_steps,
                task_type=task_type
                )

        # Define action and observation spaces
        # For continuous actions: turn_rate and walk_speed
        self.action_space = spaces.Box(
            low=numpy.array([-1.0, -1.0], dtype=numpy.float32), 
            high=numpy.array([1.0, 1.0], dtype=numpy.float32), 
            dtype=numpy.float32
        )
        self.observation_space = spaces.Box(
            low=0.0, 
            high=255.0, 
            shape=(resolution[0], resolution[1], 3), 
            dtype=numpy.float32
        )

        self.need_reset = True
        self.need_set_task = True

        # Initialize RNG
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_task(self, task_config):
        self.maze_core.set_task(task_config)
        self.need_set_task = False
        print("Task has been set successfully.")

    def reset(self, seed=None, options=None):
        """
        Reset the environment and return the initial observation and info.

        Returns:
            observation (numpy.ndarray): The initial observation of the environment.
            info (dict): Additional information about the reset.
        """
        if self.need_set_task:
            raise Exception('Must call "set_task" before reset')
        
        # Reset the maze core
        observation = self.maze_core.reset()
        info = {}
        
        if self.enable_render:
            self.maze_core.render_init(self.render_viewsize)
            self.keyboard_press = pygame.key.get_pressed()
            print("Environment has been reset and rendering initialized.")
        
        self.need_reset = False
        self.key_done = False
        return observation, info  # Return a tuple as per Gymnasium's API

    def step(self, action=None):
        """
        Take an action in the environment.

        Args:
            action (numpy.ndarray or tuple): The action to take. If None, uses keyboard control.

        Returns:
            observation (numpy.ndarray): The next observation.
            reward (float): The reward received.
            terminated (bool): Whether the episode has terminated.
            truncated (bool): Whether the episode was truncated.
            info (dict): Additional information about the step.
        """
        if self.need_reset:
            raise Exception('Must "reset" before doing any actions')
        
        if action is None:  # Keyboard control
            pygame.time.delay(20)  # 50 FPS
            turn_rate, walk_speed = self.maze_core.movement_control(self.keyboard_press)
            print(f"Keyboard Action: Turn Rate = {turn_rate}, Walk Speed = {walk_speed}")
        else:
            turn_rate, walk_speed = action
            print(f"Continuous Action: Turn Rate = {turn_rate}, Walk Speed = {walk_speed}")
            
        try:
            # Take a step in the environment
            reward, done, collision = self.maze_core.do_action(turn_rate, walk_speed)
    
            # Collision penalty is already handled in do_action
            # No need to apply it again here
    
            # Update termination flags based on phase transitions
            if done:
                self.need_reset = True
                terminated = True
                truncated = False
                print("Episode Terminated.")

                # Gather per-phase reports
                phase1 = self.maze_core.phase_metrics[1]
                phase2 = self.maze_core.phase_metrics[2]
                reports = {
                    "Phase1": {
                        "Goal Rewards": phase1["goal_rewards"],
                        "Total Steps": phase1["steps"],
                        "Total Step Rewards": phase1["step_rewards"],
                        "Total Collisions": phase1["collisions"],
                        "Total Collision Rewards": phase1["collision_rewards"]
                    },
                    "Phase2": {
                        "Goal Rewards": phase2["goal_rewards"],
                        "Total Steps": phase2["steps"],
                        "Total Step Rewards": phase2["step_rewards"],
                        "Total Collisions": phase2["collisions"],
                        "Total Collision Rewards": phase2["collision_rewards"]
                    }
                }
            else:
                terminated = False
                truncated = False  # Modify based on your logic
                reports = {}

            # Add 'phase' and 'agent_grid' to info
            info = {
                "steps": self.maze_core.steps, 
                "collision": collision, 
                "phase": self.maze_core.phase,
                "agent_grid": self.maze_core._agent_grid  # Added agent position
            }

            # Include reports if episode is done
            if done:
                info["phase_reports"] = reports

            return self.maze_core.get_observation(), reward, terminated, truncated, info  # Return five values
        except Exception as e:
            print(f"Error during environment step: {e}")
            raise  # Re-raise the exception to be caught in the testing script

    def render(self, mode="human"):
        """
        Render the environment.

        Args:
            mode (str): The mode to render with. Only "human" is supported.
        """
        if mode != "human":
            raise NotImplementedError("Only 'human' mode is supported")
        done, keys = self.maze_core.render_update()
        if done:
            print("Render Window Closed by User.")

    def save_trajectory(self, file_name):
        """
        Save the trajectory of the agent.

        Args:
            file_name (str): The file path to save the trajectory image.
        """
        self.maze_core.render_trajectory(file_name)
        print(f"Trajectory saved to {file_name}.")

