# env\maze_discrete3d.py

import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding
from copy import deepcopy

from env.maze_base import MazeBase
from env.ray_caster_utils import maze_view
from env.maze_task import MAZE_TASK_MANAGER

# Define discrete actions: Left, Right, Down, Up
DISCRETE_ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Left, Right, Down, Up

class MazeCoreDiscrete3D(MazeBase):
    """
    Core logic for the discrete 3D maze environment.
    Handles agent movement, phase transitions, and reward calculations.
    """
    def __init__(
            self,
            collision_dist=0.20,          # Collision distance (not used in discrete)
            max_vision_range=12.0,        # Agent's vision range
            fol_angle=0.6 * np.pi,        # Field of View angle
            resolution_horizon=320,       # Horizontal resolution
            resolution_vertical=320,      # Vertical resolution
            max_steps=5000,               # Maximum steps per episode
            task_type="ESCAPE",           # Task type: "SURVIVAL" or "ESCAPE"
            phase_step_limit=250,         # Steps allocated per phase
            collision_penalty=-0.001      # Collision penalty
        ):
        super(MazeCoreDiscrete3D, self).__init__(
                collision_dist=collision_dist,
                max_vision_range=max_vision_range,
                fol_angle=fol_angle,
                resolution_horizon=resolution_horizon,
                resolution_vertical=resolution_vertical,
                task_type=task_type,
                max_steps=max_steps,
                phase_step_limit=phase_step_limit  # Pass to MazeBase
                )
        # Initialize tracking variables
        self._starting_position = None
        self._phase_rewards = {1: 0.0, 2: 0.0}

        # Initialize orientation attributes
        self._agent_ori_choice = [0, np.pi / 2, np.pi, 3 * np.pi / 2]  # East, North, West, South
        self._agent_ori_index = 0  # Start facing East
        self._agent_ori = self._agent_ori_choice[self._agent_ori_index]

        # Store collision penalty
        self.collision_penalty = collision_penalty

    def reset(self):
        """
        Resets the environment to the initial state.
        """
        # Call the base class reset method
        observation = super(MazeCoreDiscrete3D, self).reset()

        # Store the starting position for teleportation
        self._starting_position = deepcopy(self._agent_grid)

        # Reset phase rewards
        self._phase_rewards = {1: 0.0, 2: 0.0}

        # Reset agent orientation
        self._agent_ori_index = 0  # Reset to initial orientation
        self._agent_ori = self._agent_ori_choice[self._agent_ori_index]

        return observation

    def do_action(self, action):
        """
        Performs an action by the agent: turning and moving.

        Args:
            action (int): The action index (0: Left, 1: Right, 2: Down, 3: Up)

        Returns:
            tuple: (reward, done, collision)
        """
        assert isinstance(action, tuple) and len(action) == 2, "Action must be a tuple of (turn_direction, move_step)"
        assert abs(action[0]) < 2 and abs(action[1]) < 2, "Invalid action values"

        direction, step = action

        self.turn(direction)
        previous_grid = deepcopy(self._agent_grid)
        self.move(step)

        # Collision occurs if the agent did not move despite attempting to move
        collision = np.array_equal(previous_grid, self._agent_grid)

        # Increment phase step counter
        self.current_phase_steps += 1  # Increment the step count for the current phase

        # Perform evaluation to get reward and done flag
        reward, done = self.evaluation_rule()

        # Apply collision penalty
        if collision:
            reward += self.collision_penalty  # Note: collision_penalty is negative

        # Accumulate reward for the current phase
        self._phase_rewards[self.phase] += reward

        # Check if agent has reached the goal
        agent_at_goal = tuple(self._agent_grid) == self._goal

        if self.task_type == "ESCAPE":
            if agent_at_goal:
                if self.phase == 1:
                    # Transition to phase2
                    # Switch to phase2
                    self.phase = 2
                    self.current_phase_steps = 0

                    # Teleport agent back to starting position
                    self._agent_grid = deepcopy(self._starting_position)
                    self._agent_loc = self.get_cell_center(self._agent_grid)

                    # Update observation after teleportation
                    self.update_observation()
                    done = False  # Continue the episode

                elif self.phase == 2:
                    # Goal reached in phase2, terminate the episode
                    done = True

        # Check if phase_step_limit is reached
        if self.phase == 1 and self.current_phase_steps >= self.phase_step_limit:
            # Transition to phase2
            self.phase = 2
            self.current_phase_steps = 0

            # Teleport agent back to starting position
            self._agent_grid = deepcopy(self._starting_position)
            self._agent_loc = self.get_cell_center(self._agent_grid)

            # Update observation after teleportation
            self.update_observation()

        elif self.phase == 2 and self.current_phase_steps >= self.phase_step_limit:
            # Terminate the episode after phase2 step limit
            done = True

        # Update observation after the action
        self.update_observation()

        return reward, done, collision  # Return collision flag

    def turn(self, direction):
        """
        Turns the agent based on the direction.

        Args:
            direction (int): -1 for left, 1 for right
        """
        self._agent_ori_index += direction
        self._agent_ori_index = self._agent_ori_index % len(self._agent_ori_choice)
        self._agent_ori = self._agent_ori_choice[self._agent_ori_index]

    def move(self, step):
        """
        Moves the agent forward or backward based on the step.

        Args:
            step (int): -1 for backward, 1 for forward
        """
        tmp_grid = deepcopy(self._agent_grid)

        # Determine movement direction based on current orientation
        if self._agent_ori_index == 0:
            tmp_grid[0] += step  # Move East
        elif self._agent_ori_index == 1:
            tmp_grid[1] += step  # Move North
        elif self._agent_ori_index == 2:
            tmp_grid[0] -= step  # Move West
        elif self._agent_ori_index == 3:
            tmp_grid[1] -= step  # Move South
        else:
            raise ValueError(f"Unexpected agent orientation index: {self._agent_ori_index}")

        # Check for wall collisions and grid boundaries
        if (
            0 <= tmp_grid[0] < self._n
            and 0 <= tmp_grid[1] < self._n
            and self._cell_walls[tmp_grid[0], tmp_grid[1]] == 0
        ):
            self._agent_grid = tmp_grid
            self._agent_loc = self.get_cell_center(self._agent_grid)
        else:
            pass  # Move is blocked; agent stays in the same grid

    def render_init(self, view_size):
        """
        Initializes rendering parameters.
        """
        super(MazeCoreDiscrete3D, self).render_init(view_size)
        self._pos_conversion = self._render_cell_size / self._cell_size
        self._ori_size = 0.60 * self._pos_conversion

    def render_observation(self):
        """
        Renders the agent and its orientation on the screen.
        """
        # Paint Observation
        view_obs_surf = pygame.transform.scale(self._obs_surf, (self._view_size, self._view_size))
        self._screen.blit(view_obs_surf, (0, 0))

        # Paint God-view (Top-down view)
        agent_pos = np.array(self._agent_loc) * self._pos_conversion
        dx = self._ori_size * np.cos(self._agent_ori)
        dy = self._ori_size * np.sin(self._agent_ori)
        
        # Convert coordinates to integers for rendering
        center_pos = (int(agent_pos[0] + self._view_size), int(self._view_size - agent_pos[1]))
        end_pos = (int(agent_pos[0] + self._view_size + dx), int(self._view_size - agent_pos[1] - dy))
        radius = int(0.15 * self._pos_conversion)

        pygame.draw.circle(self._screen, pygame.Color("green"), center_pos, radius)
        pygame.draw.line(self._screen, pygame.Color("green"), center_pos, end_pos, width=1)

    def movement_control(self, keys):
        """
        Handles keyboard inputs for agent movement.

        Args:
            keys (pygame.key.ScancodeWrapper): Current state of keyboard keys.

        Returns:
            tuple: (turn_direction, move_step)
        """
        if keys[pygame.K_LEFT]:
            return (-1, 0)  # Turn left
        if keys[pygame.K_RIGHT]:
            return (1, 0)   # Turn right
        if keys[pygame.K_UP]:
            return (0, 1)   # Move forward
        if keys[pygame.K_DOWN]:
            return (0, -1)  # Move backward
        return (0, 0)       # No action

    def update_observation(self):
        """
        Updates the current observation based on the agent's position.
        This method uses maze_view to generate the observation.
        """
        self._observation = maze_view(
            self._agent_loc,
            self._agent_ori,
            self._agent_height,
            self._cell_walls,
            self._cell_transparents,
            self._cell_texts,
            self._cell_size,
            MAZE_TASK_MANAGER.grounds,
            MAZE_TASK_MANAGER.ceil,
            self._wall_height,
            1.0,
            self.max_vision_range,
            0.20,
            self.fol_angle,
            self.resolution_horizon,
            self.resolution_vertical
        )

        if self.task_type == "SURVIVAL":
            lifebar_l = self._life / self._max_life * self._lifebar_l
            start_x = int(self._lifebar_start_x)
            start_y = int(self._lifebar_start_y)
            end_x = int(self._lifebar_start_x + lifebar_l)
            end_y = int(self._lifebar_start_y + self._lifebar_w)
            self._observation[start_x:end_x, start_y:end_y, 0] = 255  # Red channel
            self._observation[start_x:end_x, start_y:end_y, 1] = 0    # Green channel
            self._observation[start_x:end_x, start_y:end_y, 2] = 0    # Blue channel
        self._obs_surf = pygame.surfarray.make_surface(self._observation)
        self._observation = np.clip(self._observation, 0.0, 255.0).astype("float32")

    def get_observation(self):
        """
        Returns the current observation.

        Returns:
            numpy.ndarray: The current observation.
        """
        return np.copy(self._observation)
