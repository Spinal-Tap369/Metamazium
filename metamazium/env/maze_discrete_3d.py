# env/maze_discrete3d.py

import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding
from copy import deepcopy

from metamazium.env.maze_base import MazeBase
from metamazium.env.ray_caster_utils import maze_view
from metamazium.env.maze_task import MAZE_TASK_MANAGER

# Define discrete actions: L R D U
DISCRETE_ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

class MazeCoreDiscrete3D(MazeBase):
    """
    Core logic for the discrete 3D maze environment.
    Handles agent movement, phase transitions, and reward calculations.
    """

    def __init__(
            self,
            collision_dist=0.20,          
            max_vision_range=12.0,        
            fol_angle=0.6 * np.pi,        
            resolution_horizon=320,       
            resolution_vertical=320,      
            max_steps=5000,              
            task_type="ESCAPE",          
            phase_step_limit=250,        
            collision_penalty=-0.001     
        ):
        super(MazeCoreDiscrete3D, self).__init__(
                collision_dist=collision_dist,
                max_vision_range=max_vision_range,
                fol_angle=fol_angle,
                resolution_horizon=resolution_horizon,
                resolution_vertical=resolution_vertical,
                task_type=task_type,
                max_steps=max_steps,
                phase_step_limit=phase_step_limit
        )
        # Orientation
        self._agent_ori_choice = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        self._agent_ori_index = 0
        self._agent_ori = self._agent_ori_choice[self._agent_ori_index]

        self.collision_penalty = collision_penalty  # negative collision penalty

        # Keep track of total reward in each phase (optional):
        self._phase_rewards = {1: 0.0, 2: 0.0}

        # Per-phase metrics:
        self.phase_metrics = {
            1: {"goal_rewards": 0.0, "steps": 0, "step_rewards": 0.0,
                "collisions": 0, "collision_rewards": 0.0},
            2: {"goal_rewards": 0.0, "steps": 0, "step_rewards": 0.0,
                "collisions": 0, "collision_rewards": 0.0}
        }

    def reset(self):
        observation = super(MazeCoreDiscrete3D, self).reset()

        # Store starting position for possible teleportation
        self._starting_position = deepcopy(self._agent_grid)

        self._phase_rewards = {1: 0.0, 2: 0.0}

        # Reset orientation
        self._agent_ori_index = 0
        self._agent_ori = self._agent_ori_choice[self._agent_ori_index]

        # Reset phase metrics each episode
        self.phase_metrics = {
            1: {"goal_rewards": 0.0, "steps": 0, "step_rewards": 0.0,
                "collisions": 0, "collision_rewards": 0.0},
            2: {"goal_rewards": 0.0, "steps": 0, "step_rewards": 0.0,
                "collisions": 0, "collision_rewards": 0.0}
        }
        self._gave_exploration_bonus = False
        return observation

    def do_action(self, action):
        """
        action is a tuple: (turn_dir, step)
        """
        assert isinstance(action, tuple) and len(action) == 2
        direction, step = action

        # Turn + Move
        self.turn(direction)
        old_grid = deepcopy(self._agent_grid)
        self.move(step)

        # Collision if agent_grid unchanged
        collision = np.array_equal(old_grid, self._agent_grid)

        # Increase phase step count
        self.phase_metrics[self.phase]["steps"] += 1
        self.current_phase_steps += 1

        # Evaluate reward
        reward, done = self.evaluation_rule()

        # If collision, apply penalty
        if collision:
            reward += self.collision_penalty
            self.phase_metrics[self.phase]["collisions"] += 1
            self.phase_metrics[self.phase]["collision_rewards"] += self.collision_penalty

        # Add to the per-phase total reward if desired
        self._phase_rewards[self.phase] += reward

        # Check if agent_at_goal
        agent_at_goal = (tuple(self._agent_grid) == self._goal)

        if self.task_type == "ESCAPE":
            if agent_at_goal:
                if self.phase == 1:
                    # Switch to phase 2
                    self.phase = 2
                    self.current_phase_steps = 0
                    # Teleport agent back to start
                    self._agent_grid = deepcopy(self._starting_position)
                    self._agent_loc = self.get_cell_center(self._agent_grid)
                    done = False  # continue episode
                    self.update_observation()
                    done = False
                elif self.phase == 2:
                    # Reached goal in phase 2 => done
                    done = True

        # Check phase step limit
        if self.phase == 1 and self.current_phase_steps >= self.phase_step_limit:
            self.phase = 2
            self.current_phase_steps = 0
            self._agent_grid = deepcopy(self._starting_position)
            self._agent_loc = self.get_cell_center(self._agent_grid)
            done = False
            self.update_observation()
        elif self.phase == 2 and self.current_phase_steps >= self.phase_step_limit:
            done = True

        # Finally update observation after movement
        self.update_observation()
        return reward, done, collision

    def evaluation_rule(self):
        """
        Tied to step-based reward & goal reward. Also track in phase_metrics.
        """
        self.steps += 1
        self._agent_trajectory.append(np.copy(self._agent_grid))

        # Check if agent at goal
        agent_at_goal = (tuple(self._goal) == tuple(self._agent_grid))

        if self.task_type == "ESCAPE":
            if self.phase == 1:
                # step_reward is negative => add to step_rewards
                self.phase_metrics[1]["step_rewards"] += self._step_reward
                reward = self._step_reward
                # add a  exploration bonus if agent first hits the goal in phase1
                if agent_at_goal and not self._gave_exploration_bonus:
                    bonus = 0.5
                    reward += bonus
                    self.phase_metrics[1]["goal_rewards"] += self._goal_reward  # goal reward
                    self._gave_exploration_bonus = True
                    # print("Exploration bonus of", bonus, "awarded!")
                done = False
            elif self.phase == 2:
                # In phase 2, if agent_at_goal => get goal reward
                # Also add negative step_reward
                step_r = self._step_reward
                self.phase_metrics[2]["step_rewards"] += step_r
                if agent_at_goal:
                    # agent gets goal reward
                    self.phase_metrics[2]["goal_rewards"] += self._goal_reward
                    reward = step_r + self._goal_reward
                else:
                    reward = step_r
                done = agent_at_goal or self.episode_is_over()
        else:
            # If SURVIVAL or other tasks, you'd do similarly
            reward = 0.0
            done = False

        return reward, done

    def turn(self, direction):
        self._agent_ori_index += direction
        self._agent_ori_index %= len(self._agent_ori_choice)
        self._agent_ori = self._agent_ori_choice[self._agent_ori_index]

    def move(self, step):
        tmp_grid = deepcopy(self._agent_grid)
        if self._agent_ori_index == 0:
            tmp_grid[0] += step
        elif self._agent_ori_index == 1:
            tmp_grid[1] += step
        elif self._agent_ori_index == 2:
            tmp_grid[0] -= step
        elif self._agent_ori_index == 3:
            tmp_grid[1] -= step

        # If not blocked, update agent_grid
        if (0 <= tmp_grid[0] < self._n and
            0 <= tmp_grid[1] < self._n and
            self._cell_walls[tmp_grid[0], tmp_grid[1]] == 0):
            self._agent_grid = tmp_grid
            self._agent_loc = self.get_cell_center(self._agent_grid)

    def render_init(self, view_size):
        super(MazeCoreDiscrete3D, self).render_init(view_size)
        self._pos_conversion = self._render_cell_size / self._cell_size
        self._ori_size = 0.60 * self._pos_conversion

    def render_observation(self):
        # Paint observation
        import pygame
        view_obs_surf = pygame.transform.scale(self._obs_surf, (self._view_size, self._view_size))
        self._screen.blit(view_obs_surf, (0, 0))

        # Paint top-down view
        agent_pos = np.array(self._agent_loc) * self._pos_conversion
        dx = self._ori_size * np.cos(self._agent_ori)
        dy = self._ori_size * np.sin(self._agent_ori)

        center_pos = (int(agent_pos[0] + self._view_size), int(self._view_size - agent_pos[1]))
        end_pos = (int(agent_pos[0] + self._view_size + dx), int(self._view_size - agent_pos[1] - dy))
        radius = int(0.15 * self._pos_conversion)

        pygame.draw.circle(self._screen, pygame.Color("green"), center_pos, radius)
        pygame.draw.line(self._screen, pygame.Color("green"), center_pos, end_pos, width=1)

    def movement_control(self, keys):
        if keys[pygame.K_LEFT]:
            return (-1, 0)
        if keys[pygame.K_RIGHT]:
            return (1, 0)
        if keys[pygame.K_UP]:
            return (0, 1)
        if keys[pygame.K_DOWN]:
            return (0, -1)
        return (0, 0)

    def update_observation(self):
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

        self._obs_surf = pygame.surfarray.make_surface(self._observation)
        self._observation = np.clip(self._observation, 0.0, 255.0).astype("float32")

    def get_observation(self):
        return np.copy(self._observation)
