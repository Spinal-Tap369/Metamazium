# generate_mazes.py

import os
import json
import random

from metamazium.env.maze_task import MazeTaskSampler

def generate_tasks(n_tasks, size, allow_loops=False, mode="ESCAPE"):
    """
    Generate a list of random tasks (dict configs) using MazeTaskSampler.
    `size` is the `n` param for the maze (e.g. 15 for smaller, 25 for larger).
    `allow_loops` controls loops in the maze.
    `mode` typically "ESCAPE" or "SURVIVAL".
    """
    tasks = []
    for _ in range(n_tasks):
        # fix crowd_ratio=0 for a tree-like maze, set it > 0 for more walls
        config = {
            "n": size,
            "allow_loops": allow_loops,
            "crowd_ratio": 0.0,  # no extra walls
            "cell_size": 2.0,
            "wall_height": 3.2,
            "agent_height": 1.6,     
            "step_reward": -0.01,
            "goal_reward": 1.0,
            "initial_life": 1.0,
            "max_life": 2.0,
            "food_density": 0.01,
            "food_interval": 100,
            # mode is set via environment's `task_type` param (ESCAPE or SURVIVAL),
        }
        tasks.append(config)
    return tasks

def main():
    os.makedirs("mazes_data", exist_ok=True)
    
    # 1) Train: 1500 small mazes 
    train_tasks = generate_tasks(n_tasks=1000, size=7)
    with open("mazes_data/train_tasks.json", "w") as f:
        json.dump(train_tasks, f, indent=2)
    print("Saved 1500 small train tasks -> mazes_data/train_tasks.json")

    # 2) Test (same-size as train)
    test_small = generate_tasks(n_tasks=250, size=7)
    with open("mazes_data/test_small_tasks.json", "w") as f:
        json.dump(test_small, f, indent=2)
    print("Saved 250 small test tasks -> mazes_data/test_small_tasks.json")

    # 3) Test (larger-size)
    test_large = generate_tasks(n_tasks=250, size=10)
    with open("mazes_data/test_large_tasks.json", "w") as f:
        json.dump(test_large, f, indent=2)
    print("Saved 250 large test tasks -> mazes_data/test_large_tasks.json")

if __name__ == "__main__":
    main()
