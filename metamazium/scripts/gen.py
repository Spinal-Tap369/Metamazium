import os
import json
import numpy as np
from metamazium.env.maze_task import MazeTaskSampler

def convert_ndarray_to_list(obj):
    """
    Recursively convert any NumPy array in a dictionary or list to a list.
    """
    if isinstance(obj, dict):
        return {key: convert_ndarray_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def generate_tasks(n_tasks, size, allow_loops=False, mode="ESCAPE"):
    """
    Generate a list of maze task configurations by calling MazeTaskSampler.
    
    Parameters:
      - n_tasks: Number of tasks to generate.
      - size: The maze 'n' parameter (e.g., 7 for simple mazes).
      - allow_loops: Whether the maze allows loops.
      - mode: Typically "ESCAPE" or "SURVIVAL" (mode is set in the environment).
      
    Returns:
      A list of task dictionaries including the generated cell_walls.
    """
    tasks = []
    for _ in range(n_tasks):
        # Define the configuration parameters for the maze.
        config = {
            "n": size,
            "allow_loops": allow_loops,
            "crowd_ratio": 0.0,   # no extra walls (tree-like maze)
            "cell_size": 2.0,
            "wall_height": 3.2,
            "agent_height": 1.6,
            "step_reward": -0.01,
            "goal_reward": 1.0,
            "initial_life": 1.0,
            "max_life": 2.0,
            "food_density": 0.01,
            "food_interval": 100,
            # mode (ESCAPE or SURVIVAL) is typically set via the environment's task_type parameter.
        }
        # Call MazeTaskSampler with the configuration.
        task = MazeTaskSampler(**config)
        # Convert the namedtuple to a dictionary.
        task_dict = task._asdict()
        # Recursively convert any NumPy arrays to lists.
        task_dict = convert_ndarray_to_list(task_dict)
        tasks.append(task_dict)
    return tasks

def main():
    os.makedirs("mazes_data", exist_ok=True)
    
    # Generate, for example, 1000 maze tasks (all with n=7).
    tasks = generate_tasks(n_tasks=1000, size=7, allow_loops=False)
    # Save the tasks (which now include the cell_walls layout) to JSON.
    output_path = "mazes_data_try/train_tasks.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"Saved {len(tasks)} tasks with full maze layouts (including cell_walls) to {output_path}")

if __name__ == "__main__":
    main()
