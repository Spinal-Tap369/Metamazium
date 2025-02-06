# metamazium\env\__init__.py

from gymnasium.envs.registration import register

register(
    id="MetaMazeDiscrete3D-v0",
    entry_point="metamazium.env.maze_env:MetaMazeDiscrete3D",
    kwargs={
        "enable_render": True,
        "render_scale": 480,
        "resolution": (40, 30),
        "max_steps": 5000,
        "task_type": "ESCAPE",
        "phase_step_limit": 250,  # Define steps per phase
        "collision_penalty": -0.005  # Define collision penalty
    }
)

register(
    id="MetaMaze2D-v0",
    entry_point="metamazium.env.maze_env:MetaMaze2D",
    kwargs={
        "enable_render": True,
        "render_scale": 480,
        "max_steps": 5000,
        "task_type": "ESCAPE",
        "view_grid": 2
    }
)
