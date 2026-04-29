"""
dynamic_obstacles.py — DynamicObstaclesEnvWrapper from LICORICE.

Uses concept_version=3 (13 concepts for 5x5 grid):
  agent_position_x, agent_position_y, agent_direction,
  obstacle1_position_x, obstacle1_position_y,
  obstacle2_position_x, obstacle2_position_y,
  movable_right, movable_down, movable_left, movable_up,
  obstacle1_move_direction, obstacle2_move_direction

obstacle move_direction encoding (5 classes):
  0=stayed, 1=right, 2=down, 3=left, 4=up

Observation: RGB image [3*n_stack, ROWS, COLS].
  n_stack=1 (default) → [3, ROWS, COLS]  (single frame, no stacking)
  n_stack=4           → [12, ROWS, COLS] (4-frame stack for stacked temporal encoding)

Temporal concepts:
  - obstacle1/2_move_direction: requires comparing two frames — invisible from single frame
  - agent_direction, movable flags: change each step but visible from single frame
Static concepts: positions (visible from single frame).
"""

from collections import deque
from typing import Optional

import gymnasium as gym
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None

try:
    from minigrid.core.constants import DIR_TO_VEC
    _MINIGRID_AVAILABLE = True
except ImportError:
    _MINIGRID_AVAILABLE = False
    DIR_TO_VEC = None


class DynamicObstaclesEnvWrapper(gym.Wrapper):
    """
    Wraps a MiniGrid DynamicObstacles environment.
    Provides pixel observations + concept annotations.
    """

    def __init__(
        self,
        env: gym.Env,
        ROWS: int = 160,
        COLS: int = 160,
        concept_version: int = 3,
        n_stack: int = 1,
        grid_size: int = 8,
    ):
        super().__init__(env)
        if cv2 is None:
            raise ImportError(
                "DynamicObstaclesEnvWrapper requires OpenCV (cv2). Install opencv-python in this environment."
            )
        assert _MINIGRID_AVAILABLE, "minigrid package required for DynamicObstaclesEnvWrapper"
        self.ROWS = ROWS
        self.COLS = COLS
        self.concept_version = concept_version
        self.n_stack = n_stack
        self.grid_size = grid_size

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3 * n_stack, ROWS, COLS), dtype=np.uint8
        )
        if n_stack > 1:
            self._frames: Optional[deque] = deque(maxlen=n_stack)
        else:
            self._frames = None

        # Only concept_version==3 supported here (13 concepts)
        # Position classes scale with grid_size (coordinates 0..grid_size-1)
        assert concept_version == 3, "Only concept_version=3 supported in this wrapper"
        pos_classes = grid_size  # positions 0..grid_size-1
        self.task_types = ["classification"] * 13
        self.num_classes = [
            pos_classes, pos_classes, 4,            # agent x, y, dir
            pos_classes, pos_classes,               # obstacle1 x, y
            pos_classes, pos_classes,               # obstacle2 x, y
            2, 2, 2, 2,                             # movable r/d/l/u
            5, 5,                                   # obstacle1/2 move direction
        ]
        self.concept_names = [
            "agent_position_x", "agent_position_y", "agent_direction",
            "obstacle1_position_x", "obstacle1_position_y",
            "obstacle2_position_x", "obstacle2_position_y",
            "movable_right", "movable_down", "movable_left", "movable_up",
            "obstacle1_move_direction", "obstacle2_move_direction",
        ]
        # Truly temporal (invisible from single frame): obstacle move directions
        # Also temporal but partially visible: direction, movable flags
        self.temporal_concepts = [2, 7, 8, 9, 10, 11, 12]

        self.current_concept: Optional[np.ndarray] = None
        self._prev_obstacle_positions: Optional[list] = None
        self._current_obstacle_velocities: list = [(0, 0), (0, 0)]

    # ------------------------------------------------------------------

    def get_concept(self) -> np.ndarray:
        return self.current_concept.copy() if self.current_concept is not None else np.zeros(13, dtype=np.float32)

    @property
    def concept_reward_active(self) -> float:
        """1.0 only when at least one obstacle moved this step.
        Keeps the concept AC reward focused on temporal concept prediction
        rather than trivial static concept lookup or 'stayed' predictions."""
        v1, v2 = self._current_obstacle_velocities
        return 1.0 if (v1 != (0, 0) or v2 != (0, 0)) else 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = self._get_image()
        if self.n_stack > 1:
            self._frames.clear()
            for _ in range(self.n_stack):
                self._frames.append(frame)
            stacked = np.concatenate(list(self._frames), axis=0)
        else:
            stacked = frame
        # Initialise velocity tracking — obstacles haven't moved yet
        self._prev_obstacle_positions = self._get_obstacle_positions()
        self._current_obstacle_velocities = [(0, 0), (0, 0)]
        self.current_concept = self._compute_concept()
        info["concept"] = self.current_concept.copy()
        return stacked, info

    def step(self, action):
        # Record obstacle positions BEFORE the step
        prev_positions = self._get_obstacle_positions()
        obs, reward, done, truncated, info = self.env.step(action)
        frame = self._get_image()
        if self.n_stack > 1:
            self._frames.append(frame)
            stacked = np.concatenate(list(self._frames), axis=0)
        else:
            stacked = frame
        # Compute obstacle velocities from position delta
        curr_positions = self._get_obstacle_positions()
        self._current_obstacle_velocities = [
            (curr_positions[i][0] - prev_positions[i][0],
             curr_positions[i][1] - prev_positions[i][1])
            for i in range(2)
        ]
        self.current_concept = self._compute_concept()
        if done or truncated:
            info["terminal_observation"] = stacked
        info["concept"] = self.current_concept.copy()
        return stacked, reward, done, truncated, info

    # ------------------------------------------------------------------

    def _get_obstacle_positions(self) -> list:
        """Returns sorted list of (x, y) for up to 2 obstacles, padded with (0,0)."""
        unwrapped = self.env.unwrapped
        grid = unwrapped.grid
        positions = []
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell and cell.type == "ball":
                    positions.append((x, y))
        positions = sorted(positions)
        while len(positions) < 2:
            positions.append((0, 0))
        return positions

    @staticmethod
    def _vel_to_direction(dx: int, dy: int) -> int:
        """Map (dx, dy) delta to direction class: 0=stayed, 1=right, 2=down, 3=left, 4=up."""
        if dx == 1:  return 1
        if dy == 1:  return 2
        if dx == -1: return 3
        if dy == -1: return 4
        return 0  # stayed or unexpected

    def _get_image(self) -> np.ndarray:
        img = self.env.render()
        img = cv2.resize(img, (self.COLS, self.ROWS))
        return img.transpose(2, 0, 1)  # [3, H, W]

    def _compute_concept(self) -> np.ndarray:
        unwrapped = self.env.unwrapped
        agent_pos = unwrapped.agent_pos
        agent_dir = unwrapped.agent_dir
        grid      = unwrapped.grid

        def get_cell(x, y):
            if x < 0 or x >= grid.width or y < 0 or y >= grid.height:
                return None
            return grid.get(x, y)

        def can_move(pos, direction):
            dx, dy = DIR_TO_VEC[direction]
            nx, ny = pos[0] + dx, pos[1] + dy
            if 1 <= nx < grid.width - 1 and 1 <= ny < grid.height - 1:
                cell = get_cell(nx, ny)
                return cell is None or (hasattr(cell, "can_overlap") and cell.can_overlap())
            return False

        obstacle_positions = self._get_obstacle_positions()

        movable = [
            int(can_move(agent_pos, 0)),  # right
            int(can_move(agent_pos, 1)),  # down
            int(can_move(agent_pos, 2)),  # left
            int(can_move(agent_pos, 3)),  # up
        ]

        # Obstacle velocity directions (0=stayed,1=right,2=down,3=left,4=up)
        obs_dirs = [
            self._vel_to_direction(*self._current_obstacle_velocities[i])
            for i in range(2)
        ]

        values = [
            agent_pos[0], agent_pos[1], agent_dir,
            obstacle_positions[0][0], obstacle_positions[0][1],
            obstacle_positions[1][0], obstacle_positions[1][1],
        ] + movable + obs_dirs

        return np.array(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_dynamic_obstacles_env(n_envs: int = 4, seed: int = 0, n_stack: int = 1, grid_size: int = 8) -> gym.Env:
    """
    Returns a vectorised DynamicObstaclesEnvWrapper (grid_size x grid_size, 2 obstacles).
    Requires minigrid and the CustomDynamicObstaclesEnv to be registered.
    """
    from gymnasium.vector import AsyncVectorEnv
    from gymnasium.envs.registration import register

    env_id = f"MiniGrid-Custom-Dynamic-Obstacles-{grid_size}x{grid_size}-v0"
    try:
        import sys, os
        scripts_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "scripts"
        )
        sys.path.insert(0, os.path.abspath(scripts_dir))
        from custom_dynamic_obstacles_env import CustomDynamicObstaclesEnv
        try:
            register(
                id=env_id,
                entry_point="custom_dynamic_obstacles_env:CustomDynamicObstaclesEnv",
                kwargs={"size": grid_size, "n_obstacles": 2},
            )
        except Exception:
            pass  # already registered
    except ImportError:
        import minigrid  # noqa: F401
        env_id = f"MiniGrid-Dynamic-Obstacles-{grid_size}x{grid_size}-v0"

    def _make(rank: int):
        def _init():
            env = gym.make(env_id, render_mode="rgb_array", highlight=False)
            env = DynamicObstaclesEnvWrapper(env, concept_version=3, n_stack=n_stack, grid_size=grid_size)
            env.reset(seed=seed + rank)
            return env
        return _init

    return AsyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_dynamic_obstacles_env(seed: int = 0, n_stack: int = 1, grid_size: int = 8) -> DynamicObstaclesEnvWrapper:
    from gymnasium.envs.registration import register

    env_id = f"MiniGrid-Custom-Dynamic-Obstacles-{grid_size}x{grid_size}-v0"
    try:
        import sys, os
        scripts_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "scripts"
        )
        sys.path.insert(0, os.path.abspath(scripts_dir))
        from custom_dynamic_obstacles_env import CustomDynamicObstaclesEnv
        try:
            register(
                id=env_id,
                entry_point="custom_dynamic_obstacles_env:CustomDynamicObstaclesEnv",
                kwargs={"size": grid_size, "n_obstacles": 2},
            )
        except Exception:
            pass
    except ImportError:
        import minigrid  # noqa: F401
        env_id = f"MiniGrid-Dynamic-Obstacles-{grid_size}x{grid_size}-v0"

    env = gym.make(env_id, render_mode="rgb_array", highlight=False)
    env = DynamicObstaclesEnvWrapper(env, concept_version=3, n_stack=n_stack, grid_size=grid_size)
    env.reset(seed=seed)
    return env
