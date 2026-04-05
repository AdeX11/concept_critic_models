"""
dynamic_obstacles.py — DynamicObstaclesEnvWrapper from LICORICE.

Uses concept_version=3 (11 concepts for 5x5 grid):
  agent_position_x, agent_position_y, agent_direction,
  obstacle1_position_x, obstacle1_position_y,
  obstacle2_position_x, obstacle2_position_y,
  movable_right, movable_down, movable_left, movable_up

Observation: RGB image [3*n_stack, ROWS, COLS].
  n_stack=1 (default) → [3, ROWS, COLS]  (single frame, no stacking)
  n_stack=4           → [12, ROWS, COLS] (4-frame stack for stacked temporal encoding)

Temporal concepts (move each step): indices 2 (direction), 6-10 (movable).
Static concepts: positions (change rarely).
"""

from collections import deque
from typing import Optional

import cv2
import gymnasium as gym
import numpy as np

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
    ):
        super().__init__(env)
        assert _MINIGRID_AVAILABLE, "minigrid package required for DynamicObstaclesEnvWrapper"
        self.ROWS = ROWS
        self.COLS = COLS
        self.concept_version = concept_version
        self.n_stack = n_stack

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3 * n_stack, ROWS, COLS), dtype=np.uint8
        )
        if n_stack > 1:
            self._frames: Optional[deque] = deque(maxlen=n_stack)
        else:
            self._frames = None

        # Only concept_version==3 supported here (11 concepts for 5x5)
        assert concept_version == 3, "Only concept_version=3 supported in this wrapper"
        self.task_types = ["classification"] * 11
        self.num_classes = [6, 6, 4, 6, 6, 6, 6, 2, 2, 2, 2]
        self.concept_names = [
            "agent_position_x", "agent_position_y", "agent_direction",
            "obstacle1_position_x", "obstacle1_position_y",
            "obstacle2_position_x", "obstacle2_position_y",
            "movable_right", "movable_down", "movable_left", "movable_up",
        ]
        # Temporal = direction + movable flags (change most steps)
        self.temporal_concepts = [2, 7, 8, 9, 10]

        self._current_concept: Optional[np.ndarray] = None

    # ------------------------------------------------------------------

    def get_concept(self) -> np.ndarray:
        return self._current_concept.copy() if self._current_concept is not None else np.zeros(11, dtype=np.float32)

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
        self._current_concept = self._compute_concept()
        info["concept"] = self._current_concept.copy()
        return stacked, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        frame = self._get_image()
        if self.n_stack > 1:
            self._frames.append(frame)
            stacked = np.concatenate(list(self._frames), axis=0)
        else:
            stacked = frame
        self._current_concept = self._compute_concept()
        if done or truncated:
            info["terminal_observation"] = stacked
        info["concept"] = self._current_concept.copy()
        return stacked, reward, done, truncated, info

    # ------------------------------------------------------------------

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

        def is_movable(x, y):
            cell = get_cell(x, y)
            return cell is None or cell.type == "empty"

        def can_move(pos, direction):
            dx, dy = DIR_TO_VEC[direction]
            nx, ny = pos[0] + dx, pos[1] + dy
            if 1 <= nx < grid.width - 1 and 1 <= ny < grid.height - 1:
                cell = get_cell(nx, ny)
                return cell is None or (hasattr(cell, "can_overlap") and cell.can_overlap())
            return False

        # Obstacle positions
        obstacle_positions = []
        for x in range(grid.width):
            for y in range(grid.height):
                cell = get_cell(x, y)
                if cell and cell.type == "ball":
                    obstacle_positions.append((x, y))
        obstacle_positions = sorted(obstacle_positions)
        # Pad to 2 obstacles
        while len(obstacle_positions) < 2:
            obstacle_positions.append((0, 0))

        movable = [
            int(can_move(agent_pos, 0)),  # right
            int(can_move(agent_pos, 1)),  # down
            int(can_move(agent_pos, 2)),  # left
            int(can_move(agent_pos, 3)),  # up
        ]

        values = [
            agent_pos[0], agent_pos[1], agent_dir,
            obstacle_positions[0][0], obstacle_positions[0][1],
            obstacle_positions[1][0], obstacle_positions[1][1],
        ] + movable

        return np.array(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_dynamic_obstacles_env(n_envs: int = 4, seed: int = 0, n_stack: int = 1) -> gym.Env:
    """
    Returns a vectorised DynamicObstaclesEnvWrapper (5x5 grid, 2 obstacles).
    Requires minigrid and the CustomDynamicObstaclesEnv to be registered.
    """
    from gymnasium.vector import SyncVectorEnv
    from gymnasium.envs.registration import register

    # Try to register custom env; fall back to standard if script not available
    env_id = "MiniGrid-Custom-Dynamic-Obstacles-5x5-v0"
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
                kwargs={"size": 5, "n_obstacles": 2},
            )
        except Exception:
            pass  # already registered
    except ImportError:
        # Fall back to standard minigrid env
        import minigrid  # noqa: F401
        env_id = "MiniGrid-Dynamic-Obstacles-5x5-v0"

    def _make(rank: int):
        def _init():
            env = gym.make(env_id, render_mode="rgb_array", highlight=False)
            env = DynamicObstaclesEnvWrapper(env, concept_version=3, n_stack=n_stack)
            env.reset(seed=seed + rank)
            return env
        return _init

    return SyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_dynamic_obstacles_env(seed: int = 0, n_stack: int = 1) -> DynamicObstaclesEnvWrapper:
    from gymnasium.envs.registration import register

    env_id = "MiniGrid-Custom-Dynamic-Obstacles-5x5-v0"
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
                kwargs={"size": 5, "n_obstacles": 2},
            )
        except Exception:
            pass
    except ImportError:
        env_id = "MiniGrid-Dynamic-Obstacles-5x5-v0"

    env = gym.make(env_id, render_mode="rgb_array", highlight=False)
    env = DynamicObstaclesEnvWrapper(env, concept_version=3, n_stack=n_stack)
    env.reset(seed=seed)
    return env
