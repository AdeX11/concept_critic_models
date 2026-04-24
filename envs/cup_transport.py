"""
cup_transport.py — wrapper around CupTransportEnv (cup_transport_env.py).

Provides pixel observations plus per-step concept vectors for the cup transport
task (pick up, fill at fountain, deliver filled cups to the right wall).

Observation: RGB image [3*n_stack, ROWS, COLS] (same stacking contract as
dynamic_obstacles.py).

Concept layout (one scalar per index; classification head sizes in num_classes):
  agent_position_x, agent_position_y, agent_direction,
  fountain_position_x, fountain_position_y,
  for each cup i: cup_i_x, cup_i_y, cup_i_filled, cup_i_delivered,
  movable_right, movable_down, movable_left, movable_up,
  carrying_cup_index (0..num_cups-1 = carrying that cup, num_cups = none)

Temporal concepts: direction, movable flags, filled/delivered, carrying index.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Optional

import cv2
import gymnasium as gym
import numpy as np

try:
    from minigrid.core.constants import DIR_TO_VEC
    _MINIGRID_AVAILABLE = True
except ImportError:
    _MINIGRID_AVAILABLE = False
    DIR_TO_VEC = None

if TYPE_CHECKING:
    from .cup_transport_env import Cup, CupTransportEnv


class CupTransportEnvWrapper(gym.Wrapper):
    """Wraps CupTransportEnv: RGB tensor observations + concept annotations."""

    def __init__(
        self,
        env: gym.Env,
        ROWS: int = 160,
        COLS: int = 160,
        n_stack: int = 1,
    ):
        super().__init__(env)
        assert _MINIGRID_AVAILABLE, "minigrid package required for CupTransportEnvWrapper"
        self.ROWS = ROWS
        self.COLS = COLS
        self.n_stack = n_stack

        unwrapped = env.unwrapped
        self._width = unwrapped.width
        self._height = unwrapped.height
        self._num_cups = unwrapped.num_cups

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3 * n_stack, ROWS, COLS), dtype=np.uint8
        )
        if n_stack > 1:
            self._frames: Optional[deque] = deque(maxlen=n_stack)
        else:
            self._frames = None

        w, h, nc = self._width, self._height, self._num_cups
        self.concept_names = (
            ["agent_position_x", "agent_position_y", "agent_direction"]
            + ["fountain_position_x", "fountain_position_y"]
            + [name for i in range(nc) for name in (
                f"cup{i}_x", f"cup{i}_y", f"cup{i}_filled", f"cup{i}_delivered",
            )]
            + ["movable_right", "movable_down", "movable_left", "movable_up"]
            + ["carrying_cup_index"]
        )
        n_concepts = len(self.concept_names)
        self.task_types = ["classification"] * n_concepts
        self.num_classes = (
            [w, h, 4, w, h]
            + [w, h, 2, 2] * nc
            + [2, 2, 2, 2, nc + 1]
        )
        movable_start = 5 + 4 * nc
        self.temporal_concepts = (
            [2]
            + [5 + 4 * i + j for i in range(nc) for j in (2, 3)]
            + list(range(movable_start, movable_start + 5))
        )

        self.current_concept: Optional[np.ndarray] = None

    def get_concept(self) -> np.ndarray:
        n = len(self.concept_names)
        if self.current_concept is not None:
            return self.current_concept.copy()
        return np.zeros(n, dtype=np.float32)

    def reset(self, **kwargs):
        _obs, info = self.env.reset(**kwargs)
        frame = self._get_image()
        if self.n_stack > 1:
            assert self._frames is not None
            self._frames.clear()
            for _ in range(self.n_stack):
                self._frames.append(frame)
            stacked = np.concatenate(list(self._frames), axis=0)
        else:
            stacked = frame
        self.current_concept = self._compute_concept()
        info["concept"] = self.current_concept.copy()
        return stacked, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self._get_image()
        if self.n_stack > 1:
            assert self._frames is not None
            self._frames.append(frame)
            stacked = np.concatenate(list(self._frames), axis=0)
        else:
            stacked = frame
        self.current_concept = self._compute_concept()
        if terminated or truncated:
            info["terminal_observation"] = stacked
        info["concept"] = self.current_concept.copy()
        return stacked, reward, terminated, truncated, info

    def _get_image(self) -> np.ndarray:
        img = self.env.render()
        img = cv2.resize(img, (self.COLS, self.ROWS))
        return img.transpose(2, 0, 1)

    def _locate_cup(self, cup_idx: int) -> tuple[int, int]:
        from .cup_transport_env import Cup

        env: CupTransportEnv = self.env.unwrapped  # type: ignore[assignment]
        color = env.cup_positions[cup_idx][1]
        carrying = env.carrying
        if carrying and isinstance(carrying, Cup) and carrying.color == color:
            return int(env.agent_pos[0]), int(env.agent_pos[1])
        for x in range(env.grid.width):
            for y in range(env.grid.height):
                cell = env.grid.get(x, y)
                if (
                    cell is not None
                    and getattr(cell, "type", None) == "cup"
                    and getattr(cell, "color", None) == color
                ):
                    return int(x), int(y)
        return 0, 0

    def _carrying_cup_index(self) -> int:
        from .cup_transport_env import Cup

        env: CupTransportEnv = self.env.unwrapped  # type: ignore[assignment]
        if not env.carrying or not isinstance(env.carrying, Cup):
            return self._num_cups
        for i in range(self._num_cups):
            if env.carrying.color == env.cup_positions[i][1]:
                return i
        return self._num_cups

    def _compute_concept(self) -> np.ndarray:
        env: CupTransportEnv = self.env.unwrapped  # type: ignore[assignment]
        agent_pos = env.agent_pos
        agent_dir = env.agent_dir
        grid = env.grid
        fx, fy = env.fountain_pos

        def get_cell(x: int, y: int):
            if x < 0 or x >= grid.width or y < 0 or y >= grid.height:
                return None
            return grid.get(x, y)

        def can_move(pos: tuple[int, int], direction: int) -> bool:
            dx, dy = DIR_TO_VEC[direction]
            nx, ny = pos[0] + dx, pos[1] + dy
            if 1 <= nx < grid.width - 1 and 1 <= ny < grid.height - 1:
                cell = get_cell(nx, ny)
                return cell is None or (hasattr(cell, "can_overlap") and cell.can_overlap())
            return False

        movable = [
            int(can_move(agent_pos, 0)),
            int(can_move(agent_pos, 1)),
            int(can_move(agent_pos, 2)),
            int(can_move(agent_pos, 3)),
        ]

        cup_vals: list[float] = []
        for i in range(self._num_cups):
            cx, cy = self._locate_cup(i)
            cup_vals.extend(
                [
                    float(cx),
                    float(cy),
                    float(int(env.cups_filled[i])),
                    float(int(env.cups_delivered[i])),
                ]
            )

        carry_idx = self._carrying_cup_index()

        values = [
            float(agent_pos[0]),
            float(agent_pos[1]),
            float(agent_dir),
            float(fx),
            float(fy),
        ] + cup_vals + [float(m) for m in movable] + [float(carry_idx)]

        return np.array(values, dtype=np.float32)


def _register_cup_transport_env() -> str:
    import os
    import sys

    from gymnasium.envs.registration import register

    env_id = "MiniGrid-Cup-Spilling-v0"
    envs_dir = os.path.dirname(os.path.abspath(__file__))
    if envs_dir not in sys.path:
        sys.path.insert(0, envs_dir)
    try:
        register(
            id=env_id,
            entry_point="cup_transport_env:CupTransportEnv",
            kwargs={},
        )
    except Exception:
        pass
    return env_id


def _fixed_height_kwargs(grid_height: int) -> dict:
    """CupTransportEnv draws height via _rand_int(min_height+2, max_height+3)."""
    mh = grid_height - 2
    return {"min_height": mh, "max_height": mh}


def make_cup_transport_env(
    n_envs: int = 4,
    seed: int = 0,
    n_stack: int = 1,
    grid_height: int = 7,
) -> gym.Env:
    """
    Vectorised CupTransportEnvWrapper around CupTransportEnv.

    ``grid_height`` is fixed across sub-envs so observation / concept shapes match.
    num_cups = grid_height - 3.
    """
    from gymnasium.vector import AsyncVectorEnv

    env_id = _register_cup_transport_env()
    kwargs = _fixed_height_kwargs(grid_height)

    def _make(rank: int):
        def _init():
            env = gym.make(env_id, render_mode="rgb_array", highlight=False, **kwargs)
            env = CupTransportEnvWrapper(env, n_stack=n_stack)
            env.reset(seed=seed + rank)
            return env

        return _init

    return AsyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_cup_transport_env(
    seed: int = 0,
    n_stack: int = 1,
    grid_height: int = 7,
) -> CupTransportEnvWrapper:
    env_id = _register_cup_transport_env()
    kwargs = _fixed_height_kwargs(grid_height)
    env = gym.make(env_id, render_mode="rgb_array", highlight=False, **kwargs)
    env = CupTransportEnvWrapper(env, n_stack=n_stack)
    env.reset(seed=seed)
    return env
