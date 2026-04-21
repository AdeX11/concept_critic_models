"""
synchrony_window.py - Gymnasium wrappers for Synchrony Window.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import gymnasium as gym
import numpy as np

from .base import ConceptMetadataMixin
from .synchrony_window_core import (
    ACTION_DELTAS,
    CONCEPT_NAMES,
    NUM_CLASSES,
    OBS_SIZE,
    TASK_TYPES,
    TEMPORAL_CONCEPTS,
    SynchronyWindowSimulator,
)


class _SynchronyWindowEnvBase(ConceptMetadataMixin, gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}
    actions = SynchronyWindowSimulator.actions

    def __init__(self, seed: int = 0):
        super().__init__()
        self.sim = SynchronyWindowSimulator(seed=seed)
        self.task_types = list(TASK_TYPES)
        self.num_classes = list(NUM_CLASSES)
        self.concept_names = list(CONCEPT_NAMES)
        self.temporal_concepts = list(TEMPORAL_CONCEPTS)
        self.current_concept = np.zeros(len(self.task_types), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(ACTION_DELTAS))
        self._validate_concept_spec()

    def render(self):
        return self.sim.render_rgb()

    def _parse_reset_controls(self, options):
        forced_top_x = None
        forced_top_direction = None
        forced_bottom_x = None
        forced_bottom_direction = None
        if options is None:
            return forced_top_x, forced_top_direction, forced_bottom_x, forced_bottom_direction
        if not isinstance(options, dict):
            raise ValueError("reset options must be a dict or None")
        unknown = set(options.keys()) - {
            "forced_top_x",
            "forced_top_direction",
            "forced_bottom_x",
            "forced_bottom_direction",
        }
        if unknown:
            raise ValueError(f"Unknown reset options: {sorted(unknown)}")
        return (
            options.get("forced_top_x"),
            options.get("forced_top_direction"),
            options.get("forced_bottom_x"),
            options.get("forced_bottom_direction"),
        )


class SynchronyWindowStateEnv(_SynchronyWindowEnvBase):
    def __init__(self, seed: int = 0):
        super().__init__(seed=seed)
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(5,),
            dtype=np.float32,
        )

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        forced = self._parse_reset_controls(options)
        state = self.sim.reset(
            seed=seed,
            forced_top_x=forced[0],
            forced_top_direction=forced[1],
            forced_bottom_x=forced[2],
            forced_bottom_direction=forced[3],
        )
        self.current_concept = self.sim.extract_concepts(state)
        obs = self.sim.get_state_observation(state)
        info = {"concept": self.current_concept.copy()}
        return obs, info

    def step(self, action):
        state, reward, terminated, truncated, info = self.sim.step(action)
        self.current_concept = self.sim.extract_concepts(state)
        obs = self.sim.get_state_observation(state)
        info = dict(info)
        info["concept"] = self.current_concept.copy()
        if terminated or truncated:
            info["terminal_observation"] = obs.copy()
        return obs, reward, terminated, truncated, info


class SynchronyWindowPixelEnv(_SynchronyWindowEnvBase):
    def __init__(self, seed: int = 0, n_stack: int = 1):
        super().__init__(seed=seed)
        self.n_stack = int(n_stack)
        if self.n_stack < 1:
            raise ValueError("n_stack must be >= 1")
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3 * self.n_stack, OBS_SIZE, OBS_SIZE),
            dtype=np.uint8,
        )
        self._frames: Optional[deque[np.ndarray]]
        if self.n_stack > 1:
            self._frames = deque(maxlen=self.n_stack)
        else:
            self._frames = None

    def _render_state(self, state) -> np.ndarray:
        return self.sim.render_rgb(state)

    def _frame_to_obs(self, frame: np.ndarray) -> np.ndarray:
        chw = frame.transpose(2, 0, 1)
        if self._frames is None:
            return chw
        self._frames.append(chw)
        return np.concatenate(list(self._frames), axis=0)

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        forced = self._parse_reset_controls(options)
        state = self.sim.reset(
            seed=seed,
            forced_top_x=forced[0],
            forced_top_direction=forced[1],
            forced_bottom_x=forced[2],
            forced_bottom_direction=forced[3],
        )
        self.current_concept = self.sim.extract_concepts(state)
        frame = self._render_state(state)
        if self._frames is not None:
            self._frames.clear()
            chw = frame.transpose(2, 0, 1)
            for _ in range(self.n_stack):
                self._frames.append(chw.copy())
            obs = np.concatenate(list(self._frames), axis=0)
        else:
            obs = frame.transpose(2, 0, 1)
        info = {"concept": self.current_concept.copy()}
        return obs, info

    def step(self, action):
        state, reward, terminated, truncated, info = self.sim.step(action)
        self.current_concept = self.sim.extract_concepts(state)
        frame = self._render_state(state)
        obs = self._frame_to_obs(frame)
        info = dict(info)
        info["concept"] = self.current_concept.copy()
        if terminated or truncated:
            info["terminal_observation"] = obs.copy()
        return obs, reward, terminated, truncated, info


class SynchronyWindowVisibleEnv(SynchronyWindowPixelEnv):
    """
    Reactive control twin with both mover directions exposed in a HUD band.
    """

    def _render_state(self, state) -> np.ndarray:
        frame = self.sim.render_rgb(state).copy()
        hud_y0 = 72
        frame[hud_y0:84, :, :] = np.array([17, 18, 22], dtype=np.uint8)

        dir_color = np.array([236, 239, 244], dtype=np.uint8)
        # Top mover indicator on left half.
        if state.top_direction == "left":
            frame[hud_y0 + 2:hud_y0 + 10, 10:16] = dir_color
            frame[hud_y0 + 4:hud_y0 + 8, 6:10] = dir_color
        else:
            frame[hud_y0 + 2:hud_y0 + 10, 28:34] = dir_color
            frame[hud_y0 + 4:hud_y0 + 8, 34:38] = dir_color

        # Bottom mover indicator on right half.
        if state.bottom_direction == "left":
            frame[hud_y0 + 2:hud_y0 + 10, 48:54] = dir_color
            frame[hud_y0 + 4:hud_y0 + 8, 44:48] = dir_color
        else:
            frame[hud_y0 + 2:hud_y0 + 10, 66:72] = dir_color
            frame[hud_y0 + 4:hud_y0 + 8, 72:76] = dir_color

        return frame


def make_synchrony_window_env(
    n_envs: int = 4,
    seed: int = 0,
    n_stack: int = 1,
) -> gym.Env:
    from gymnasium.vector import AsyncVectorEnv

    def _make(rank: int):
        def _init():
            env = SynchronyWindowPixelEnv(seed=seed + rank, n_stack=n_stack)
            env.reset(seed=seed + rank)
            return env
        return _init

    return AsyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_synchrony_window_env(
    seed: int = 0,
    n_stack: int = 1,
) -> SynchronyWindowPixelEnv:
    env = SynchronyWindowPixelEnv(seed=seed, n_stack=n_stack)
    env.reset(seed=seed)
    return env


def make_synchrony_window_visible_env(
    n_envs: int = 4,
    seed: int = 0,
    n_stack: int = 1,
) -> gym.Env:
    from gymnasium.vector import AsyncVectorEnv

    def _make(rank: int):
        def _init():
            env = SynchronyWindowVisibleEnv(seed=seed + rank, n_stack=n_stack)
            env.reset(seed=seed + rank)
            return env
        return _init

    return AsyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_synchrony_window_visible_env(
    seed: int = 0,
    n_stack: int = 1,
) -> SynchronyWindowVisibleEnv:
    env = SynchronyWindowVisibleEnv(seed=seed, n_stack=n_stack)
    env.reset(seed=seed)
    return env


def make_synchrony_window_state_env(
    n_envs: int = 4,
    seed: int = 0,
    n_stack: Optional[int] = None,
    **kwargs,
) -> gym.Env:
    if kwargs:
        raise ValueError(f"Unexpected kwargs for synchrony_window_state: {sorted(kwargs)}")
    if n_stack not in (None, 1):
        raise ValueError("synchrony_window_state only supports n_stack=None or 1")

    from gymnasium.vector import AsyncVectorEnv

    def _make(rank: int):
        def _init():
            env = SynchronyWindowStateEnv(seed=seed + rank)
            env.reset(seed=seed + rank)
            return env
        return _init

    return AsyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_synchrony_window_state_env(
    seed: int = 0,
    n_stack: Optional[int] = None,
    **kwargs,
) -> SynchronyWindowStateEnv:
    if kwargs:
        raise ValueError(f"Unexpected kwargs for synchrony_window_state: {sorted(kwargs)}")
    if n_stack not in (None, 1):
        raise ValueError("synchrony_window_state only supports n_stack=None or 1")
    env = SynchronyWindowStateEnv(seed=seed)
    env.reset(seed=seed)
    return env
