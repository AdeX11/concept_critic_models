"""
armed_corridor.py - Gymnasium wrappers for Armed Corridor.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import gymnasium as gym
import numpy as np

from .armed_corridor_core import (
    ACTION_DELTAS,
    CONCEPT_NAMES,
    GRID_HEIGHT,
    GRID_WIDTH,
    NUM_CLASSES,
    OBS_SIZE,
    TASK_TYPES,
    TEMPORAL_CONCEPTS,
    ArmedCorridorSimulator,
)
from .base import ConceptMetadataMixin


class _ArmedCorridorEnvBase(ConceptMetadataMixin, gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}
    actions = ArmedCorridorSimulator.actions

    def __init__(self, seed: int = 0):
        super().__init__()
        self.sim = ArmedCorridorSimulator(seed=seed)
        self.task_types = list(TASK_TYPES)
        self.num_classes = list(NUM_CLASSES)
        self.concept_names = list(CONCEPT_NAMES)
        self.temporal_concepts = list(TEMPORAL_CONCEPTS)
        self.current_concept = np.zeros(len(self.task_types), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(ACTION_DELTAS))
        self._validate_concept_spec()

    def render(self):
        return self.sim.render_rgb()

    def _parse_forced_fuse_type(self, options) -> Optional[str]:
        if options is None:
            return None
        if not isinstance(options, dict):
            raise ValueError("reset options must be a dict or None")
        return options.get("forced_fuse_type")


class ArmedCorridorStateEnv(_ArmedCorridorEnvBase):
    """
    Position-only diagnostic environment.
    """

    def __init__(self, seed: int = 0):
        super().__init__(seed=seed)
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        forced_fuse_type = self._parse_forced_fuse_type(options)
        state = self.sim.reset(seed=seed, forced_fuse_type=forced_fuse_type)
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


class ArmedCorridorPixelEnv(_ArmedCorridorEnvBase):
    """
    Pixel benchmark environment with optional frame stacking.
    """

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
        forced_fuse_type = self._parse_forced_fuse_type(options)
        state = self.sim.reset(seed=seed, forced_fuse_type=forced_fuse_type)
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


class ArmedCorridorVisibleEnv(ArmedCorridorPixelEnv):
    """
    Reactive control twin: same mechanics, but fuse state and remaining budget are
    explicitly visible in the observation via a bottom HUD band.
    """

    def _render_state(self, state) -> np.ndarray:
        frame = self.sim.render_rgb(state).copy()
        hud_y0 = 72
        frame[hud_y0:84, :, :] = np.array([18, 18, 22], dtype=np.uint8)

        # Fuse indicator: red=short, green=long, gray=untriggered
        if not state.triggered or state.fuse_type is None:
            fuse_color = np.array([105, 110, 118], dtype=np.uint8)
        elif state.fuse_type == "short":
            fuse_color = np.array([214, 48, 49], dtype=np.uint8)
        else:
            fuse_color = np.array([42, 157, 88], dtype=np.uint8)
        frame[hud_y0 + 2: hud_y0 + 10, 4:16] = fuse_color

        # Remaining-budget bar: up to 8 ticks, visible even after expiry.
        remaining_budget = self.sim.compute_remaining_budget(state)
        tick_w = 5
        x0 = 22
        max_budget = 8
        if remaining_budget is None:
            remaining_budget = 0
        for i in range(max_budget):
            color = np.array([80, 84, 90], dtype=np.uint8)
            if i < remaining_budget:
                color = np.array([245, 197, 66], dtype=np.uint8)
            xs = x0 + i * (tick_w + 2)
            frame[hud_y0 + 3: hud_y0 + 9, xs: xs + tick_w] = color

        # Triggered flag: blue when active, dark otherwise.
        trig_color = np.array([54, 124, 255], dtype=np.uint8) if state.triggered else np.array([60, 62, 66], dtype=np.uint8)
        frame[hud_y0 + 2: hud_y0 + 10, 72:80] = trig_color
        return frame


def make_armed_corridor_env(
    n_envs: int = 4,
    seed: int = 0,
    n_stack: int = 1,
) -> gym.Env:
    from gymnasium.vector import AsyncVectorEnv

    def _make(rank: int):
        def _init():
            env = ArmedCorridorPixelEnv(seed=seed + rank, n_stack=n_stack)
            env.reset(seed=seed + rank)
            return env
        return _init

    return AsyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_armed_corridor_env(
    seed: int = 0,
    n_stack: int = 1,
) -> ArmedCorridorPixelEnv:
    env = ArmedCorridorPixelEnv(seed=seed, n_stack=n_stack)
    env.reset(seed=seed)
    return env


def make_armed_corridor_visible_env(
    n_envs: int = 4,
    seed: int = 0,
    n_stack: int = 1,
) -> gym.Env:
    from gymnasium.vector import AsyncVectorEnv

    def _make(rank: int):
        def _init():
            env = ArmedCorridorVisibleEnv(seed=seed + rank, n_stack=n_stack)
            env.reset(seed=seed + rank)
            return env
        return _init

    return AsyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_armed_corridor_visible_env(
    seed: int = 0,
    n_stack: int = 1,
) -> ArmedCorridorVisibleEnv:
    env = ArmedCorridorVisibleEnv(seed=seed, n_stack=n_stack)
    env.reset(seed=seed)
    return env


def make_armed_corridor_state_env(
    n_envs: int = 4,
    seed: int = 0,
    n_stack: Optional[int] = None,
    **kwargs,
) -> gym.Env:
    if kwargs:
        raise ValueError(f"Unexpected kwargs for armed_corridor_state: {sorted(kwargs)}")
    if n_stack not in (None, 1):
        raise ValueError("armed_corridor_state only supports n_stack=None or 1")

    from gymnasium.vector import AsyncVectorEnv

    def _make(rank: int):
        def _init():
            env = ArmedCorridorStateEnv(seed=seed + rank)
            env.reset(seed=seed + rank)
            return env
        return _init

    return AsyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_armed_corridor_state_env(
    seed: int = 0,
    n_stack: Optional[int] = None,
    **kwargs,
) -> ArmedCorridorStateEnv:
    if kwargs:
        raise ValueError(f"Unexpected kwargs for armed_corridor_state: {sorted(kwargs)}")
    if n_stack not in (None, 1):
        raise ValueError("armed_corridor_state only supports n_stack=None or 1")
    env = ArmedCorridorStateEnv(seed=seed)
    env.reset(seed=seed)
    return env
