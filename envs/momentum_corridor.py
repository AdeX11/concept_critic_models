"""
momentum_corridor.py - Gymnasium wrappers for Momentum Corridor.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import gymnasium as gym
import numpy as np

from .base import ConceptMetadataMixin
from .momentum_corridor_core import (
    ACTION_DELTAS,
    CONCEPT_NAMES,
    DEFAULT_HAZARD_AGENT_Y,
    HARD_MAX_STEPS,
    HARD_HAZARD_AGENT_Y,
    HARD_VELOCITY_VALUES,
    NUM_CLASSES,
    OBS_SIZE,
    TASK_TYPES,
    TEMPORAL_CONCEPTS,
    MomentumCorridorSimulator,
)


class _MomentumCorridorEnvBase(ConceptMetadataMixin, gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}
    actions = MomentumCorridorSimulator.actions

    def __init__(
        self,
        seed: int = 0,
        hazard_agent_y=DEFAULT_HAZARD_AGENT_Y,
        max_steps=None,
        velocity_values=None,
    ):
        super().__init__()
        kwargs = {"seed": seed, "hazard_agent_y": tuple(hazard_agent_y)}
        if max_steps is not None:
            kwargs["max_steps"] = max_steps
        if velocity_values is not None:
            kwargs["velocity_values"] = tuple(velocity_values)
        self.sim = MomentumCorridorSimulator(**kwargs)
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
        forced_x = None
        forced_velocity = None
        if options is None:
            return forced_x, forced_velocity
        if not isinstance(options, dict):
            raise ValueError("reset options must be a dict or None")
        unknown = set(options.keys()) - {"forced_mover_x", "forced_mover_velocity"}
        if unknown:
            raise ValueError(f"Unknown reset options: {sorted(unknown)}")
        forced_x = options.get("forced_mover_x")
        forced_velocity = options.get("forced_mover_velocity")
        return forced_x, forced_velocity


class MomentumCorridorStateEnv(_MomentumCorridorEnvBase):
    def __init__(
        self,
        seed: int = 0,
        hazard_agent_y=DEFAULT_HAZARD_AGENT_Y,
        max_steps=None,
        velocity_values=None,
    ):
        super().__init__(
            seed=seed,
            hazard_agent_y=hazard_agent_y,
            max_steps=max_steps,
            velocity_values=velocity_values,
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        forced_x, forced_velocity = self._parse_reset_controls(options)
        state = self.sim.reset(
            seed=seed,
            forced_mover_x=forced_x,
            forced_mover_velocity=forced_velocity,
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


class MomentumCorridorPixelEnv(_MomentumCorridorEnvBase):
    def __init__(
        self,
        seed: int = 0,
        n_stack: int = 1,
        hazard_agent_y=DEFAULT_HAZARD_AGENT_Y,
        max_steps=None,
        velocity_values=None,
    ):
        super().__init__(
            seed=seed,
            hazard_agent_y=hazard_agent_y,
            max_steps=max_steps,
            velocity_values=velocity_values,
        )
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
        forced_x, forced_velocity = self._parse_reset_controls(options)
        state = self.sim.reset(
            seed=seed,
            forced_mover_x=forced_x,
            forced_mover_velocity=forced_velocity,
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


class MomentumCorridorVisibleEnv(MomentumCorridorPixelEnv):
    """
    Reactive control twin with mover velocity made explicit in a HUD band.
    """

    def _render_state(self, state) -> np.ndarray:
        frame = self.sim.render_rgb(state).copy()
        hud_y0 = 72
        frame[hud_y0:84, :, :] = np.array([17, 18, 22], dtype=np.uint8)

        vel = state.mover_velocity
        dir_color = np.array([236, 239, 244], dtype=np.uint8)
        if vel < 0:
            frame[hud_y0 + 2:hud_y0 + 10, 8:14] = dir_color
            frame[hud_y0 + 4:hud_y0 + 8, 4:8] = dir_color
        else:
            frame[hud_y0 + 2:hud_y0 + 10, 70:76] = dir_color
            frame[hud_y0 + 4:hud_y0 + 8, 76:80] = dir_color

        mag = abs(vel)
        for i in range(2):
            xs = 34 + i * 10
            color = np.array([77, 81, 89], dtype=np.uint8)
            if i < mag:
                color = np.array([255, 118, 67], dtype=np.uint8)
            frame[hud_y0 + 3:hud_y0 + 9, xs:xs + 7] = color
        return frame


def make_momentum_corridor_env(
    n_envs: int = 4,
    seed: int = 0,
    n_stack: int = 1,
) -> gym.Env:
    from gymnasium.vector import AsyncVectorEnv

    def _make(rank: int):
        def _init():
            env = MomentumCorridorPixelEnv(seed=seed + rank, n_stack=n_stack)
            env.reset(seed=seed + rank)
            return env
        return _init

    return AsyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_momentum_corridor_env(
    seed: int = 0,
    n_stack: int = 1,
) -> MomentumCorridorPixelEnv:
    env = MomentumCorridorPixelEnv(seed=seed, n_stack=n_stack)
    env.reset(seed=seed)
    return env


def make_momentum_corridor_visible_env(
    n_envs: int = 4,
    seed: int = 0,
    n_stack: int = 1,
) -> gym.Env:
    from gymnasium.vector import AsyncVectorEnv

    def _make(rank: int):
        def _init():
            env = MomentumCorridorVisibleEnv(seed=seed + rank, n_stack=n_stack)
            env.reset(seed=seed + rank)
            return env
        return _init

    return AsyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_momentum_corridor_visible_env(
    seed: int = 0,
    n_stack: int = 1,
) -> MomentumCorridorVisibleEnv:
    env = MomentumCorridorVisibleEnv(seed=seed, n_stack=n_stack)
    env.reset(seed=seed)
    return env


def make_momentum_corridor_state_env(
    n_envs: int = 4,
    seed: int = 0,
    n_stack: Optional[int] = None,
    **kwargs,
) -> gym.Env:
    if kwargs:
        raise ValueError(f"Unexpected kwargs for momentum_corridor_state: {sorted(kwargs)}")
    if n_stack not in (None, 1):
        raise ValueError("momentum_corridor_state only supports n_stack=None or 1")

    from gymnasium.vector import AsyncVectorEnv

    def _make(rank: int):
        def _init():
            env = MomentumCorridorStateEnv(seed=seed + rank)
            env.reset(seed=seed + rank)
            return env
        return _init

    return AsyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_momentum_corridor_state_env(
    seed: int = 0,
    n_stack: Optional[int] = None,
    **kwargs,
) -> MomentumCorridorStateEnv:
    if kwargs:
        raise ValueError(f"Unexpected kwargs for momentum_corridor_state: {sorted(kwargs)}")
    if n_stack not in (None, 1):
        raise ValueError("momentum_corridor_state only supports n_stack=None or 1")
    env = MomentumCorridorStateEnv(seed=seed)
    env.reset(seed=seed)
    return env


def make_momentum_corridor_hard_env(
    n_envs: int = 4,
    seed: int = 0,
    n_stack: int = 1,
) -> gym.Env:
    from gymnasium.vector import AsyncVectorEnv

    def _make(rank: int):
        def _init():
            env = MomentumCorridorPixelEnv(
                seed=seed + rank,
                n_stack=n_stack,
                hazard_agent_y=HARD_HAZARD_AGENT_Y,
                max_steps=HARD_MAX_STEPS,
                velocity_values=HARD_VELOCITY_VALUES,
            )
            env.reset(seed=seed + rank)
            return env
        return _init

    return AsyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_momentum_corridor_hard_env(
    seed: int = 0,
    n_stack: int = 1,
) -> MomentumCorridorPixelEnv:
    env = MomentumCorridorPixelEnv(
        seed=seed,
        n_stack=n_stack,
        hazard_agent_y=HARD_HAZARD_AGENT_Y,
        max_steps=HARD_MAX_STEPS,
        velocity_values=HARD_VELOCITY_VALUES,
    )
    env.reset(seed=seed)
    return env


def make_momentum_corridor_hard_visible_env(
    n_envs: int = 4,
    seed: int = 0,
    n_stack: int = 1,
) -> gym.Env:
    from gymnasium.vector import AsyncVectorEnv

    def _make(rank: int):
        def _init():
            env = MomentumCorridorVisibleEnv(
                seed=seed + rank,
                n_stack=n_stack,
                hazard_agent_y=HARD_HAZARD_AGENT_Y,
                max_steps=HARD_MAX_STEPS,
                velocity_values=HARD_VELOCITY_VALUES,
            )
            env.reset(seed=seed + rank)
            return env
        return _init

    return AsyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_momentum_corridor_hard_visible_env(
    seed: int = 0,
    n_stack: int = 1,
) -> MomentumCorridorVisibleEnv:
    env = MomentumCorridorVisibleEnv(
        seed=seed,
        n_stack=n_stack,
        hazard_agent_y=HARD_HAZARD_AGENT_Y,
        max_steps=HARD_MAX_STEPS,
        velocity_values=HARD_VELOCITY_VALUES,
    )
    env.reset(seed=seed)
    return env


def make_momentum_corridor_hard_state_env(
    n_envs: int = 4,
    seed: int = 0,
    n_stack: Optional[int] = None,
    **kwargs,
) -> gym.Env:
    if kwargs:
        raise ValueError(f"Unexpected kwargs for momentum_corridor_hard_state: {sorted(kwargs)}")
    if n_stack not in (None, 1):
        raise ValueError("momentum_corridor_hard_state only supports n_stack=None or 1")

    from gymnasium.vector import AsyncVectorEnv

    def _make(rank: int):
        def _init():
            env = MomentumCorridorStateEnv(
                seed=seed + rank,
                hazard_agent_y=HARD_HAZARD_AGENT_Y,
                max_steps=HARD_MAX_STEPS,
                velocity_values=HARD_VELOCITY_VALUES,
            )
            env.reset(seed=seed + rank)
            return env
        return _init

    return AsyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_momentum_corridor_hard_state_env(
    seed: int = 0,
    n_stack: Optional[int] = None,
    **kwargs,
) -> MomentumCorridorStateEnv:
    if kwargs:
        raise ValueError(f"Unexpected kwargs for momentum_corridor_hard_state: {sorted(kwargs)}")
    if n_stack not in (None, 1):
        raise ValueError("momentum_corridor_hard_state only supports n_stack=None or 1")
    env = MomentumCorridorStateEnv(
        seed=seed,
        hazard_agent_y=HARD_HAZARD_AGENT_Y,
        max_steps=HARD_MAX_STEPS,
        velocity_values=HARD_VELOCITY_VALUES,
    )
    env.reset(seed=seed)
    return env
