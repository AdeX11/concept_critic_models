"""
tmaze.py - delayed-cue memory task.

The agent moves down a corridor, sees a binary cue only at the beginning, and
must remember it at the junction. This is a compact diagnostic for recurrent
memory and concept steerability.

Observation:
    [norm_pos, cue_active, cue_value, at_junction]

Actions:
    0 = move forward
    1 = choose left  (correct when cue == 0)
    2 = choose right (correct when cue == 1)

Concepts:
    0 cue         classification {0, 1}, temporal
    1 at_junction classification {0, 1}, static
"""

from __future__ import annotations

from collections import deque
from typing import Deque

import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv


CUE_STEPS = 3
CORRIDOR_LEN = 10


class TMazeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0, corridor_len: int = CORRIDOR_LEN):
        super().__init__()
        self.corridor_len = int(corridor_len)
        self.max_steps = self.corridor_len + 5
        self.observation_space = gym.spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(3)

        self.task_types = ["classification", "classification"]
        self.num_classes = [2, 2]
        self.concept_names = ["cue", "at_junction"]
        self.temporal_concepts = [0]

        self._rng = np.random.default_rng(seed)
        self._pos = 0
        self._cue = 0
        self._steps = 0
        self.current_concept = np.zeros(2, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._pos = 0
        self._cue = int(self._rng.integers(0, 2))
        self._steps = 0
        self.current_concept = self._compute_concept()
        return self._get_obs(), {"concept": self.current_concept.copy()}

    def step(self, action: int):
        action = int(action)
        self._steps += 1
        reward = -0.01
        terminated = False

        at_junction = self._pos == self.corridor_len
        if at_junction:
            if action == 1:
                reward += 1.0 if self._cue == 0 else -1.0
            elif action == 2:
                reward += 1.0 if self._cue == 1 else -1.0
            else:
                reward += -1.0
            terminated = True
        else:
            if action == 0:
                self._pos += 1
            else:
                reward -= 0.05

        truncated = (not terminated) and self._steps >= self.max_steps
        self.current_concept = self._compute_concept()
        return (
            self._get_obs(),
            float(reward),
            bool(terminated),
            bool(truncated),
            {"concept": self.current_concept.copy()},
        )

    def get_concept(self) -> np.ndarray:
        return self.current_concept.copy()

    @property
    def concept_reward_active(self) -> float:
        return 1.0 if self._pos == self.corridor_len else 0.0

    def _get_obs(self) -> np.ndarray:
        cue_active = 1.0 if self._pos < CUE_STEPS else 0.0
        cue_value = float(self._cue) if self._pos < CUE_STEPS else 0.0
        at_junction = 1.0 if self._pos == self.corridor_len else 0.0
        return np.array(
            [self._pos / self.corridor_len, cue_active, cue_value, at_junction],
            dtype=np.float32,
        )

    def _compute_concept(self) -> np.ndarray:
        at_junction = 1.0 if self._pos == self.corridor_len else 0.0
        return np.array([float(self._cue), at_junction], dtype=np.float32)


class FrameStackFlatWrapper(gym.Wrapper):
    """Flat frame stack wrapper for low-dimensional observations."""

    def __init__(self, env: gym.Env, n_stack: int):
        super().__init__(env)
        self.n_stack = int(n_stack)
        self._frames: Deque[np.ndarray] = deque(maxlen=self.n_stack)
        self.observation_space = gym.spaces.Box(
            low=np.tile(env.observation_space.low, self.n_stack).astype(np.float32),
            high=np.tile(env.observation_space.high, self.n_stack).astype(np.float32),
            dtype=np.float32,
        )
        for attr in (
            "task_types",
            "num_classes",
            "concept_names",
            "temporal_concepts",
            "current_concept",
        ):
            if hasattr(env, attr):
                setattr(self, attr, getattr(env, attr))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_stack):
            self._frames.append(obs.copy())
        self._sync_concept()
        return self._stacked_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs.copy())
        self._sync_concept()
        return self._stacked_obs(), reward, terminated, truncated, info

    def get_concept(self) -> np.ndarray:
        return self.env.get_concept()

    @property
    def concept_reward_active(self) -> float:
        return getattr(self.env, "concept_reward_active", 1.0)

    def _sync_concept(self) -> None:
        if hasattr(self.env, "current_concept"):
            self.current_concept = self.env.current_concept

    def _stacked_obs(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=0).astype(np.float32)


def _make_tmaze(seed: int, n_stack: int = 1) -> gym.Env:
    env: gym.Env = TMazeEnv(seed=seed, corridor_len=CORRIDOR_LEN)
    if n_stack > 1:
        env = FrameStackFlatWrapper(env, n_stack=n_stack)
    return env


def make_tmaze_env(n_envs: int = 4, seed: int = 0, n_stack: int = 1, **_) -> SyncVectorEnv:
    def _make(rank: int):
        def _init():
            return _make_tmaze(seed + rank, n_stack=n_stack)

        return _init

    return SyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_tmaze_env(seed: int = 0, n_stack: int = 1, **_) -> gym.Env:
    return _make_tmaze(seed=seed, n_stack=n_stack)
