"""
hidden_velocity.py - 2D navigation with velocity hidden from observations.

The current observation contains position and goal only:
    [x, y, goal_x, goal_y]

Velocity is part of the simulator state but not the observation. The temporal
concepts therefore require history rather than a single frame.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv


BOUNDS = 1.0
FRICTION = 0.85
THRUST = 0.08
GOAL_RADIUS = 0.12
MAX_STEPS = 400
N_CONCEPTS = 8

_THRUST_MAP = np.array(
    [
        [0.0, 0.0],
        [THRUST, 0.0],
        [-THRUST, 0.0],
        [0.0, THRUST],
        [0.0, -THRUST],
    ],
    dtype=np.float32,
)


class HiddenVelocityEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=np.array([-BOUNDS, -BOUNDS, -BOUNDS, -BOUNDS], dtype=np.float32),
            high=np.array([BOUNDS, BOUNDS, BOUNDS, BOUNDS], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(5)

        self.task_types = ["regression"] * 7 + ["classification"]
        self.num_classes = [0] * 7 + [2]
        self.concept_names = [
            "x_pos",
            "y_pos",
            "goal_x",
            "goal_y",
            "dist_to_goal",
            "x_velocity",
            "y_velocity",
            "approaching_goal",
        ]
        self.temporal_concepts = [5, 6, 7]

        self._rng = np.random.default_rng(seed)
        self._x = 0.0
        self._y = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._gx = 0.0
        self._gy = 0.0
        self._steps = 0
        self.current_concept = np.zeros(N_CONCEPTS, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._x, self._y = self._rng.uniform(-BOUNDS * 0.8, BOUNDS * 0.8, size=2)
        max_v = THRUST * 3.0
        self._vx, self._vy = self._rng.uniform(-max_v, max_v, size=2)
        self._gx, self._gy = self._spawn_goal()
        self._steps = 0
        self.current_concept = self._compute_concept()
        return self._get_obs(), {"concept": self.current_concept.copy()}

    def step(self, action: int):
        thrust = _THRUST_MAP[int(action)]
        self._vx = FRICTION * self._vx + float(thrust[0])
        self._vy = FRICTION * self._vy + float(thrust[1])

        new_x = self._x + self._vx
        new_y = self._y + self._vy
        reward = -0.01
        hit_wall = False

        if new_x < -BOUNDS or new_x > BOUNDS:
            self._vx = 0.0
            new_x = float(np.clip(new_x, -BOUNDS, BOUNDS))
            hit_wall = True
        if new_y < -BOUNDS or new_y > BOUNDS:
            self._vy = 0.0
            new_y = float(np.clip(new_y, -BOUNDS, BOUNDS))
            hit_wall = True
        if hit_wall:
            reward -= 0.05

        self._x = float(new_x)
        self._y = float(new_y)
        self._steps += 1

        dist = float(np.sqrt((self._x - self._gx) ** 2 + (self._y - self._gy) ** 2))
        if dist < GOAL_RADIUS:
            reward += 1.0
            self._gx, self._gy = self._spawn_goal()

        self.current_concept = self._compute_concept()
        return (
            self._get_obs(),
            float(reward),
            False,
            bool(self._steps >= MAX_STEPS),
            {"concept": self.current_concept.copy()},
        )

    def get_concept(self) -> np.ndarray:
        return self.current_concept.copy()

    def _spawn_goal(self) -> tuple[float, float]:
        for _ in range(50):
            gx, gy = self._rng.uniform(-BOUNDS * 0.85, BOUNDS * 0.85, size=2)
            if np.sqrt((self._x - gx) ** 2 + (self._y - gy) ** 2) > 0.3:
                return float(gx), float(gy)
        gx, gy = self._rng.uniform(-BOUNDS * 0.8, BOUNDS * 0.8, size=2)
        return float(gx), float(gy)

    def _get_obs(self) -> np.ndarray:
        return np.array([self._x, self._y, self._gx, self._gy], dtype=np.float32)

    def _compute_concept(self) -> np.ndarray:
        dx = self._gx - self._x
        dy = self._gy - self._y
        dist = float(np.sqrt(dx**2 + dy**2))
        norm = dist + 1e-8
        projected_velocity = (self._vx * dx + self._vy * dy) / norm
        approaching = 1.0 if projected_velocity > 0.0 else 0.0
        return np.array(
            [
                self._x,
                self._y,
                self._gx,
                self._gy,
                dist,
                self._vx,
                self._vy,
                approaching,
            ],
            dtype=np.float32,
        )


def make_hidden_velocity_env(n_envs: int = 4, seed: int = 0, **_) -> SyncVectorEnv:
    def _make(rank: int):
        def _init():
            return HiddenVelocityEnv(seed=seed + rank)

        return _init

    return SyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_hidden_velocity_env(seed: int = 0, **_) -> HiddenVelocityEnv:
    return HiddenVelocityEnv(seed=seed)
