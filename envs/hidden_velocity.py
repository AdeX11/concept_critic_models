"""
hidden_velocity.py — HiddenVelocityEnv

A 2D continuous navigation task designed to stress-test temporal concept prediction.

Observation (4 floats): [x, y, goal_x, goal_y]   — NO velocity in obs
Hidden state:           [vx, vy]                   — never shown to policy

This guarantees that temporal concepts are genuinely unrecoverable from a single
frame: the same (x, y, goal_x, goal_y) obs can correspond to any velocity.

Concepts:
  Index  Name               Type            Derivable from single frame?
  ─────────────────────────────────────────────────────────────────────
  Static (vanilla CBM can predict these):
  0      x_pos              regression      YES  (directly in obs)
  1      y_pos              regression      YES  (directly in obs)
  2      goal_x             regression      YES  (directly in obs)
  3      goal_y             regression      YES  (directly in obs)
  4      dist_to_goal       regression      YES  (computed from obs)

  Temporal (vanilla CBM cannot predict these):
  5      x_velocity         regression      NO   (hidden state)
  6      y_velocity         regression      NO   (hidden state)
  7      approaching_goal   classification  NO   (requires vel + goal direction)

Dynamics:
  vx ← FRICTION * vx + thrust_x[action]
  vy ← FRICTION * vy + thrust_y[action]
  x  ← clip(x + vx, -BOUNDS, +BOUNDS)    # velocity zeroed on wall contact
  y  ← clip(y + vy, -BOUNDS, +BOUNDS)

Reward:
  +1.0   reaching goal (goal teleports to a new random location)
  -0.01  per step (time pressure)
  -0.05  hitting a wall (velocity component zeroed)
"""

import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv


BOUNDS     = 1.0
FRICTION   = 0.85    # strong momentum — agent must decelerate in advance
THRUST     = 0.08    # force per action step
GOAL_RADIUS = 0.12
MAX_STEPS  = 400
N_CONCEPTS = 8

# action → (thrust_x, thrust_y)
_THRUST_MAP = np.array([
    [ 0.0,  0.0],   # 0: coast
    [ THRUST,  0.0],   # 1: right
    [-THRUST,  0.0],   # 2: left
    [ 0.0,  THRUST],   # 3: up
    [ 0.0, -THRUST],   # 4: down
], dtype=np.float32)


class HiddenVelocityEnv(gym.Env):
    """
    2D navigation with hidden velocity — designed so temporal concepts
    are unrecoverable by vanilla CBM (no velocity in observation).
    """

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low  = np.array([-BOUNDS, -BOUNDS, -BOUNDS, -BOUNDS], dtype=np.float32),
            high = np.array([ BOUNDS,  BOUNDS,  BOUNDS,  BOUNDS], dtype=np.float32),
            dtype = np.float32,
        )
        self.action_space = gym.spaces.Discrete(5)

        # Concept metadata (consumed by policy_kwargs builder in train.py)
        self.task_types = (
            ["regression"] * 5 +    # x_pos, y_pos, goal_x, goal_y, dist_to_goal
            ["regression"] * 2 +    # x_velocity, y_velocity
            ["classification"]      # approaching_goal (2 classes)
        )
        self.num_classes  = [0] * 7 + [2]
        self.concept_names = [
            "x_pos", "y_pos", "goal_x", "goal_y", "dist_to_goal",
            "x_velocity", "y_velocity",
            "approaching_goal",
        ]
        # Indices of temporal concepts (for plot_results.py splitting)
        self.temporal_concepts = [5, 6, 7]

        self._rng = np.random.default_rng(seed)
        self._x = self._y = self._vx = self._vy = 0.0
        self._gx = self._gy = 0.0
        self._steps = 0
        self.current_concept = np.zeros(N_CONCEPTS, dtype=np.float32)

    # ------------------------------------------------------------------
    # gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Random start position and random initial velocity
        self._x, self._y = self._rng.uniform(-BOUNDS * 0.8, BOUNDS * 0.8, size=2)
        max_v = THRUST * 3
        self._vx, self._vy = self._rng.uniform(-max_v, max_v, size=2)
        self._gx, self._gy = self._spawn_goal()
        self._steps = 0

        self.current_concept = self._compute_concept()
        obs = self._get_obs()
        return obs, {"concept": self.current_concept.copy()}

    def step(self, action: int):
        thrust = _THRUST_MAP[int(action)]
        self._vx = FRICTION * self._vx + thrust[0]
        self._vy = FRICTION * self._vy + thrust[1]

        new_x = self._x + self._vx
        new_y = self._y + self._vy

        reward = -0.01  # time penalty
        hit_wall = False

        # Wall collision: zero the relevant velocity, clamp position
        if new_x < -BOUNDS or new_x > BOUNDS:
            self._vx = 0.0
            new_x = np.clip(new_x, -BOUNDS, BOUNDS)
            hit_wall = True
        if new_y < -BOUNDS or new_y > BOUNDS:
            self._vy = 0.0
            new_y = np.clip(new_y, -BOUNDS, BOUNDS)
            hit_wall = True

        if hit_wall:
            reward -= 0.05

        self._x, self._y = float(new_x), float(new_y)
        self._steps += 1

        # Goal reached?
        dist = np.sqrt((self._x - self._gx) ** 2 + (self._y - self._gy) ** 2)
        if dist < GOAL_RADIUS:
            reward += 1.0
            self._gx, self._gy = self._spawn_goal()  # new random goal

        terminated = False
        truncated  = self._steps >= MAX_STEPS

        self.current_concept = self._compute_concept()
        obs = self._get_obs()
        return obs, reward, terminated, truncated, {"concept": self.current_concept.copy()}

    def get_concept(self):
        return self.current_concept.copy()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _spawn_goal(self) -> tuple:
        """Spawn goal at least 0.3 away from current position."""
        for _ in range(50):
            gx, gy = self._rng.uniform(-BOUNDS * 0.85, BOUNDS * 0.85, size=2)
            if np.sqrt((self._x - gx) ** 2 + (self._y - gy) ** 2) > 0.3:
                return float(gx), float(gy)
        return float(self._rng.uniform(-0.8, 0.8)), float(self._rng.uniform(-0.8, 0.8))

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [self._x, self._y, self._gx, self._gy], dtype=np.float32
        )

    def _compute_concept(self) -> np.ndarray:
        dist = float(np.sqrt((self._x - self._gx) ** 2 + (self._y - self._gy) ** 2))

        # approaching_goal: dot(velocity, goal_direction) > 0
        dx, dy = self._gx - self._x, self._gy - self._y
        norm    = np.sqrt(dx ** 2 + dy ** 2) + 1e-8
        dot     = (self._vx * dx + self._vy * dy) / norm
        approaching = 1.0 if dot > 0.0 else 0.0

        return np.array([
            self._x,          # 0: x_pos
            self._y,          # 1: y_pos
            self._gx,         # 2: goal_x
            self._gy,         # 3: goal_y
            dist,             # 4: dist_to_goal
            self._vx,         # 5: x_velocity   ← TEMPORAL
            self._vy,         # 6: y_velocity   ← TEMPORAL
            approaching,      # 7: approaching_goal ← TEMPORAL
        ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_hidden_velocity_env(n_envs: int, seed: int) -> SyncVectorEnv:
    def _make(i):
        def _init():
            return HiddenVelocityEnv(seed=seed + i)
        return _init
    return SyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_hidden_velocity_env(seed: int) -> HiddenVelocityEnv:
    return HiddenVelocityEnv(seed=seed)
