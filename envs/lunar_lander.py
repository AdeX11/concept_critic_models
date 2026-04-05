"""
lunar_lander.py — LunarLanderConceptEnv.

Wraps gymnasium LunarLander-v2 (continuous=False, render_mode='rgb_array').

Observation: 4-frame stacked RGB rendered image [12, H, W].

Concepts:
  Index  Name               Type            Temporal?
  0      x_position         regression      static
  1      y_position         regression      static
  2      x_velocity         regression      TEMPORAL
  3      y_velocity         regression      TEMPORAL
  4      angle              regression      static
  5      angular_velocity   regression      TEMPORAL
  6      left_leg_contact   classification  static
  7      right_leg_contact  classification  static

Ground truth is read from the Box2D physics body (env.unwrapped.lander).
"""

from collections import deque

import cv2
import gymnasium as gym
import numpy as np


ROWS = 84
COLS = 84
IMG_STACK = 4
N_CONCEPTS = 8


class LunarLanderConceptEnv(gym.Wrapper):
    """
    LunarLander-v2 with 4-frame stacked image observations and physics-based concepts.
    """

    def __init__(self, env: gym.Env, rows: int = ROWS, cols: int = COLS, img_stack: int = IMG_STACK):
        super().__init__(env)
        self.rows = rows
        self.cols = cols
        self.img_stack = img_stack

        # Stack 4 RGB frames → [img_stack*3, rows, cols]
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(img_stack * 3, rows, cols),
            dtype=np.uint8,
        )

        self.task_types  = ["regression"] * 6 + ["classification"] * 2
        self.num_classes = [0] * 6 + [2, 2]
        self.concept_names = [
            "x_position", "y_position",
            "x_velocity", "y_velocity",
            "angle", "angular_velocity",
            "left_leg_contact", "right_leg_contact",
        ]
        # Temporal concepts: x_velocity (2), y_velocity (3), angular_velocity (5)
        self.temporal_concepts = [2, 3, 5]

        self.frames = deque(maxlen=img_stack)
        self._current_concept = np.zeros(N_CONCEPTS, dtype=np.float32)

    # ------------------------------------------------------------------

    def get_concept(self) -> np.ndarray:
        return self._current_concept.copy()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        frame = self._get_frame()
        for _ in range(self.img_stack):
            self.frames.append(frame)
        self._current_concept = self._read_physics_state(obs)
        info["concept"] = self._current_concept.copy()
        return self._stack_frames(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        frame = self._get_frame()
        self.frames.append(frame)
        self._current_concept = self._read_physics_state(obs)
        stacked = self._stack_frames()
        if done or truncated:
            info["terminal_observation"] = stacked
        info["concept"] = self._current_concept.copy()
        return stacked, reward, done, truncated, info

    # ------------------------------------------------------------------

    def _get_frame(self) -> np.ndarray:
        img = self.env.render()
        if img is None:
            img = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        img = cv2.resize(img, (self.cols, self.rows))
        return img  # [H, W, 3]

    def _stack_frames(self) -> np.ndarray:
        # [img_stack, H, W, 3] → [img_stack*3, H, W]
        frames = np.stack(list(self.frames), axis=0)    # [4, H, W, 3]
        return frames.transpose(0, 3, 1, 2).reshape(-1, self.rows, self.cols)  # [12, H, W]

    def _read_physics_state(self, gym_obs: np.ndarray) -> np.ndarray:
        """
        Read concepts directly from the Box2D physics state.
        LunarLander-v2 returns an 8-dim observation:
          [x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact]
        We directly use this vector (it IS the physics state).
        """
        if gym_obs is not None and len(gym_obs) >= 8:
            c = np.array([
                float(gym_obs[0]),  # x_position
                float(gym_obs[1]),  # y_position
                float(gym_obs[2]),  # x_velocity
                float(gym_obs[3]),  # y_velocity
                float(gym_obs[4]),  # angle
                float(gym_obs[5]),  # angular_velocity
                float(gym_obs[6]),  # left_leg_contact  (0 or 1)
                float(gym_obs[7]),  # right_leg_contact (0 or 1)
            ], dtype=np.float32)
            return c
        return np.zeros(N_CONCEPTS, dtype=np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_lunar_lander_env(n_envs: int = 4, seed: int = 0, n_stack: int = 4) -> gym.Env:
    from gymnasium.vector import SyncVectorEnv

    def _make(rank: int):
        def _init():
            env = gym.make("LunarLander-v2", render_mode="rgb_array")
            env = LunarLanderConceptEnv(env, img_stack=n_stack)
            env.reset(seed=seed + rank)
            return env
        return _init

    return SyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_lunar_lander_env(seed: int = 0, n_stack: int = 4) -> LunarLanderConceptEnv:
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    env = LunarLanderConceptEnv(env, img_stack=n_stack)
    env.reset(seed=seed)
    return env
