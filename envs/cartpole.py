"""
cartpole.py — VisionCartPoleEnv wrapper.

Observation: dict with
  'images'      : (n_stack, ROWS, COLS)  — n_stack stacked grayscale frames
  'last_action' : (1,)

Concepts: [cart_pos, cart_vel, pole_angle, pole_ang_vel]  (all regression)
Ground truth is the raw physics state from CartPole-v1.

Flicker: with probability flicker_prob, frames are zeroed (black).
Concepts (physics state) are unaffected — only the visual observation is blanked.
This forces the agent to rely on temporal memory (GRU) to bridge gaps.
"""

from collections import deque
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None


class VisionCartPoleEnv(gym.Wrapper):
    """
    Wraps CartPole-v1 (rgb_array render mode) to provide:
      - n_stack grayscale image stack as primary observation
      - last_action in observation dict
      - get_concept() returning physics state (4 floats)
      - optional flicker: zero out frames with probability flicker_prob
        to stress-test temporal encoding (GRU must bridge blank frames)
    """

    ROWS = 160
    COLS = 240

    def __init__(self, env: gym.Env, ROWS: int = 160, COLS: int = 240,
                 img_stack: int = 1, flicker_prob: float = 0.25):
        super().__init__(env)
        if cv2 is None:
            raise ImportError(
                "VisionCartPoleEnv requires OpenCV (cv2). Install opencv-python in this environment."
            )
        self.ROWS = ROWS
        self.COLS = COLS
        self.img_stack = img_stack
        self.flicker_prob = flicker_prob

        self.observation_space = gym.spaces.Dict({
            "images": gym.spaces.Box(
                low=0, high=255,
                shape=(img_stack, ROWS, COLS),
                dtype=np.uint8,
            ),
            "last_action": gym.spaces.Box(
                low=np.array([0]), high=np.array([1]), dtype=np.uint8
            ),
        })

        self.task_types    = ["regression"] * 4
        self.num_classes   = [0, 0, 0, 0]
        self.concept_names = ["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"]
        # temporal concepts: cart_vel (index 1), pole_ang_vel (index 3)
        self.temporal_concepts = [1, 3]

        self.frames = deque(maxlen=img_stack)
        self.last_action = 0
        self.current_concept: Optional[np.ndarray] = None
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------

    def get_concept(self) -> np.ndarray:
        return self.current_concept.copy()


    @property
    def concept_reward_active(self) -> float:
        """1.0 always — all concepts are relevant at every step for CartPole."""
        return 1.0

    def reset(self, **kwargs):
        seed = kwargs.pop("seed", None)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        observation, info = self.env.reset(**kwargs)
        self.frames.clear()
        img = self._get_image()
        img = self._maybe_flicker(img)
        for _ in range(self.img_stack):
            self.frames.append(img)
        self.last_action = 0
        self.current_concept = np.array(observation, dtype=np.float32)
        obs = self._make_obs()
        info["concept"] = self.current_concept.copy()
        return obs, info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        img = self._get_image()
        img = self._maybe_flicker(img)
        self.frames.append(img)
        self.last_action = int(action)
        self.current_concept = np.array(observation, dtype=np.float32)
        obs = self._make_obs()
        if done or truncated:
            info["terminal_observation"] = obs
        info["concept"] = self.current_concept.copy()
        return obs, reward, done, truncated, info

    # ------------------------------------------------------------------

    def _get_image(self) -> np.ndarray:
        img = self.env.render()
        assert img is not None, "render() returned None — use render_mode='rgb_array'"
        img = cv2.resize(img, (self.COLS, self.ROWS))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def _maybe_flicker(self, frame: np.ndarray) -> np.ndarray:
        """With probability flicker_prob, return a zero (black) frame instead.
        Concepts (physics state) are unaffected — only the visual observation is blanked.
        This forces the agent to rely on temporal memory (GRU) to bridge gaps."""
        if self.flicker_prob > 0 and self._rng.random() < self.flicker_prob:
            return np.zeros_like(frame)
        return frame

    def _make_obs(self) -> dict:
        return {
            "images":      np.array(self.frames, dtype=np.uint8),
            "last_action": np.array([self.last_action], dtype=np.uint8),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_cartpole_env(n_envs: int = 4, seed: int = 0, n_stack: int = 1,
                       flicker_prob: float = 0.25) -> gym.Env:
    """
    Returns a vectorised VisionCartPoleEnv.
    Uses gymnasium's AsyncVectorEnv for parallel env stepping.
    n_stack=1 disables frame stacking (single frame, GRU handles temporal).
    flicker_prob=0.25 blanks 25% of frames to stress-test temporal memory.
    """
    from gymnasium.vector import AsyncVectorEnv

    def _make(rank: int):
        def _init():
            base = gym.make("CartPole-v1", render_mode="rgb_array")
            env  = VisionCartPoleEnv(base, img_stack=n_stack, flicker_prob=flicker_prob)
            env.reset(seed=seed + rank)
            return env
        return _init

    return AsyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_cartpole_env(seed: int = 0, n_stack: int = 1,
                              flicker_prob: float = 0.25) -> VisionCartPoleEnv:
    base = gym.make("CartPole-v1", render_mode="rgb_array")
    env  = VisionCartPoleEnv(base, img_stack=n_stack, flicker_prob=flicker_prob)
    env.reset(seed=seed)
    return env

# """
# cartpole.py — CartPoleStateEnv wrapper.

# Observation: flat Box(2,) — position-only [cart_pos, pole_angle]
#   Velocities (cart_vel, pole_ang_vel) are HIDDEN from observation and must be
#   inferred via temporal integration (GRU).

# Concepts: [cart_pos, cart_vel, pole_angle, pole_ang_vel]  (all regression)
# Ground truth is the raw physics state from CartPole-v1.
# """

# from typing import Optional

# import gymnasium as gym
# import numpy as np


# class CartPoleStateEnv(gym.Wrapper):
#     """
#     Wraps CartPole-v1 with position-only observations and physics-based concepts.
#     Temporal concepts (cart_vel, pole_ang_vel) are hidden from observation —
#     they must be inferred via temporal integration (GRU). No rendering.

#     Optional flicker: zero out the observation with probability flicker_prob
#     to stress-test temporal encoding (GRU must bridge blank steps).
#     Concepts (physics state) are unaffected — only the observation is blanked.
#     """

#     OBS_INDICES = [0, 2]  # cart_pos, pole_angle  (velocities hidden)

#     def __init__(self, env: gym.Env, flicker_prob: float = 0.25):
#         super().__init__(env)
#         self.flicker_prob = flicker_prob
#         self.observation_space = gym.spaces.Box(
#             low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
#         )

#         self.task_types    = ["regression"] * 4
#         self.num_classes   = [0, 0, 0, 0]
#         self.concept_names = ["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"]
#         # temporal concepts: cart_vel (index 1), pole_ang_vel (index 3)
#         self.temporal_concepts = [1, 3]

#         self.current_concept: Optional[np.ndarray] = None
#         self._rng = np.random.default_rng()

#     # ------------------------------------------------------------------

#     def get_concept(self) -> np.ndarray:
#         return self.current_concept.copy()

#     @property
#     def concept_reward_active(self) -> float:
#         """1.0 always — all concepts are relevant at every step for CartPole."""
#         return 1.0

#     def _pos_obs(self, full_obs: np.ndarray) -> np.ndarray:
#         return full_obs[self.OBS_INDICES].astype(np.float32)

#     def reset(self, **kwargs):
#         seed = kwargs.pop("seed", None)
#         if seed is not None:
#             self._rng = np.random.default_rng(seed)
#         observation, info = self.env.reset(**kwargs)
#         self.current_concept = np.array(observation, dtype=np.float32)
#         info["concept"] = self.current_concept.copy()
#         obs = self._maybe_flicker(self._pos_obs(observation))
#         return obs, info

#     def step(self, action):
#         observation, reward, done, truncated, info = self.env.step(action)
#         self.current_concept = np.array(observation, dtype=np.float32)
#         info["concept"] = self.current_concept.copy()
#         obs = self._maybe_flicker(self._pos_obs(observation))
#         return obs, reward, done, truncated, info

#     # ------------------------------------------------------------------

#     def _maybe_flicker(self, obs: np.ndarray) -> np.ndarray:
#         """With probability flicker_prob, return a zero vector instead.
#         Concepts (physics state) are unaffected — only the observation is blanked.
#         This forces the agent to rely on temporal memory (GRU) to bridge gaps."""
#         if self.flicker_prob > 0 and self._rng.random() < self.flicker_prob:
#             return np.zeros_like(obs)
#         return obs


# # ---------------------------------------------------------------------------
# # Factory
# # ---------------------------------------------------------------------------

# def make_cartpole_env(n_envs: int = 4, seed: int = 0,
#                        flicker_prob: float = 0.25, **_) -> gym.Env:
#     """
#     Returns a vectorised CartPoleStateEnv.
#     Uses gymnasium's AsyncVectorEnv for parallel env stepping.
#     """
#     from gymnasium.vector import AsyncVectorEnv

#     def _make(rank: int):
#         def _init():
#             base = gym.make("CartPole-v1")
#             env  = CartPoleStateEnv(base, flicker_prob=flicker_prob)
#             env.reset(seed=seed + rank)
#             return env
#         return _init

#     return AsyncVectorEnv([_make(i) for i in range(n_envs)])


# def make_single_cartpole_env(seed: int = 0,
#                               flicker_prob: float = 0.25, **_) -> CartPoleStateEnv:
#     base = gym.make("CartPole-v1")
#     env  = CartPoleStateEnv(base, flicker_prob=flicker_prob)
#     env.reset(seed=seed)
#     return env