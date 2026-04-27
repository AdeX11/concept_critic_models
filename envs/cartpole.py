"""
cartpole.py — VisionCartPoleEnv wrapper with Temporal Occlusion (Flicker).

Observation: dict with
  'images'      : (N_STACK, ROWS, COLS) — grayscale frames (zeroed during blackout)
  'last_action' : (1,)

Concepts: [cart_pos, cart_vel, pole_angle, pole_ang_vel]
"""

from collections import deque
from typing import Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np

class VisionCartPoleEnv(gym.Wrapper):
    """
    Wraps CartPole-v1 to provide:
      - N-frame grayscale image stack
      - last_action in observation dict
      - Temporal Occlusion: Randomly 'blacks out' frames to test GRU memory.
    """

    def __init__(
        self, 
        env: gym.Env, 
        ROWS: int = 160, 
        COLS: int = 240, 
        img_stack: int = 4,
        flicker_prob: float = 0.0,
        blackout_duration: int = 1
    ):
        super().__init__(env)
        self.ROWS = ROWS
        self.COLS = COLS
        self.img_stack = img_stack
        
        # Memory Task Parameters
        self.flicker_prob = flicker_prob
        self.blackout_duration = blackout_duration
        self.blackout_counter = 0

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
        self.temporal_concepts = [1, 3]

        self.frames = deque(maxlen=img_stack)
        self.last_action = 0
        self.current_concept: Optional[np.ndarray] = None
        # All steps are concept-relevant (no hidden junctions like TMaze)
        self.concept_reward_active = True

    def get_concept(self) -> np.ndarray:
        return self.current_concept.copy()

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.frames.clear()
        self.blackout_counter = 0 
        
        img = self._get_image()
        for _ in range(self.img_stack):
            self.frames.append(img)
            
        self.last_action = 0
        self.current_concept = np.array(observation, dtype=np.float32)
        return self._make_obs(), info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        
        # Capture ground truth physics BEFORE potential occlusion
        self.current_concept = np.array(observation, dtype=np.float32)
        
        img = self._get_image()
        self.frames.append(img)
        self.last_action = int(action)

        # Logic for triggering a new blackout
        if self.blackout_counter == 0:
            if np.random.rand() < self.flicker_prob:
                self.blackout_counter = self.blackout_duration

        obs = self._make_obs()
        
        # Countdown the blackout duration
        if self.blackout_counter > 0:
            self.blackout_counter -= 1

        if done or truncated:
            info["terminal_observation"] = obs
            
        info["concept"] = self.current_concept.copy()
        # Useful for analyzing Concept Error during blind spots
        info["is_blackout"] = self.blackout_counter > 0 
        
        return obs, reward, done, truncated, info

    def _get_image(self) -> np.ndarray:
        img = self.env.render()
        img = cv2.resize(img, (self.COLS, self.ROWS))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def _make_obs(self) -> dict:
        img_data = np.array(self.frames, dtype=np.uint8)
        
        # If blackout is active, overwrite the stack with zeros
        if self.blackout_counter > 0:
            img_data = np.zeros_like(img_data)
            
        return {
            "images":      img_data,
            "last_action": np.array([self.last_action], dtype=np.uint8),
        }


# ---------------------------------------------------------------------------
# Factory (Keeping original names)
# ---------------------------------------------------------------------------

def make_cartpole_env(
    n_envs: int = 4, 
    seed: int = 0, 
    n_stack: int = 4, 
    flicker_p: float = 0.0, 
    duration: int = 1
) -> gym.Env:
    """
    Returns a vectorized VisionCartPoleEnv.
    """
    from gymnasium.vector import AsyncVectorEnv

    def _make(rank: int):
        def _init():
            base = gym.make("CartPole-v1", render_mode="rgb_array")
            env  = VisionCartPoleEnv(
                base, 
                img_stack=n_stack, 
                flicker_prob=flicker_p, 
                blackout_duration=duration
            )
            env.reset(seed=seed + rank)
            return env
        return _init

    return AsyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_cartpole_env(
    seed: int = 0, 
    n_stack: int = 4, 
    flicker_p: float = 0.0, 
    duration: int = 1
) -> VisionCartPoleEnv:
    """
    Returns a single VisionCartPoleEnv.
    """
    base = gym.make("CartPole-v1", render_mode="rgb_array")
    env  = VisionCartPoleEnv(
        base, 
        img_stack=n_stack, 
        flicker_prob=flicker_p, 
        blackout_duration=duration
    )
    env.reset(seed=seed)
    return env


# class VisionCartPoleEnv(gym.Wrapper):
#     """
#     Wraps CartPole-v1 (rgb_array render mode) to provide:
#       - 4-frame grayscale image stack as primary observation
#       - last_action in observation dict
#       - get_concept() returning physics state (4 floats)
#     """

#     ROWS = 160
#     COLS = 240
#     IMG_STACK = 4

#     def __init__(self, env: gym.Env, ROWS: int = 160, COLS: int = 240, img_stack: int = 4):
#         super().__init__(env)
#         self.ROWS = ROWS
#         self.COLS = COLS
#         self.img_stack = img_stack

#         self.observation_space = gym.spaces.Dict({
#             "images": gym.spaces.Box(
#                 low=0, high=255,
#                 shape=(img_stack, ROWS, COLS),
#                 dtype=np.uint8,
#             ),
#             "last_action": gym.spaces.Box(
#                 low=np.array([0]), high=np.array([1]), dtype=np.uint8
#             ),
#         })

#         self.task_types    = ["regression"] * 4
#         self.num_classes   = [0, 0, 0, 0]
#         self.concept_names = ["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"]
#         # temporal concepts: cart_vel (index 1), pole_ang_vel (index 3)
#         self.temporal_concepts = [1, 3]

#         self.frames = deque(maxlen=img_stack)
#         self.last_action = 0
#         self.current_concept: Optional[np.ndarray] = None

#     # ------------------------------------------------------------------

#     def get_concept(self) -> np.ndarray:
#         return self.current_concept.copy()

#     def reset(self, **kwargs):
#         observation, info = self.env.reset(**kwargs)
#         self.frames.clear()
#         img = self._get_image()
#         for _ in range(self.img_stack):
#             self.frames.append(img)
#         self.last_action = 0
#         self.current_concept = np.array(observation, dtype=np.float32)
#         obs = self._make_obs()
#         return obs, info

#     def step(self, action):
#         observation, reward, done, truncated, info = self.env.step(action)
#         img = self._get_image()
#         self.frames.append(img)
#         self.last_action = int(action)
#         self.current_concept = np.array(observation, dtype=np.float32)
#         obs = self._make_obs()
#         if done or truncated:
#             info["terminal_observation"] = obs
#         info["concept"] = self.current_concept.copy()
#         return obs, reward, done, truncated, info

#     # ------------------------------------------------------------------

#     def _get_image(self) -> np.ndarray:
#         img = self.env.render()
#         assert img is not None, "render() returned None — use render_mode='rgb_array'"
#         img = cv2.resize(img, (self.COLS, self.ROWS))
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         return img

#     def _make_obs(self) -> dict:
#         return {
#             "images":      np.array(self.frames, dtype=np.uint8),
#             "last_action": np.array([self.last_action], dtype=np.uint8),
#         }


# # ---------------------------------------------------------------------------
# # Factory
# # ---------------------------------------------------------------------------

# def make_cartpole_env(n_envs: int = 4, seed: int = 0, n_stack: int = 4) -> gym.Env:
#     """
#     Returns a vectorised VisionCartPoleEnv.
#     Uses gymnasium's AsyncVectorEnv for parallel env stepping.
#     n_stack=1 disables frame stacking (single frame).
#     """
#     from gymnasium.vector import AsyncVectorEnv

#     def _make(rank: int):
#         def _init():
#             base = gym.make("CartPole-v1", render_mode="rgb_array")
#             env  = VisionCartPoleEnv(base, img_stack=n_stack)
#             env.reset(seed=seed + rank)
#             return env
#         return _init

#     return AsyncVectorEnv([_make(i) for i in range(n_envs)])


# def make_single_cartpole_env(seed: int = 0, n_stack: int = 4) -> VisionCartPoleEnv:
#     base = gym.make("CartPole-v1", render_mode="rgb_array")
#     env  = VisionCartPoleEnv(base, img_stack=n_stack)
#     env.reset(seed=seed)
#     return env
