"""
mountain_car.py — MountainCarConceptEnv

Position-only observation with velocity as the single hidden temporal concept.

Observation (1-dim): [position]
Concepts   (1-dim):  [velocity]
  - velocity: temporal — NOT in obs, NOT in reward, corr(pos,vel)≈0.06
    The only way to infer it is by integrating position history.

Reward: shaped with a position bonus to make the task tractable for PPO.
  r = r_orig + position_bonus
  No velocity terms — velocity is not leaked through rewards.
"""

import gymnasium as gym
import numpy as np


N_CONCEPTS = 1


class MountainCarConceptEnv(gym.Wrapper):
    """
    MountainCar-v0 with position-only observation and velocity as the
    single temporal concept.
    """

    POS_MIN = -1.2
    POS_MAX =  0.6

    def __init__(self, env: gym.Env, reward_shaping: float = 3.0):
        super().__init__(env)
        self.reward_shaping = reward_shaping

        self.observation_space = gym.spaces.Box(
            low=np.array([self.POS_MIN], dtype=np.float32),
            high=np.array([self.POS_MAX], dtype=np.float32),
        )

        self.task_types    = ["regression"]
        self.num_classes   = [0]
        self.concept_names = ["velocity"]
        self.temporal_concepts = [0]

        self.current_concept = np.zeros(N_CONCEPTS, dtype=np.float32)

    def get_concept(self) -> np.ndarray:
        return self.current_concept.copy()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_concept = np.array([obs[1]], dtype=np.float32)
        info["concept"] = self.current_concept.copy()
        return np.array([obs[0]], dtype=np.float32), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.current_concept = np.array([obs[1]], dtype=np.float32)

        norm_pos = (obs[0] - self.POS_MIN) / (self.POS_MAX - self.POS_MIN)
        reward += self.reward_shaping * norm_pos

        info["concept"] = self.current_concept.copy()
        return np.array([obs[0]], dtype=np.float32), reward, done, truncated, info


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_mountain_car_env(n_envs: int = 4, seed: int = 0, **_) -> gym.Env:
    from gymnasium.vector import AsyncVectorEnv

    def _make(rank: int):
        def _init():
            env = gym.make("MountainCar-v0")
            env = MountainCarConceptEnv(env)
            env.reset(seed=seed + rank)
            return env
        return _init

    return AsyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_mountain_car_env(seed: int = 0, **_) -> MountainCarConceptEnv:
    env = gym.make("MountainCar-v0")
    env = MountainCarConceptEnv(env)
    env.reset(seed=seed)
    return env
