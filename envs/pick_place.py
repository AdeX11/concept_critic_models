"""
panda_gym_triple_stack.py — PandaTripleStackConceptEnv.

Wraps panda-gym with custom 17D observations and 20D concepts.

OBSERVATION SPACE (14D) - VELOCITY HIDDEN:
  - ee_pos (3), obj1_pos (3), obj2_pos (3), goal_pos (3)
  - gripper_width (1), any_contact (1)

Concepts (17D):
  0-2:   ee_to_obj1 (xyz)
  3-5:   ee_to_obj2 (xyz)
  6-8:  obj1_to_obj2 (xyz) - Stack 1 Alignment
  9-11: obj2_to_obj3 (xyz) - Stack 2 Alignment
  12:    gripper_opening (reg)
  13:    any_contact (class)
  14:    is_obj_grasped (class) - Logic: width < block & contact
  15:    ee_speed (reg)         - TEMPORAL (Inferred)
  16:    tower_height (reg)     - TEMPORAL (Z-max of blocks)
"""

import argparse
import numpy as np
import gymnasium as gym
import panda_gym
from typing import Optional, Dict, Any
import time
from gymnasium.vector import SyncVectorEnv

N_CONCEPTS = 17  

# --- Action Wrapper ---
class DiscretizePandaActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(7)
        self._gripper_state = 1.0  

    @property
    def task_types(self): return self.env.task_types
    @property
    def num_classes(self): return self.env.num_classes
    @property
    def concept_names(self): return self.env.concept_names
    @property
    def temporal_concepts(self): return getattr(self.env, "temporal_concepts", None)
    def get_concept(self): return self.env.get_concept()

    def action(self, act: int) -> np.ndarray:
        ctrl = np.zeros(4, dtype=np.float32)
        scale = 0.03 # Smoother, more deliberate movement
        
        if act == 0: ctrl[0] = scale    # +X
        elif act == 1: ctrl[0] = -scale # -X
        elif act == 2: ctrl[1] = scale  # +Y
        elif act == 3: ctrl[1] = -scale # -Y
        elif act == 4: ctrl[2] = scale  # +Z
        elif act == 5: ctrl[2] = -scale # -Z
        elif act == 6: self._gripper_state *= -1.0 
        
        ctrl[3] = self._gripper_state
        return ctrl

# --- Core Concept Wrapper ---
class PandaStackConceptEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        self.task_types = (["regression"] * 12 + ["regression", "classification", 
                           "classification", "regression", "regression"])
        self.num_classes = [0]*13 + [2, 2] + [0, 0]
        self.concept_names = [
            "ee_obj1_x", "ee_obj1_y", "ee_obj1_z", "ee_obj2_x", "ee_obj2_y", "ee_obj2_z",
            "obj1_obj2_x", "obj1_obj2_y", "obj1_obj2_z", "goal_x", "goal_y", "goal_z",
            "gripper_width", "any_contact", "is_grasped", "ee_speed", "tower_height"
        ]
        self.current_concept = np.zeros(N_CONCEPTS, dtype=np.float32)
        self._prev_dist = 0.0

    def _get_physics_state(self):
        sim = self.unwrapped.sim
        robot = self.unwrapped.robot
        ee_pos = robot.get_ee_position()
        obj1_pos = sim.get_base_position("object1")
        obj2_pos = sim.get_base_position("object2")
        
        any_c = 0.0
        is_grasped = 0.0
        g_width = robot.get_fingers_width()
        
        for name in ["object1", "object2"]:
            t_id = sim._bodies_idx[name]
            conts = sim.physics_client.getContactPoints(bodyA=sim._bodies_idx["panda"], bodyB=t_id)
            if len(conts) > 0:
                any_c = 1.0
                if g_width < 0.045: is_grasped = 1.0
        
        return {
            "ee_pos": ee_pos, "ee_vel": robot.get_ee_velocity(),
            "obj1": obj1_pos, "obj2": obj2_pos, "goal": self.unwrapped.task.goal,
            "g_width": g_width, "any_contact": any_c, "is_grasped": is_grasped,
            "tower_h": max([obj1_pos[2], obj2_pos[2]]) - 0.4
        }

    def _assemble_obs_and_concepts(self, s):
        # Slice the goal: s["goal"] is (6,), we want the first (3,)
        goal_3d = s["goal"][:3] 

        # 17D Concept Vector
        self.current_concept = np.concatenate([
            s["obj1"]-s["ee_pos"], s["obj2"]-s["ee_pos"], 
            s["obj2"]-s["obj1"], goal_3d,
            [s["g_width"], s["any_contact"], s["is_grasped"], 
             np.linalg.norm(s["ee_vel"]), s["tower_h"]]
        ]).astype(np.float32)
        
        # 14D OBSERVATION: Now with matching shapes
        obs = np.concatenate([
            s["ee_pos"],             # [0:3]
            s["obj1"] - s["ee_pos"], # [3:6]
            s["obj2"] - s["ee_pos"], # [6:9]
            goal_3d - s["ee_pos"],    # [9:12] - This was the (6,) vs (3,) error!
            [s["g_width"], s["any_contact"]] # [12, 13]
        ]).astype(np.float32)
        
        return obs

    def step(self, action):
        _, _, terminated, truncated, info = self.env.step(action)
        state = self._get_physics_state()
        dist_ee_obj1 = np.linalg.norm(state["obj1"] - state["ee_pos"])

        # 1. Progress Reward
        reward = (self._prev_dist - dist_ee_obj1) * 50.0
        self._prev_dist = dist_ee_obj1

        # 2. Proximity Bonus (Stops the hovering/shaking)
        if dist_ee_obj1 < 0.05:
            reward += 1.0 * (1.0 - (dist_ee_obj1 / 0.05))

        # 3. Grasping & Stacking
        if state["is_grasped"] > 0:
            reward += 2.0
            reward += state["tower_h"] * 50.0

        # 4. Anti-Bully & Out-of-Bounds
        if dist_ee_obj1 < 0.08 and np.linalg.norm(state["ee_vel"]) > 0.5:
            reward -= 0.1 # Soft penalty for coming in too hot
            
        if state["obj1"][2] < -5.0 or state["obj2"][2] < -5.0:
            terminated = True # Episode ends if blocks fly off table
            reward -= 5.0

        obs = self._assemble_obs_and_concepts(state)
        info["concept"] = self.current_concept.copy()
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        state = self._get_physics_state()
        self._prev_dist = np.linalg.norm(state["obj1"] - state["ee_pos"])
        return self._assemble_obs_and_concepts(state), {"concept": self.current_concept.copy()}

# --- Headless/State Wrapper (Corrected to Sync properly) ---
class PickPlaceStateEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        self.current_concept = np.zeros(N_CONCEPTS, dtype=np.float32)

    # Add these properties to pass through from PandaStackConceptEnv
    @property
    def task_types(self):
        return self.env.task_types

    @property
    def num_classes(self):
        return self.env.num_classes

    @property
    def concept_names(self):
        return self.env.concept_names

    @property
    def temporal_concepts(self):
        return getattr(self.env, "temporal_concepts", None)
    
    def get_concept(self) -> np.ndarray:
        return self.current_concept.copy()

    def _sync_concept(self, info):
        if info and "concept" in info:
            self.current_concept = np.array(info["concept"], dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._sync_concept(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._sync_concept(info)
        return obs, reward, terminated, truncated, info

# --- Factory Functions (Standard Names) ---
def make_panda_pickplace_env(n_envs: int = 4, seed: int = 0):
    """Basic Vectorized Env (Visual/Standard)"""
    def _init():
        base = gym.make("PandaStack-v3", render_mode="rgb_array", max_episode_steps=500)
        env = PandaStackConceptEnv(base)
        env = DiscretizePandaActionWrapper(env)
        return env
    
    return SyncVectorEnv([_init for _ in range(n_envs)])

def make_single_panda_env(seed: int = 0):
    """Single Env with Human Rendering for testing"""
    base = gym.make("PandaStack-v3", render_mode="human", max_episode_steps=500)
    # Action wrapper must be inside Concept wrapper so Concept can see the discretized results if needed
    env = PandaStackConceptEnv(base)
    env = DiscretizePandaActionWrapper(env)
    return env

def make_panda_pickplace_state_env(n_envs: int = 4, seed: int = 0) -> gym.Env:
    """Vectorized Env for your Architecture/PPO Training"""
    def _init():
        # 1. Base Physics
        base = gym.make("PandaStack-v3", max_episode_steps=500)
        # 3. Physics Concept Logic (Rewards & Concept Vectors)
        env = PandaStackConceptEnv(base)
        # 4. Architecture / State Logic (Syncing Concepts for the Agent)
        env = PickPlaceStateEnv(env)
        env = DiscretizePandaActionWrapper(env)
        return env
    
    # Using SyncVectorEnv eliminates the "NoneType" pipe errors
    return SyncVectorEnv([_init for _ in range(n_envs)])

def make_single_panda_pickplace_state_env(seed: int = 0) -> gym.Env:
    """Single Env version of the State Env"""
    base = gym.make("PandaStack-v3", max_episode_steps=500)
    env = PandaStackConceptEnv(base)
    env = PickPlaceStateEnv(env)
    env = DiscretizePandaActionWrapper(env)
    return env

# --- Smoke Test ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--state", action="store_true")
    args = parser.parse_args()

    if args.render:
        env = make_single_panda_env()
        obs, _ = env.reset()
        
        # We want to match the environment's internal 40Hz clock
        step_duration = 1.0 / 40.0 
        
        for i in range(1000): # More steps to give you time to watch
            start_time = time.time()
            
            # Sample or provide a specific action
            action = env.action_space.sample()
            obs, rew, term, trunc, info = env.step(action)
            
            if i % 20 == 0:
                print(f"Step {i:3} | Height: {info['concept'][16]:.4f} | Grasped: {info['concept'][14]}")
            
            if term or trunc:
                print("Episode finished, resetting...")
                env.reset()

            # --- THE SPEED FIX ---
            # Calculate how much time we have left to wait
            elapsed = time.time() - start_time
            if elapsed < step_duration:
                time.sleep(step_duration - elapsed)
            # ---------------------

        env.close()

    if args.state:
        s_env = make_single_panda_pickplace_state_env()
        obs, _ = s_env.reset()
        print(f"State Obs Shape: {obs.shape}") # Expect 14
        for j in range(5):
            obs, rew, term, trunc, info = s_env.step(s_env.action_space.sample())
            print(f"State Concept Vector Size: {len(s_env.get_concept())}") # Expect 17
        s_env.close()
