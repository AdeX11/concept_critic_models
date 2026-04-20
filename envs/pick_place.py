"""
panda_gym_pickplace.py — PandaPickPlaceConceptEnv.

Wraps panda-gym's PickAndPlace-v3 with custom 11D observations 
and force-based breakability concepts.

OBSERVATION SPACE (11D) - VELOCITY HIDDEN:
  - ee_pos (3)
  - obj_pos (3)
  - goal_pos (3)
  - gripper_width (1)
  - contact (1)

Concepts (11D):
  Index  Name                  Type           Temporal?
  0      ee_to_obj_dx          regression     static
  1      ee_to_obj_dy          regression     static
  2      ee_to_obj_dz          regression     static
  3      gripper_opening       regression     static
  4      contact               classification static/partial
  5      crush_risk            regression     TEMPORAL   (core hidden concept)
  6      is_broken             classification TEMPORAL   (failure signal)
  7      ee_speed              regression     TEMPORAL   (must be inferred from hidden pos changes)
  8      obj_to_goal_dx        regression     static
  9      obj_to_goal_dy        regression     static
  10     obj_to_goal_dz        regression     static
"""

import argparse
import numpy as np
import gymnasium as gym
import panda_gym
from typing import Optional, Dict, Any
import time

N_CONCEPTS = 11  

class DiscretizePandaActionWrapper(gym.ActionWrapper):
    """Wraps a continuous Panda environment to act as a discrete environment."""
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
        if act == 0: ctrl[0] = 1.0      
        elif act == 1: ctrl[0] = -1.0   
        elif act == 2: ctrl[1] = 1.0    
        elif act == 3: ctrl[1] = -1.0   
        elif act == 4: ctrl[2] = 1.0    
        elif act == 5: ctrl[2] = -1.0   
        elif act == 6: self._gripper_state *= -1.0 
        ctrl[3] = self._gripper_state
        return ctrl

class PandaPickPlaceConceptEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, max_force_threshold: float = 50.0, force_range: tuple = (20.0, 100.0)):
        super().__init__(env)
        self.max_force_threshold = max_force_threshold
        self.force_range = force_range 
        self.current_max_force = 50.0  
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        
        self.task_types = [
            "regression", "regression", "regression",  # ee_to_obj (x, y, z)
            "regression",                              # gripper
            "classification",                          # contact
            "regression",                              # crush_risk
            "classification",                          # is_broken
            "regression",                              # ee_speed
            "regression", "regression", "regression"   # obj_to_goal (x, y, z)
        ]
        self.num_classes = [0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0] 
        
        self.concept_names = [
            "ee_to_obj_dx", "ee_to_obj_dy", "ee_to_obj_dz",
            "gripper_opening",
            "contact",
            "crush_risk",
            "is_broken",
            "ee_speed",
            "obj_to_goal_dx", "obj_to_goal_dy", "obj_to_goal_dz"
        ]
        
        # 5: crush_risk, 6: is_broken, 7: ee_speed 
        self.temporal_concepts = [5, 6, 7] 
        
        self.current_concept = np.zeros(N_CONCEPTS, dtype=np.float32)
        self._previous_dist = None
        self._prev_goal_dist = None
        
        # Temporal damage tracking
        self.accumulated_damage = 0.0
        self._is_broken_latched = False

    def _get_physics_state(self):
        sim = self.unwrapped.sim      
        robot = self.unwrapped.robot  

        ee_pos = robot.get_ee_position()
        ee_vel = robot.get_ee_velocity()
        obj_pos = sim.get_base_position("object")          
        goal_pos = self.unwrapped.task.goal                
        gripper_width = robot.get_fingers_width()

        # ---------------------------------------------------------
        # THE FIX: Targeted PyBullet Contact Polling
        # ---------------------------------------------------------
        try:
            # Grab the specific integer IDs PyBullet uses for these objects
            robot_id = sim._bodies_idx["panda"]
            object_id = sim._bodies_idx["object"]

            # Ask PyBullet ONLY for contacts between the robot and the block
            contact_points = sim.physics_client.getContactPoints(bodyA=robot_id, bodyB=object_id)
            finger_contacts = [c for c in contact_points if c[3] in [9, 10]]

            # Sum the normal forces (Index 9) of ONLY the finger contacts
            max_force = sum(c[9] for c in finger_contacts) if finger_contacts else 0.0

        except KeyError:
            # Fallback in case body names differ slightly or the object hasn't spawned yet
            print("Warning: Could not find body IDs for contact polling. Defaulting max_force to 0.")
            max_force = 0.0
        # ---------------------------------------------------------

        if not self._is_broken_latched:
            # 1. Impact Risk: Force amplified by hidden velocity
            speed = np.linalg.norm(ee_vel)
            impact_damage = max_force * speed * 2.0  
            
            # 2. Fatigue Risk: Accumulate excess force over time
            # Only accumulate if force is above a "safe" handling threshold (e.g., 10.0)
            excess_force = max(0.0, max_force - 10.0)
            self.accumulated_damage += excess_force * 0.05 
            
            # Calculate total risk against THIS episode's specific toughness
            current_risk = (self.accumulated_damage + impact_damage) / self.current_max_force
            crush_risk = min(current_risk, 1.0)
            
            if crush_risk >= 1.0:
                self._is_broken_latched = True
        else:
            crush_risk = 1.0  # Stays maxed out once broken

        return {
            "ee_pos": ee_pos, "ee_vel": ee_vel,
            "obj_pos": obj_pos, "goal_pos": goal_pos,
            "gripper_width": gripper_width,
            "contact": float(len(finger_contacts) > 0),
            "crush_risk": crush_risk, 
            "is_broken": float(self._is_broken_latched)
        }

    def _assemble_obs_and_concepts(self, state):
        ee_to_obj_vec = state["obj_pos"] - state["ee_pos"]
        obj_to_goal_vec = state["goal_pos"] - state["obj_pos"]
        
        speed = np.linalg.norm(state["ee_vel"])
        
        dist = np.linalg.norm(ee_to_obj_vec) 
        
        # Concepts (11D) 
        self.current_concept = np.array([
            ee_to_obj_vec[0], ee_to_obj_vec[1], ee_to_obj_vec[2],
            state["gripper_width"],
            state["contact"],
            state["crush_risk"],  
            state["is_broken"],   
            speed,                
            obj_to_goal_vec[0], obj_to_goal_vec[1], obj_to_goal_vec[2]
        ], dtype=np.float32)
        
        # Observation (11D) - Velocity has been strictly removed
        obs = np.concatenate([
            state["ee_pos"],              
            state["obj_pos"], state["goal_pos"],            
            [state["gripper_width"]], [state["contact"]],           
        ]).astype(np.float32)
        
        return obs, dist

    def step(self, action):
        _, _, terminated, truncated, info = self.env.step(action)
        
        state = self._get_physics_state()
        obs, dist = self._assemble_obs_and_concepts(state)
        dist_obj_goal = np.linalg.norm(state["obj_pos"] - state["goal_pos"])
        
        ee_progress = (self._previous_dist - dist) if self._previous_dist is not None else 0.0
        goal_progress = (self._prev_goal_dist - dist_obj_goal) if self._prev_goal_dist is not None else 0.0
        reward = (ee_progress * 50.0) + (goal_progress * 150.0)

        if state["contact"] > 0:
            if not getattr(self, "_has_touched", False):
                reward += 50.0  
                self._has_touched = True
            reward += 0.1  

        if state["is_broken"] > 0:
            if not getattr(self, "_has_broken", False):
                reward -= 50.0  
                self._has_broken = True
            
        elif dist_obj_goal < 0.05:
            reward += 10.0  
            
        self._previous_dist = dist
        self._prev_goal_dist = dist_obj_goal
        
        info["concept"] = self.current_concept.copy()
        info["ground_truth_concepts"] = dict(zip(self.concept_names, self.current_concept))
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self._has_touched = False
        self._has_broken = False  
        
        # Reset temporal tracking for the new episode
        self.accumulated_damage = 0.0
        self._is_broken_latched = False
        
        np_random = self.unwrapped.np_random if hasattr(self.unwrapped, "np_random") else np.random
        # The object gets a new random toughness here!
        self.current_max_force = np_random.uniform(self.force_range[0], self.force_range[1])
        
        obs_dict, info = self.env.reset(seed=seed, options=options)
        state = self._get_physics_state()
        obs, dist = self._assemble_obs_and_concepts(state)
        
        self._previous_dist = dist
        self._prev_goal_dist = np.linalg.norm(state["obj_pos"] - state["goal_pos"])
        info["concept"] = self.current_concept.copy()
        
        return obs, info

# ---------------------------------------------------------------------------
# State-based variants
# ---------------------------------------------------------------------------

class PickPlaceStateEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

        self.task_types = [
            "regression", "regression", "regression", 
            "regression", "classification", "regression", 
            "classification", "regression", 
            "regression", "regression", "regression"
        ]
        self.num_classes = [0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0]
        self.concept_names = [
            "ee_to_obj_dx", "ee_to_obj_dy", "ee_to_obj_dz",
            "gripper_opening",
            "contact",
            "crush_risk",
            "is_broken",
            "ee_speed",
            "obj_to_goal_dx", "obj_to_goal_dy", "obj_to_goal_dz"
        ]
        self.temporal_concepts = [5, 6, 7]

        self.current_concept = np.zeros(N_CONCEPTS, dtype=np.float32)

    def get_concept(self) -> np.ndarray:
        return self.current_concept.copy()

    def _extract_concept_from_info(self, info: dict):
        if info is None: return None
        c = info.get("concept")
        if c is not None: return np.array(c, dtype=np.float32)
        fn = getattr(self.env, "get_concept", None)
        if callable(fn):
            try: return np.array(fn(), dtype=np.float32)
            except Exception: return None
        return None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        state = np.array(obs, dtype=np.float32) if obs is not None else None
        concept = self._extract_concept_from_info(info)
        self.current_concept = concept if concept is not None else np.zeros(N_CONCEPTS, dtype=np.float32)
        info = info or {}
        info["concept"] = self.current_concept.copy()
        return state, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = np.array(obs, dtype=np.float32) if obs is not None else None
        concept = self._extract_concept_from_info(info)
        self.current_concept = concept if concept is not None else np.zeros(N_CONCEPTS, dtype=np.float32)
        info = info or {}
        info["concept"] = self.current_concept.copy()
        return state, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Factory Functions (Regular RGB Variants)
# ---------------------------------------------------------------------------

def make_panda_pickplace_env(n_envs: int = 4, seed: int = 0, max_force: float = 50.0):
    """Vectorized environment factory - optimized for training."""
    from gymnasium.vector import AsyncVectorEnv
    
    def _make(rank: int):
        def _init():
            base_env = gym.make("PandaPickAndPlace-v3", render_mode="rgb_array")
            env = PandaPickPlaceConceptEnv(base_env, max_force_threshold=max_force)
            env.reset(seed=seed + rank)
            return DiscretizePandaActionWrapper(env)
        return _init
    
    return AsyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_panda_env(seed: int = 0, max_force: float = 50.0):
    """Single environment for debugging / future visualization."""
    base_env = gym.make("PandaPickAndPlace-v3", render_mode="rgb_array")
    env = PandaPickPlaceConceptEnv(base_env, max_force_threshold=max_force)
    env.reset(seed=seed)
    return DiscretizePandaActionWrapper(env)

def make_panda_pickplace_state_env(n_envs: int = 4, seed: int = 0, max_force: float = 50.0) -> gym.Env:
    from gymnasium.vector import AsyncVectorEnv
    def _make(rank: int):
        def _init():
            base = gym.make("PandaPickAndPlace-v3")
            env = PandaPickPlaceConceptEnv(base, max_force_threshold=max_force)
            env.reset(seed=seed + rank)
            state_env = PickPlaceStateEnv(env)
            return DiscretizePandaActionWrapper(state_env) 
        return _init
    return AsyncVectorEnv([_make(i) for i in range(n_envs)])

def make_single_panda_pickplace_state_env(seed: int = 0, max_force: float = 50.0) -> gym.Env:
    base = gym.make("PandaPickAndPlace-v3")
    env = PandaPickPlaceConceptEnv(base, max_force_threshold=max_force)
    env.reset(seed=seed)
    state_env = PickPlaceStateEnv(env)
    return DiscretizePandaActionWrapper(state_env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run smoke tests for pick_place envs")
    parser.add_argument("--render", action="store_true", help="Run the rendering smoke test (panda_gym)")
    parser.add_argument("--state", action="store_true", help="Run the state-only smoke test (no rendering)")
    parser.add_argument("--max-force", type=float, default=30.0, help="Max force threshold used to compute crush_risk")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for env reset")
    parser.add_argument("--render-steps", type=int, default=2000, help="Number of steps for rendering smoke test")
    parser.add_argument("--state-steps", type=int, default=20, help="Number of steps for state smoke test")
    args = parser.parse_args()

    # If neither flag provided, run both smoke tests
    if not args.render and not args.state:
        args.render = True
        args.state = True

    if args.render:
        try:
            env = make_single_panda_env(seed=args.seed, max_force=args.max_force)
            obs, info = env.reset()
            print('Render smoke test — Obs shape:', getattr(obs, 'shape', None) or (len(obs) if hasattr(obs, '__len__') else type(obs)))
            print('Initial Concepts:', info.get('concept'))
            for i in range(args.render_steps):
                obs, reward, term, trunc, info = env.step(env.action_space.sample())
                print(f"Step {i} | Crush Risk: {info['concept'][5]:.4f} | Reward: {reward:.2f}")
                time.sleep(0.05)
                if term or trunc:
                    break
            env.close()
        except Exception as e:
            print('Render smoke test skipped (panda_gym may be missing or failed):', e)

    if args.state:
        try:
            state_env = make_single_panda_pickplace_state_env(seed=args.seed, max_force=args.max_force)
            s_obs, s_info = state_env.reset()
            print('\nState smoke test: obs shape:', getattr(s_obs, 'shape', None) or (len(s_obs) if hasattr(s_obs, '__len__') else type(s_obs)))
            print('State initial concepts:', s_info.get('concept'))
            for j in range(args.state_steps):
                s_obs, s_reward, s_term, s_trunc, s_info = state_env.step(state_env.action_space.sample())
                print(f"State Step {j} | Crush Risk: {s_info['concept'][5]:.4f} | Reward: {s_reward:.2f}")
                if s_term or s_trunc:
                    break
            state_env.close()
        except Exception as e:
            print('State smoke test skipped (state env may be unavailable):', e)