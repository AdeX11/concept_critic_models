"""
Highway Environment Concept Wrapper and Tests
===================================================

This script adapts the Farama Foundation's `highway-env` for the ConceptActorCritic.
It turns a standard driving simulation into a Partially Observable Markov Decision 
Process (POMDP) by extracting both observable kinematics and hidden behavioral 
parameters into a complete Concept Bottleneck.

The Concepts per NPC (5 x 4 = 20 total) are as follows:
------------------------------------
1. Aggressiveness (Temporal Regression) - Latent: 0.0 to 1.0 behavior profile.
2. Relative X-Position (Static Regression) - Observable: Distance in meters.
3. Relative Lane (Static Classification) - Observable: 0 = Left, 1 = Same, 2 = Right.
4. Relative Speed (Static Regression) - Observable: Difference in velocity (m/s).
5. Acceleration (Temporal Regression) - Latent: Change in speed from the previous frame.
"""

import argparse
import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv
import highway_env 

class HighwayConceptWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # highway-env usually outputs a V x F matrix (e.g., 5 vehicles, 5 features)
        original_shape = self.env.observation_space.shape
        self.obs_dim = np.prod(original_shape)
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # Track 1 Ego + N closest NPCs
        self.num_npc = original_shape[0] - 1
        self.concepts_per_npc = 5  # Expanded to include Speed and Acceleration
        self.concept_dim = self.num_npc * self.concepts_per_npc
        self.tracked_slots = [None] * self.num_npc

        # ---------------------------------------------------------
        # CBM API PROPERTIES
        # ---------------------------------------------------------
        self.task_types = []
        self.num_classes = []
        self.concept_names = []
        self.temporal_concepts = []
        
        for i in range(self.num_npc):
            idx = i * self.concepts_per_npc
            
            # 1. Aggressiveness (Latent / Temporal)
            self.task_types.append("regression")
            self.num_classes.append(0)
            self.concept_names.append(f"npc_{i+1}_aggress")
            self.temporal_concepts.append(idx) 
            
            # 2. Relative X Position (Observable / Static)
            self.task_types.append("regression")
            self.num_classes.append(0)
            self.concept_names.append(f"npc_{i+1}_x")
            
            # 3. Relative Lane (Observable / Static)
            self.task_types.append("classification")
            self.num_classes.append(3)
            self.concept_names.append(f"npc_{i+1}_lane")

            # 4. Relative Speed (Observable / Static)
            self.task_types.append("regression")
            self.num_classes.append(0)
            self.concept_names.append(f"npc_{i+1}_speed")

            # 5. Acceleration (Latent / Temporal)
            # The GRU must compare current speed against its memory of past speed.
            self.task_types.append("regression")
            self.num_classes.append(0)
            self.concept_names.append(f"npc_{i+1}_accel")
            self.temporal_concepts.append(idx + 4)
        
        # State tracker
        self.current_concept = np.zeros(self.concept_dim, dtype=np.float32)

        # FIX: Patch CBM properties onto the unwrapped env for VectorEnv compatibility
        self.env.unwrapped.task_types = self.task_types
        self.env.unwrapped.num_classes = self.num_classes
        self.env.unwrapped.concept_names = self.concept_names
        self.env.unwrapped.temporal_concepts = self.temporal_concepts

    # ---------------------------------------------------------
    # VECTOR ENVIRONMENT PROPERTIES
    # ---------------------------------------------------------
    @property
    def single_action_space(self):
        return self.env.single_action_space

    @property
    def single_observation_space(self):
        return self.env.single_observation_space

    def get_concept(self) -> np.ndarray:
        """Returns the current concepts, required by the CBM extraction logic."""
        return self.current_concept.copy()
        
    # ---------------------------------------------------------
    # CORE ENVIRONMENT LOOP
    # ---------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Inject hidden parameters into all spawned NPC vehicles
        for vehicle in self.env.unwrapped.road.vehicles:
            if vehicle is not self.env.unwrapped.vehicle:  
                # 1. Inject Aggressiveness
                aggressiveness = np.random.uniform(0.0, 1.0)
                vehicle.aggressiveness = aggressiveness 
                vehicle.target_speed = 20.0 + (aggressiveness * 15.0) 
                
                # 2. Initialize Speed Tracking (for Acceleration)
                vehicle._previous_speed = vehicle.speed
                
                if hasattr(vehicle, 'distance_wanted'):
                    vehicle.distance_wanted = 2.0 + ((1.0 - aggressiveness) * 8.0)
                    
        flat_obs, concepts = self._assemble_obs_and_concepts(obs)
        
        # Update state and info dict
        self.current_concept = concepts
        info["concept"] = self.current_concept.copy()
        
        return flat_obs, info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # If new cars spawned during the step, give them personalities and trackers
        for vehicle in self.env.unwrapped.road.vehicles:
            if vehicle is not self.env.unwrapped.vehicle and not hasattr(vehicle, 'aggressiveness'):
                aggress = np.random.uniform(0.0, 1.0)
                vehicle.aggressiveness = aggress
                vehicle.target_speed = 20.0 + (aggress * 15.0)
                vehicle._previous_speed = vehicle.speed
        
        flat_obs, concepts = self._assemble_obs_and_concepts(obs)
        
        # Update state and info dict
        self.current_concept = concepts
        info["concept"] = self.current_concept.copy()
        
        return flat_obs, reward, terminated, truncated, info
        
    def _assemble_obs_and_concepts(self, obs):
        # obs is usually a 2D matrix 
        # Row 0 is the Ego car. Rows 1 through 4 are the NPCs.
        # Columns are usually: [presence, x, y, vx, vy]
        
        flat_obs = obs.flatten().astype(np.float32)
        concepts = np.zeros(self.concept_dim, dtype=np.float32)
        
        ego = self.env.unwrapped.vehicle
        all_vehicles = self.env.unwrapped.road.vehicles
        
        for i in range(self.num_npc):
            row = i + 1  # Skip the Ego car (row 0)
            idx = i * self.concepts_per_npc
            
            # If there is no car in this slot (presence feature is 0), skip it
            if obs[row, 0] == 0:
                continue
                
            # ---------------------------------------------------------
            # OBSERVABLE CONCEPTS: Pull straight from the matrix!
            # ---------------------------------------------------------
            rel_x = obs[row, 1]      # The exact X value the network sees
            rel_y = obs[row, 2]      # The exact Y value the network sees
            rel_speed = obs[row, 3]  # The exact Speed value the network sees
            
            concepts[idx + 1] = rel_x
            
            # Categorize the lane based on the normalized Y value
            if rel_y < -0.1: concepts[idx + 2] = 0.0     # Left
            elif rel_y > 0.1: concepts[idx + 2] = 2.0    # Right
            else: concepts[idx + 2] = 1.0                # Same
            
            concepts[idx + 3] = rel_speed

            # ---------------------------------------------------------
            # LATENT CONCEPTS: "Reverse Lookup"
            # ---------------------------------------------------------
            
            matched_vehicle = None
            for v in all_vehicles:
                if v is not ego:
                    # Check if this vehicle's relative speed matches the observation
                    if abs((v.speed - ego.speed) / 20.0 - rel_speed) < 0.1: 
                        matched_vehicle = v
                        break
            
            if matched_vehicle:
                concepts[idx] = getattr(matched_vehicle, 'aggressiveness', 0.5)
                
                # Acceleration calculation
                prev_speed = getattr(matched_vehicle, '_previous_speed', matched_vehicle.speed)
                concepts[idx + 4] = matched_vehicle.speed - prev_speed
                matched_vehicle._previous_speed = matched_vehicle.speed

        return flat_obs, concepts

# ---------------------------------------------------------------------------
# Factory Functions (For PPO/Training)
# ---------------------------------------------------------------------------

def make_single_highway_env(seed=0):
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    env = HighwayConceptWrapper(env)
    env.action_space.seed(seed)
    return env

def make_single_highway_state_env(seed=0):
    env = gym.make("highway-fast-v0")
    env = HighwayConceptWrapper(env)
    env.action_space.seed(seed)
    return env

def make_highway_env(n_envs: int = 4, seed: int = 0):
    def _make(rank: int):
        def _init():
            env = gym.make("highway-fast-v0", render_mode="rgb_array")
            env = HighwayConceptWrapper(env)
            env.action_space.seed(seed + rank)
            return env
        return _init
    venv = AsyncVectorEnv([_make(i) for i in range(n_envs)])

    # THE FIX: Extract properties from a dummy env and patch them onto the VectorEnv
    dummy = _make(0)()
    venv.task_types = dummy.task_types
    venv.num_classes = dummy.num_classes
    venv.concept_names = dummy.concept_names
    venv.temporal_concepts = dummy.temporal_concepts
    venv.concept_dim = dummy.concept_dim
    dummy.close()

    return venv

def make_highway_state_env(n_envs: int = 4, seed: int = 0):
    def _make(rank: int):
        def _init():
            env = gym.make("highway-fast-v0")
            env = HighwayConceptWrapper(env)
            env.action_space.seed(seed + rank)
            return env
        return _init
    
    venv = AsyncVectorEnv([_make(i) for i in range(n_envs)])

    # THE FIX: Extract properties from a dummy env and patch them onto the VectorEnv
    dummy = _make(0)()
    venv.task_types = dummy.task_types
    venv.num_classes = dummy.num_classes
    venv.concept_names = dummy.concept_names
    venv.temporal_concepts = dummy.temporal_concepts
    venv.concept_dim = dummy.concept_dim
    dummy.close()

    return venv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run smoke tests for Highway Concept envs")
    parser.add_argument("--render", action="store_true", help="Run the rendering smoke test")
    parser.add_argument("--state", action="store_true", help="Run the state-only smoke test (no rendering)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for env reset")
    parser.add_argument("--render-steps", type=int, default=50, help="Number of steps for rendering smoke test")
    parser.add_argument("--state-steps", type=int, default=20, help="Number of steps for state smoke test")
    args = parser.parse_args()

    # If neither flag provided, run both smoke tests
    if not args.render and not args.state:
        args.render = True
        args.state = True

    if args.render:
        try:
            env = make_single_highway_env(seed=args.seed) #to visualize, change line 209 to render_mode="human"
            env.unwrapped.configure({"real_time_rendering": True})

            obs, info = env.reset(seed=args.seed)
            print('Render smoke test — Obs shape:', getattr(obs, 'shape', None) or (len(obs) if hasattr(obs, '__len__') else type(obs)))
            
            for i in range(args.render_steps):
                action = env.action_space.sample()
                obs, reward, term, trunc, info = env.step(action)
                
                # Pull out the concepts for the closest NPC (Indices 0 through 4)
                npc_aggress = info['concept'][0]
                npc_rel_x   = info['concept'][1]
                npc_speed   = info['concept'][3]
                npc_accel   = info['concept'][4]
                
                print(f"Step {i:2d} | NPC_1 [Agg: {npc_aggress:.2f} | Dist: {npc_rel_x:6.1f}m | Speed: {npc_speed:5.1f}m/s | Accel: {npc_accel:5.2f}] | R: {reward:.2f}")
                
                if term or trunc:
                    print(f"--- Episode ended at step {i} ---")
                    break
            env.close()
            print("Render smoke test completed successfully.\n")
        except Exception as e:
            print('Render smoke test skipped/failed:', e)

    if args.state:
        try:
            state_env = make_single_highway_state_env(seed=args.seed)
            s_obs, s_info = state_env.reset(seed=args.seed)
            
            for j in range(args.state_steps):
                action = state_env.action_space.sample()
                s_obs, s_reward, s_term, s_trunc, s_info = state_env.step(action)
                
                s_aggress = s_info['concept'][0]
                s_accel   = s_info['concept'][4]
                print(f"State Step {j:2d} | NPC_1 Aggressiveness: {s_aggress:.4f} | Accel: {s_accel:.4f} | Reward: {s_reward:.2f}")
                
                if s_term or s_trunc:
                    print(f"--- Episode ended at step {j} ---")
                    break
            state_env.close()
            print("State smoke test completed successfully.")
        except Exception as e:
            print('State smoke test skipped (state env may be unavailable):', e)

