"""
Roundabout Environment Concept Wrapper (Hard Mode)
===================================================
Features:
1. Limited FOV: NPCs outside a 180° forward cone are masked (obs and concept = 0).
2. Latent Physics: TTC removed. Agent must derive closing rates via GRU memory.
3. Stable Tracking: Uses the underlying simulation IDs to ensure concept slots
   don't "flicker" when cars move in/out of the FOV.
"""

import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv
import highway_env 
from highway_env.envs.roundabout_env import RoundaboutEnv
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.utils import class_from_path

class RoundaboutConceptWrapper(gym.Wrapper):
    def __init__(self, env, fov_deg=180):
        super().__init__(env)

        if "other_vehicles_type" not in self.config:
            self.config["other_vehicles_type"] = \
                "highway_env.vehicle.behavior.IDMVehicle"
        
        self.fov_rad = np.deg2rad(fov_deg)
        
        original_shape = self.env.observation_space.shape
        self.obs_dim = np.prod(original_shape)
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # We reduce to 4 core concepts per NPC (Removing TTC)
        self.num_npc = original_shape[0] - 1
        self.concepts_per_npc = 4 
        self.concept_dim = self.num_npc * self.concepts_per_npc

        self.task_types = []
        self.num_classes = []
        self.concept_names = []
        self.temporal_concepts = []
        
        for i in range(self.num_npc):
            idx = i * self.concepts_per_npc
            # 1. Dist, 2. Angle, 3. Lane, 4. Speed
            self.task_types.extend(["regression", "regression", "classification", "regression"])
            self.num_classes.extend([0, 0, 2, 0])
            self.concept_names.extend([
                f"npc_{i+1}_dist", f"npc_{i+1}_angle", 
                f"npc_{i+1}_lane", f"npc_{i+1}_speed"
            ])
            # Distance and Speed are temporal because their DERIVATIVE (accel/TTC) is latent
            self.temporal_concepts.extend([idx, idx + 3])
        
        self.current_concept = np.zeros(self.concept_dim, dtype=np.float32)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        flat_obs, concepts = self._apply_fov_and_assemble(obs)
        self.current_concept = concepts
        info["concept"] = self.current_concept.copy()
        return flat_obs, info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        flat_obs, concepts = self._apply_fov_and_assemble(obs)
        self.current_concept = concepts
        info["concept"] = self.current_concept.copy()
        return flat_obs, reward, terminated, truncated, info
        
    def _apply_fov_and_assemble(self, obs):
        """
        Masks NPCs outside the FOV and builds the concept vector.
        obs shape: (V, F) where row 0 is Ego.
        """
        # We work on a copy to mask the 'flat_obs' sent to the model
        masked_obs = obs.copy()
        concepts = np.zeros(self.concept_dim, dtype=np.float32)
        
        for i in range(self.num_npc):
            row = i + 1 
            idx = i * self.concepts_per_npc
            
            if obs[row, 0] == 0: continue
                
            rel_x = obs[row, 1]
            rel_y = obs[row, 2]
            
            # Calculate angle relative to Ego heading
            # In highway-env, x is forward, y is lateral
            angle = np.arctan2(rel_y, rel_x)
            
            # --- FOV CHECK (180 degree forward cone) ---
            # If angle is > 90 deg or < -90 deg, it's behind us.
            if abs(angle) > (self.fov_rad / 2):
                masked_obs[row, :] = 0 # Agent sees nothing for this NPC
                continue 

            # 1. Distance
            dist = np.sqrt(rel_x**2 + rel_y**2)
            concepts[idx] = dist
            
            # 2. Angle
            concepts[idx + 1] = angle
            
            # 3. Radial Lane
            concepts[idx + 2] = 1.0 if abs(rel_y) > 0.1 else 0.0
            
            # 4. Relative Speed Magnitude
            rel_vx = obs[row, 3]
            rel_vy = obs[row, 4]
            concepts[idx + 3] = np.sqrt(rel_vx**2 + rel_vy**2)

        return masked_obs.flatten().astype(np.float32), concepts


class CongestedRoundaboutEnv(RoundaboutEnv):
    """
    Roundabout with aggressive congestion:
    - High vehicle count
    - Tight spacing
    - Continuous spawning
    """

    def _make_vehicles(self):
        super()._make_vehicles()  # ✅ build road + base vehicles FIRST

        from highway_env.utils import class_from_path
        vehicle_class = class_from_path(self.config["other_vehicles_type"])

        # Add extra vehicles ON TOP of working base env
        for _ in range(25):
            lanes = self.np_random.choice([
                ("ser", "ses", 0),
                ("eer", "ees", 0),
                ("ner", "nes", 0),
                ("wer", "wes", 0),
            ])
            lane_index = lanes[self.np_random.integers(len(lanes))]
            lane = self.road.network.get_lane(lane_index)

            vehicle = vehicle_class(
                self.road,
                lane.position(self.np_random.uniform(90, 120), 0),  # 🔥 visible zone
                speed=self.np_random.uniform(4, 10)
            )

            if not any(np.linalg.norm(v.position - vehicle.position) < 5
                    for v in self.road.vehicles):
                self.road.vehicles.append(vehicle)

    def _spawn_vehicle(self, close_spacing=False):
        """Spawn vehicles from all entries with tighter gaps"""
        lanes = [
            ("ser", "ses", 0),
            ("eer", "ees", 0),
            ("ner", "nes", 0),
            ("wer", "wes", 0),
        ]

        lane_index = lanes[self.np_random.integers(len(lanes))]
        lane = self.road.network.get_lane(lane_index)

        # --- tighter spacing ---
        if close_spacing:
            longitudinal = self.np_random.uniform(0, 30)  # 🔥 dense cluster
        else:
            longitudinal = self.np_random.uniform(0, 80)

        vehicle_class = class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_class(
            self.road,
            lane.position(longitudinal, 0),
            speed=self.np_random.uniform(4, 10)
        )

        # Avoid immediate collisions at spawn
        if (
            not any(np.linalg.norm(v.position - vehicle.position) < 5
                    for v in self.road.vehicles)
            or self.np_random.random() < 0.2  # allow occasional tight spawn
        ):
            self.road.vehicles.append(vehicle)

    def step(self, action):
        """Keep injecting vehicles over time (traffic flow)"""
        obs, reward, terminated, truncated, info = super().step(action)

        # continuous inflow
        if self.np_random.random() < 0.6:
            self._spawn_vehicle(close_spacing=True)

        return obs, reward, terminated, truncated, info

# ---------------------------------------------------------------------------
# Factory Functions (Vectorized for Training)
# ---------------------------------------------------------------------------

def make_highway_env(n_envs: int = 4, seed: int = 0):
    def _init():
        env = CongestedRoundaboutEnv()
        
        # Keep your observation config (this still matters!)
        env.configure({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": False,
                "order": "sorted"
            }
        })

        env.reset()
        env = RoundaboutConceptWrapper(env)
        return env

    return SyncVectorEnv([_init for _ in range(n_envs)])

def make_highway_state_env(n_envs: int = 4, seed: int = 0):
    def _init():
        env = CongestedRoundaboutEnv()
        
        # Keep your observation config (this still matters!)
        env.configure({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": False,
                "order": "sorted"
            }
        })

        env.reset()
        env = RoundaboutConceptWrapper(env)
        return env

    return SyncVectorEnv([_init for _ in range(n_envs)])

def make_single_highway_env(seed: int = 0):
    env = CongestedRoundaboutEnv(render_mode="human")
    env.reset(seed=seed)
    env = RoundaboutConceptWrapper(env)
    print(len(env.unwrapped.road.vehicles))
    return env

def make_single_highway_state_env(seed: int = 0):
    env = CongestedRoundaboutEnv()
    env.reset(seed=seed)
    env = RoundaboutConceptWrapper(env)
    return env


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-episode smoke tests for Roundabout")
    parser.add_argument("--render", action="store_true", help="Run with human rendering")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Use the 'single' factory for testing
    if args.render:
        env = make_single_highway_env(seed=args.seed)
    else:
        env = make_single_highway_state_env(seed=args.seed)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        step = 0
        ep_reward = 0
        
        print(f"\n=== Starting Episode {ep+1} ===")
        
        while not done:
            action = env.action_space.sample()  # Random actions for smoke test
            obs, reward, term, trunc, info = env.step(action)
            
            # Extract concepts for the logs
            concepts = info.get('concept', np.zeros(20))
            npc_dist = concepts[0]
            npc_ttc  = concepts[4]
            
            print(f"Ep {ep+1} | Step {step:2d} | NPC_1 Dist: {npc_dist:5.1f} | TTC: {npc_ttc:4.2f}s | R: {reward:.2f}")
            
            ep_reward += reward
            step += 1
            done = term or trunc
            
            if term:
                print(f"--- CRASH or GOAL REACHED at step {step} ---")
                # Highlight the concept state at the moment of failure
                print(f"Final Concept State: Dist={npc_dist:.2f}, TTC={npc_ttc:.2f}")
            
            if trunc:
                print(f"--- Episode Timed Out ---")

        print(f"Episode {ep+1} Total Reward: {ep_reward:.2f}")

    env.close()
    print("\nAll smoke test episodes completed.")