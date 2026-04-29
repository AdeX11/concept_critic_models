"""
highway.py — Highway environment with blindspot memory test.

Wraps highway-env (highway-v0) with discrete actions.
Actions (Discrete 5): 0: Left, 1: Idle, 2: Right, 3: Faster, 4: Slower

Variants:
  HighwayEnv      — 4-frame stacked RGB pixels, blindspot masking
  HighwayStateEnv — raw kinematics (25-dim), blindspot masking

Concepts (9):
  Index  Name                    Type            Temporal?   Notes
  0      ego_x                   regression      static
  1      ego_y                   regression      static
  2      ego_vx                  regression      temporal    (ego's proprioceptive speed)
  3      ego_lane_index          classification  static      {0,1,2,3}
  4      on_road                 classification  static      {0=crashed, 1=ok}
  5      blindspot_left_present  classification  TEMPORAL    {0=clear, 1=occupied}
  6      blindspot_left_rel_vx   regression      TEMPORAL    relative vx of blindspot car
  7      blindspot_right_present classification  TEMPORAL    {0=clear, 1=occupied}
  8      blindspot_right_rel_vx  regression      TEMPORAL    relative vx of blindspot car

Memory test:
  Vehicles behind or alongside the ego (x <= ego_x + margin) are zeroed out in
  the observation.  Ground-truth concepts still reflect true presence and
  relative velocity.  A car approaching from behind is visible, then enters the
  blindspot and disappears from the observation — the agent must remember both
  its existence and approach speed to avoid merging into an occupied lane.

  concept_reward_active returns 1.0 only when a vehicle is actually in a
  blindspot, gating the concept reward to the memory-demanding moments
  (same pattern as TMaze junction gating).
"""

from collections import deque

import cv2
import gymnasium as gym
import numpy as np
import highway_env  # noqa: F401

# ---------------------------------------------------------------------------

ROWS = 84
COLS = 84
IMG_STACK = 4
N_CONCEPTS = 9
N_VEHICLES = 5               # ego + 4 others
KIN_OBS_DIM = N_VEHICLES * 5  # 25

# Blindspot geometry (in highway-env normalized coordinates)
BLINDSPOT_X_MARGIN = 0.08     # vehicles with x <= ego_x + margin are in blindspot
LANE_WIDTH = 0.25             # approximate lane width
ADJACENT_LANE_Y_THRESH = 0.15  # max |dy| to consider "in adjacent lane"

# ---------------------------------------------------------------------------
# Concept metadata
# ---------------------------------------------------------------------------

CONCEPT_NAMES = [
    "ego_x", "ego_y", "ego_vx",
    "ego_lane_index", "on_road",
    "blindspot_left_present", "blindspot_left_rel_vx",
    "blindspot_right_present", "blindspot_right_rel_vx",
]

TASK_TYPES: list[str] = [
    "regression", "regression", "regression",
    "classification", "classification",
    "classification", "regression",
    "classification", "regression",
]

NUM_CLASSES: list[int] = [
    0, 0, 0,
    4, 2,
    2, 0,
    2, 0,
]

TEMPORAL_CONCEPTS: list[int] = [2, 5, 6, 7, 8]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _detect_blindspot(
    others: np.ndarray,
    ego: np.ndarray,
    target_y: float,
) -> tuple[bool, float]:
    """
    Check for a vehicle in the blindspot zone of the given adjacent lane.

    Returns (present: bool, rel_vx: float).
    rel_vx is the velocity of the blindspot vehicle relative to ego (vx_other - vx_ego).
    """
    in_lane = others[
        (others[:, 0] == 1) &
        (np.abs(others[:, 2] - target_y) < ADJACENT_LANE_Y_THRESH)
    ]
    if len(in_lane) == 0:
        return False, 0.0
    in_blindspot = in_lane[in_lane[:, 1] <= ego[1] + BLINDSPOT_X_MARGIN]
    if len(in_blindspot) == 0:
        return False, 0.0
    # Use the closest vehicle in the blindspot
    idx = np.argmax(in_blindspot[:, 1])  # closest (largest x)
    rel_vx = float(in_blindspot[idx, 3] - ego[3])
    return True, rel_vx


def _read_concepts(kin_obs: np.ndarray, crashed: bool = False) -> np.ndarray:
    """
    Extract all 9 concepts from the raw Kinematic observation matrix.

    kin_obs: (N_VEHICLES, 5) -> [presence, x, y, vx, vy]
    Row 0 is ego.

    Returns np.ndarray of shape (N_CONCEPTS,) — ground truth (never masked).
    """
    ego = kin_obs[0]
    others = kin_obs[1:]
    ego_x = float(ego[1])
    ego_y = float(ego[2])
    ego_vx = float(ego[3])

    lane_idx = int(np.clip(ego_y * 4, 0, 3))

    # Left adjacent lane: larger y, right: smaller y
    bs_left, rel_left = _detect_blindspot(others, ego, ego_y + LANE_WIDTH)
    bs_right, rel_right = _detect_blindspot(others, ego, ego_y - LANE_WIDTH)

    return np.array([
        ego_x,                          # 0: ego_x
        ego_y,                          # 1: ego_y
        ego_vx,                         # 2: ego_vx
        float(lane_idx),                # 3: ego_lane_index
        0.0 if crashed else 1.0,        # 4: on_road
        float(bs_left),                 # 5: blindspot_left_present
        rel_left,                       # 6: blindspot_left_rel_vx
        float(bs_right),                # 7: blindspot_right_present
        rel_right,                      # 8: blindspot_right_rel_vx
    ], dtype=np.float32)


def _mask_blindspot_vehicles(kin_obs: np.ndarray) -> np.ndarray:
    """
    Zero out vehicles behind or alongside the ego.

    Ego (row 0) is never masked.
    Vehicles with x > ego_x + BLINDSPOT_X_MARGIN (clearly ahead) are kept.
    """
    masked = kin_obs.copy()
    ego_x = masked[0, 1]
    for i in range(1, N_VEHICLES):
        if masked[i, 1] <= ego_x + BLINDSPOT_X_MARGIN:
            masked[i, :] = 0.0
    return masked


def _has_blindspot_vehicle(concepts: np.ndarray) -> bool:
    return bool(concepts[5] > 0.5 or concepts[7] > 0.5)


# ---------------------------------------------------------------------------
# HighwayEnv — pixel-based with blindspot masking
# ---------------------------------------------------------------------------

class HighwayEnv(gym.Wrapper):
    """
    Highway with 4-frame stacked RGB observations and blindspot masking.

    Observation: [img_stack * 3, ROWS, COLS] uint8
    Concepts:    ground truth from underlying kinematics.
    """

    def __init__(
        self,
        env: gym.Env,
        rows: int = ROWS,
        cols: int = COLS,
        img_stack: int = IMG_STACK,
    ):
        super().__init__(env)
        self.rows = rows
        self.cols = cols
        self.img_stack = img_stack

        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(img_stack * 3, rows, cols),
            dtype=np.uint8,
        )

        self.task_types = list(TASK_TYPES)
        self.num_classes = list(NUM_CLASSES)
        self.concept_names = list(CONCEPT_NAMES)
        self.temporal_concepts = list(TEMPORAL_CONCEPTS)

        self._frames: deque = deque(maxlen=img_stack)
        self.current_concept = np.zeros(N_CONCEPTS, dtype=np.float32)
        self._crashed = False

    # ----- concept interface -----

    def get_concept(self) -> np.ndarray:
        return self.current_concept.copy()

    @property
    def concept_reward_active(self) -> float:
        return 1.0 if _has_blindspot_vehicle(self.current_concept) else 0.0

    # ----- gym interface -----

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._crashed = False
        self._frames.clear()
        frame = self._get_frame()
        for _ in range(self.img_stack):
            self._frames.append(frame)
        self.current_concept = _read_concepts(obs, self._crashed)
        info["concept"] = self.current_concept.copy()
        return self._stack_frames(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self._crashed = self.env.unwrapped.vehicle.crashed
        frame = self._get_frame()
        self._frames.append(frame)
        self.current_concept = _read_concepts(obs, self._crashed)
        stacked = self._stack_frames()
        if done or truncated:
            info["terminal_observation"] = stacked
        info["concept"] = self.current_concept.copy()
        return stacked, reward, done, truncated, info

    # ----- internal -----

    def _get_frame(self) -> np.ndarray:
        img = self.env.render()
        if img is None:
            img = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        img = cv2.resize(img, (self.cols, self.rows))
        return img  # [H, W, 3]

    def _stack_frames(self) -> np.ndarray:
        # [img_stack, H, W, 3] -> [img_stack * 3, H, W]
        frames = np.stack(list(self._frames), axis=0)
        return frames.transpose(0, 3, 1, 2).reshape(-1, self.rows, self.cols)


# ---------------------------------------------------------------------------
# HighwayStateEnv — vector-based with blindspot masking
# ---------------------------------------------------------------------------

class HighwayStateEnv(gym.Wrapper):
    """
    Highway with kinematics vector observations and blindspot masking.

    Observation: [25] float32  (5 vehicles × [presence, x, y, vx, vy])
                 Vehicles behind/alongside ego are zeroed.
    Concepts:    ground truth from underlying kinematics.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(KIN_OBS_DIM,),
            dtype=np.float32,
        )
        self.task_types = list(TASK_TYPES)
        self.num_classes = list(NUM_CLASSES)
        self.concept_names = list(CONCEPT_NAMES)
        self.temporal_concepts = list(TEMPORAL_CONCEPTS)
        self.current_concept = np.zeros(N_CONCEPTS, dtype=np.float32)
        self._crashed = False

    # ----- concept interface -----

    def get_concept(self) -> np.ndarray:
        return self.current_concept.copy()

    @property
    def concept_reward_active(self) -> float:
        return 1.0 if _has_blindspot_vehicle(self.current_concept) else 0.0

    # ----- gym interface -----

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._crashed = False
        self.current_concept = _read_concepts(obs, self._crashed)
        info["concept"] = self.current_concept.copy()
        return _mask_blindspot_vehicles(obs).flatten().astype(np.float32), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self._crashed = self.env.unwrapped.vehicle.crashed
        self.current_concept = _read_concepts(obs, self._crashed)
        info["concept"] = self.current_concept.copy()
        return _mask_blindspot_vehicles(obs).flatten().astype(np.float32), reward, done, truncated, info


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _base_highway_config() -> dict:
    return {"observation": {"type": "Kinematics", "vehicles_count": N_VEHICLES}}


def make_highway_env(n_envs: int = 4, seed: int = 0, n_stack: int = 4) -> gym.Env:
    """Vectorized HighwayEnv (pixel-based, blindspot masked)."""
    from gymnasium.vector import AsyncVectorEnv

    def _make(rank: int):
        def _init():
            env = gym.make("highway-v0", render_mode="rgb_array", config=_base_highway_config())
            env = HighwayEnv(env, img_stack=n_stack)
            env.reset(seed=seed + rank)
            return env
        return _init
    return AsyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_highway_env(seed: int = 0, n_stack: int = 4) -> HighwayEnv:
    """Single HighwayEnv (pixel-based, blindspot masked)."""
    env = gym.make("highway-v0", render_mode="rgb_array", config=_base_highway_config())
    wrapped = HighwayEnv(env, img_stack=n_stack)
    wrapped.reset(seed=seed)
    return wrapped


def make_highway_state_env(n_envs: int = 4, seed: int = 0) -> gym.Env:
    """Vectorized HighwayStateEnv (kinematics, blindspot masked)."""
    from gymnasium.vector import AsyncVectorEnv

    def _make(rank: int):
        def _init():
            env = gym.make("highway-v0", config=_base_highway_config())
            env = HighwayStateEnv(env)
            return env
        return _init
    return AsyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_highway_state_env(seed: int = 0) -> HighwayStateEnv:
    """Single HighwayStateEnv (kinematics, blindspot masked)."""
    env = gym.make("highway-v0", config=_base_highway_config())
    wrapped = HighwayStateEnv(env)
    wrapped.reset(seed=seed)
    return wrapped


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=== Testing HighwayStateEnv ===")
    env = gym.make("highway-v0", render_mode="human", config={
        "observation": {"type": "Kinematics", "vehicles_count": N_VEHICLES},
        "screen_width": 600,
        "screen_height": 300,
    })
    wrapped = HighwayStateEnv(env)

    obs, info = wrapped.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Concept names: {wrapped.concept_names}")
    print(f"Temporal concepts: {wrapped.temporal_concepts}")

    for step_idx in range(250):
        action = wrapped.action_space.sample()
        obs, reward, done, truncated, info = wrapped.step(action)

        concepts = wrapped.get_concept()
        bs_l_present = concepts[5] > 0.5
        bs_r_present = concepts[7] > 0.5
        bs_l_flag = f"L({concepts[6]:+.1f})" if bs_l_present else "-"
        bs_r_flag = f"R({concepts[8]:+.1f})" if bs_r_present else "-"
        print(
            f"Step {step_idx:3d} | blindspot: [{bs_l_flag} | {bs_r_flag}] | "
            f"on_road: {concepts[4]:.0f} | ego_vx: {concepts[2]:.2f} | "
            f"active: {wrapped.concept_reward_active:.0f}   ",
            end="\r",
        )

        wrapped.render()
        if done or truncated:
            print(f"\nEpisode ended at step {step_idx}")
            obs, info = wrapped.reset()
            time.sleep(0.5)

    wrapped.close()