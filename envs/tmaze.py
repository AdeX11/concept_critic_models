"""
tmaze.py — TMazeEnv

A corridor memory task: a cue is shown at the start of each episode
and must be remembered to make the correct choice at the junction.

Layout:
  position 0 ──► 1 ──► ... ──► corridor_len (junction)
  Cue visible at positions 0..CUE_STEPS-1 only.
  Agent must choose LEFT or RIGHT at the junction based on the original cue.

Observation (4 floats, single frame):
  [norm_pos, cue_active, cue_value, at_junction]
  - norm_pos    : position / corridor_len
  - cue_active  : 1.0 during cue phase (pos < CUE_STEPS), 0.0 afterwards
  - cue_value   : cue (0 or 1) during cue phase, 0.0 afterwards  (hidden)
  - at_junction : 1.0 at position corridor_len, else 0.0

With n_stack > 1 (FrameStackFlatWrapper applied), the obs becomes
(n_stack * 4,) — the last n_stack single-frame observations concatenated.

Actions:
  0: move forward  (valid anywhere before junction)
  1: choose LEFT   (correct when cue == 0)
  2: choose RIGHT  (correct when cue == 1)

Rewards (reward_mode='dense'):
  +1.0  correct choice at junction (LEFT when cue=0, RIGHT when cue=1)
  -1.0  wrong choice at junction, or forward at junction (must commit)
  -0.05 premature choice (choose action before junction)
  -0.01 per step (time pressure)
  Episode ends on choice or max_steps.

Rewards (reward_mode='sparse'):
  +1.0  correct choice at junction only
   0.0  everywhere else (wrong junction choice, forward at junction,
        premature choice, each step)
  Episode ends on choice or max_steps.

Concepts (2-dim, always the ground truth regardless of stacking):
  0  cue          classification {0, 1}   — TEMPORAL (hidden after CUE_STEPS)
  1  at_junction  classification {0, 1}   — static (derivable from norm_pos)

Memory requirement:
  For n_stack=1 / 'none' / 'gru' with corridor_len=10:
    cue disappears at position 3, junction is at 10 — 7 blank steps.
    Only a GRU (or equivalent recurrent state) can carry the cue to the junction.

  For n_stack=4 / 'stacked' with corridor_len=10:
    cue visible at positions 0-2; with n_stack=4 the oldest frame at the
    junction (pos=10) covers pos=7 — entirely blank.  Stacking alone
    CANNOT solve the task.  Use this as an ablation: if stacked fails
    but gru succeeds, the GRU hidden state (not the obs content) is
    what carries the cue.
"""

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv


CUE_STEPS = 3   # positions at which the cue is visible: 0, 1, 2


class TMazeEnv(gym.Env):
    """T-Maze with a delayed cue — requires memory to solve above chance."""

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0, corridor_len: int = 10, reward_mode: str = "dense"):
        super().__init__()
        assert reward_mode in ("dense", "sparse"), (
            f"reward_mode must be 'dense' or 'sparse', got '{reward_mode}'"
        )
        self.reward_mode  = reward_mode
        self.corridor_len = corridor_len
        self.max_steps    = corridor_len + 5

        # Single-frame obs: [norm_pos, cue_active, cue_value, at_junction]
        self.observation_space = gym.spaces.Box(
            low  = np.zeros(4, dtype=np.float32),
            high = np.ones(4,  dtype=np.float32),
        )
        self.action_space = gym.spaces.Discrete(3)  # 0=fwd, 1=left, 2=right

        self.task_types    = ["classification", "classification"]
        self.num_classes   = [2, 2]
        self.concept_names = ["cue", "at_junction"]
        self.temporal_concepts = [0]   # cue is the temporal concept

        self._rng   = np.random.default_rng(seed)
        self._pos   = 0
        self._cue   = 0
        self._steps = 0
        self.current_concept = np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------------
    # gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._pos   = 0
        self._cue   = int(self._rng.integers(0, 2))
        self._steps = 0
        self.current_concept = self._compute_concept()
        return self._get_obs(), {"concept": self.current_concept.copy()}

    def step(self, action: int):
        self._steps += 1
        terminated = False
        at_junction = (self._pos == self.corridor_len)

        if self.reward_mode == "dense":
            reward = -0.01
            if at_junction:
                if action == 1:
                    reward += 1.0 if self._cue == 0 else -1.0
                elif action == 2:
                    reward += 1.0 if self._cue == 1 else -1.0
                else:
                    reward += -1.0
                terminated = True
            else:
                if action == 0:
                    self._pos += 1
                else:
                    reward -= 0.05
        else:  # sparse: +1 correct at junction, 0 everywhere else
            reward = 0.0
            if at_junction:
                if action == 1:
                    reward = 1.0 if self._cue == 0 else 0.0
                elif action == 2:
                    reward = 1.0 if self._cue == 1 else 0.0
                # forward at junction: reward stays 0, episode ends
                terminated = True
            else:
                if action == 0:
                    self._pos += 1
                # premature choice: no movement, no penalty

        truncated = (not terminated) and (self._steps >= self.max_steps)

        self.current_concept = self._compute_concept()
        obs = self._get_obs()
        return obs, reward, terminated, truncated, {"concept": self.current_concept.copy()}

    def get_concept(self):
        return self.current_concept.copy()

    @property
    def concept_reward_active(self) -> float:
        """1.0 only at the junction — concept reward fires only when the cue
        must be recalled, not during the blank corridor steps."""
        return 1.0 if self._pos == self.corridor_len else 0.0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        norm_pos    = self._pos / self.corridor_len
        cue_active  = 1.0 if self._pos < CUE_STEPS else 0.0
        cue_value   = float(self._cue) if self._pos < CUE_STEPS else 0.0
        at_junction = 1.0 if self._pos == self.corridor_len else 0.0
        return np.array([norm_pos, cue_active, cue_value, at_junction], dtype=np.float32)

    def _compute_concept(self) -> np.ndarray:
        at_junction = 1.0 if self._pos == self.corridor_len else 0.0
        return np.array([float(self._cue), at_junction], dtype=np.float32)


# ---------------------------------------------------------------------------
# FrameStackFlatWrapper
# ---------------------------------------------------------------------------

class FrameStackFlatWrapper(gym.Wrapper):
    """
    Stacks the last n_stack single-frame observations into a flat 1-D vector.

    obs_space becomes Box(shape=(n_stack * single_obs_dim,)).
    concept_names / task_types / temporal_concepts / current_concept are
    forwarded unchanged — they describe ground-truth concepts, not the obs.

    At reset, the deque is pre-filled with copies of the first frame so the
    agent always receives a full-length observation.
    """

    def __init__(self, env: gym.Env, n_stack: int):
        super().__init__(env)
        self.n_stack = n_stack
        single_dim = int(np.prod(env.observation_space.shape))
        self._frames: deque = deque(maxlen=n_stack)

        self.observation_space = gym.spaces.Box(
            low  = np.tile(env.observation_space.low,  n_stack).astype(np.float32),
            high = np.tile(env.observation_space.high, n_stack).astype(np.float32),
        )

        # Forward concept metadata
        for attr in ("task_types", "num_classes", "concept_names",
                     "temporal_concepts", "current_concept"):
            if hasattr(env, attr):
                setattr(self, attr, getattr(env, attr))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_stack):
            self._frames.append(obs.copy())
        return self._stacked_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs.copy())
        if hasattr(self.env, "current_concept"):
            self.current_concept = self.env.current_concept
        return self._stacked_obs(), reward, terminated, truncated, info

    @property
    def concept_reward_active(self) -> float:
        return self.env.concept_reward_active if hasattr(self.env, "concept_reward_active") else 1.0

    def _stacked_obs(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=0)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

# corridor_len=10 for all modes: cue (visible at positions 0-2) disappears
# from any n_stack=4 window by position 6, so the stack cannot carry it to
# the junction at position 10.  The GRU hidden state is the only mechanism
# that can maintain the cue across the 7 blank steps.
_CORRIDOR_LEN = 10


def _make_tmaze(seed: int, n_stack: int, reward_mode: str = "dense") -> gym.Env:
    env = TMazeEnv(seed=seed, corridor_len=_CORRIDOR_LEN, reward_mode=reward_mode)
    if n_stack > 1:
        env = FrameStackFlatWrapper(env, n_stack)
    return env


def make_tmaze_env(n_envs: int, seed: int, n_stack: int = 1,
                   reward_mode: str = "dense", **_) -> SyncVectorEnv:
    def _make(i):
        def _init():
            return _make_tmaze(seed + i, n_stack, reward_mode=reward_mode)
        return _init
    return SyncVectorEnv([_make(i) for i in range(n_envs)])


def make_single_tmaze_env(seed: int = 0, n_stack: int = 1,
                           reward_mode: str = "dense", **_) -> gym.Env:
    return _make_tmaze(seed, n_stack, reward_mode=reward_mode)
