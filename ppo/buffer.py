"""
buffer.py — RolloutBuffer with concept + hidden state storage.

Stores per step:
  observations   [buffer_size, n_envs, *obs_shape]
  concepts       [buffer_size, n_envs, concept_dim]    — ground truth from env
  hidden_states  [buffer_size, n_envs, hidden_dim]     — GRU h_t (zeros for non-GRU)
  concept_values [buffer_size, n_envs]                 — V_c from concept critic
  concept_log_probs [buffer_size, n_envs]              — log prob of concept prediction
  actions, rewards, values, log_probs, episode_starts

GAE for task policy and separate concept advantage computation.
"""

from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch


def _swap_flatten(arr: np.ndarray) -> np.ndarray:
    """[T, N, ...] → [T*N, ...]"""
    shape = arr.shape
    if len(shape) == 2:
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1])
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])


class RolloutBuffer:
    """
    On-policy rollout buffer supporting all three methods.

    Dict observations are handled transparently via the `obs_is_dict` flag.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape,                     # tuple or dict of tuples
        concept_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_envs: int = 1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: Union[str, torch.device] = "cpu",
    ):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.concept_dim = concept_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_envs = n_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = torch.device(device) if isinstance(device, str) else device

        self.obs_is_dict = isinstance(obs_shape, dict)
        self._ready = False
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        T, N = self.buffer_size, self.n_envs

        if self.obs_is_dict:
            self.observations: Dict = {
                k: np.zeros((T, N, *s), dtype=np.float32)
                for k, s in self.obs_shape.items()
            }
        else:
            self.observations = np.zeros((T, N, *self.obs_shape), dtype=np.float32)

        self.concepts        = np.zeros((T, N, self.concept_dim), dtype=np.float32)
        self.hidden_states   = np.zeros((T, N, self.hidden_dim),  dtype=np.float32)
        self.concept_values  = np.zeros((T, N),                   dtype=np.float32)
        self.concept_log_probs = np.zeros((T, N),                 dtype=np.float32)
        self.concept_rewards = np.zeros((T, N),                   dtype=np.float32)
        self.actions         = np.zeros((T, N, self.action_dim),  dtype=np.float32)
        self.rewards         = np.zeros((T, N),                   dtype=np.float32)
        self.values          = np.zeros((T, N),                   dtype=np.float32)
        self.log_probs       = np.zeros((T, N),                   dtype=np.float32)
        self.episode_starts  = np.zeros((T, N),                   dtype=np.float32)
        self.advantages      = np.zeros((T, N),                   dtype=np.float32)
        self.returns         = np.zeros((T, N),                   dtype=np.float32)
        self.concept_advantages = np.zeros((T, N),                dtype=np.float32)
        self.concept_returns    = np.zeros((T, N),                dtype=np.float32)

        self.pos   = 0
        self.full  = False
        self._ready = False

    @property
    def size(self) -> int:
        return self.buffer_size if self.full else self.pos

    # ------------------------------------------------------------------
    # Add a single timestep
    # ------------------------------------------------------------------

    def add(
        self,
        obs,
        concept: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        hidden_state: Optional[np.ndarray] = None,
        concept_value: Optional[torch.Tensor] = None,
        concept_log_prob: Optional[np.ndarray] = None,
        concept_reward: Optional[np.ndarray] = None,
    ) -> None:
        t = self.pos
        N = self.n_envs

        if self.obs_is_dict:
            for k in self.observations:
                self.observations[k][t] = np.array(obs[k]).reshape(N, *self.obs_shape[k])
        else:
            self.observations[t] = np.array(obs).reshape(N, *self.obs_shape)

        self.concepts[t]       = np.array(concept).reshape(N, self.concept_dim)
        self.actions[t]        = np.array(action).reshape(N, self.action_dim)
        self.rewards[t]        = np.array(reward).reshape(N)
        self.episode_starts[t] = np.array(episode_start).reshape(N)
        self.values[t]         = value.detach().cpu().numpy().flatten()[:N]
        self.log_probs[t]      = log_prob.detach().cpu().numpy().flatten()[:N]

        if hidden_state is not None:
            self.hidden_states[t] = np.array(hidden_state).reshape(N, self.hidden_dim)
        if concept_value is not None:
            self.concept_values[t] = concept_value.detach().cpu().numpy().flatten()[:N]
        if concept_log_prob is not None:
            self.concept_log_probs[t] = np.array(concept_log_prob).reshape(N)
        if concept_reward is not None:
            self.concept_rewards[t] = np.array(concept_reward).reshape(N)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    # ------------------------------------------------------------------
    # GAE for task policy
    # ------------------------------------------------------------------

    def compute_returns_and_advantage(
        self,
        last_values: torch.Tensor,
        dones: np.ndarray,
    ) -> None:
        """Standard GAE computation for task advantages."""
        last_values_np = last_values.detach().cpu().numpy().flatten()
        last_gae = 0.0

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values_np
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]

            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae

        self.returns = self.advantages + self.values

    # ------------------------------------------------------------------
    # Concept advantage for concept_actor_critic
    # ------------------------------------------------------------------

    def compute_concept_returns_and_advantage(
        self,
        last_concept_values: torch.Tensor,
        dones: np.ndarray,
    ) -> None:
        """
        GAE for the concept critic (mirrors compute_returns_and_advantage).
        concept_advantages[t] = delta_t + gamma * lambda * (1-done) * A_{t+1}
          where delta_t = r_c[t] + gamma * V_c(t+1) * (1-done) - V_c[t]
        concept_returns[t] = concept_advantages[t] + V_c[t]

        r_c[t] is the concept accuracy reward stored in self.concept_rewards,
        computed during rollout collection in ppo.py.
        """
        last_cv = last_concept_values.detach().cpu().numpy().flatten()
        last_gae = 0.0

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_cv = last_cv
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_cv = self.concept_values[step + 1]

            delta = (
                self.concept_rewards[step]
                + self.gamma * next_cv * next_non_terminal
                - self.concept_values[step]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.concept_advantages[step] = last_gae

        self.concept_returns = self.concept_advantages + self.concept_values

    # ------------------------------------------------------------------
    # Minibatch generator
    # ------------------------------------------------------------------

    def get(
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[dict, None, None]:
        """
        Yields minibatches of flattened [T*N, ...] tensors as plain dicts.
        Call after compute_returns_and_advantage().
        """
        assert self.full, "Buffer not full; cannot iterate."
        T, N = self.buffer_size, self.n_envs
        total = T * N

        if not self._ready:
            self._flat = self._flatten_all()
            self._ready = True

        indices = np.random.permutation(total)
        if batch_size is None:
            batch_size = total

        start = 0
        while start < total:
            idx = indices[start: start + batch_size]
            yield self._make_batch(idx)
            start += batch_size

    def _flatten_all(self) -> dict:
        flat = {}
        if self.obs_is_dict:
            flat["observations"] = {
                k: torch.as_tensor(_swap_flatten(v), dtype=torch.float32)
                for k, v in self.observations.items()
            }
        else:
            flat["observations"] = torch.as_tensor(
                _swap_flatten(self.observations), dtype=torch.float32
            )

        for name in [
            "concepts",
            "hidden_states",
            "actions",
            "values",
            "log_probs",
            "advantages",
            "returns",
            "concept_values",
            "concept_log_probs",
            "concept_rewards",
            "concept_advantages",
            "concept_returns",
            "episode_starts",
        ]:
            arr = getattr(self, name)
            flat[name] = torch.as_tensor(_swap_flatten(arr), dtype=torch.float32)
        return flat

    def _make_batch(self, idx: np.ndarray) -> dict:
        flat = self._flat
        batch = {}
        if self.obs_is_dict:
            batch["observations"] = {
                k: v[idx].to(self.device)
                for k, v in flat["observations"].items()
            }
        else:
            batch["observations"] = flat["observations"][idx].to(self.device)

        for name in [
            "concepts",
            "hidden_states",
            "actions",
            "values",
            "log_probs",
            "advantages",
            "returns",
            "concept_values",
            "concept_log_probs",
            "concept_rewards",
            "concept_advantages",
            "concept_returns",
        ]:
            batch[name] = flat[name][idx].to(self.device)

        return batch
