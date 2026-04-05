"""
ppo.py — Core PPO algorithm supporting all three concept methods.

Two training phases per iteration:
  Phase 1: train_policy()    — standard PPO clipped surrogate
  Phase 2: train_concepts()  — method-specific concept learning

training_mode controls how concept network parameters interact with policy training:
  'two_phase'   — concept net is FROZEN during train_policy() (uses optimizer_exclude_concept);
                  updated only via supervised / concept actor-critic loss in train_concepts().
                  Mirrors the LICORICE paper's vanilla_freeze setup.
  'end_to_end'  — concept net parameters are included in the policy gradient step
                  (uses full optimizer); gradients from policy loss flow through concept net.

Label collection is simple random sampling (no active learning).
"""

import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .buffer import RolloutBuffer
from .policy import ActorCriticPolicy
from .networks import ConceptActorCritic


def _obs_to_tensor(obs, device: torch.device):
    """Convert numpy obs (or dict of numpy) to tensor on device."""
    if isinstance(obs, dict):
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in obs.items()}
    return torch.as_tensor(obs, dtype=torch.float32).to(device)


def _obs_to_numpy(obs):
    """Stack a list of single-env obs into a batch."""
    if isinstance(obs, dict):
        return {k: np.stack([o[k] for o in obs]) for k in obs[0]}
    return np.stack(obs)


class PPO:
    """
    Proximal Policy Optimization supporting:
      - no_concept
      - vanilla_freeze
      - concept_actor_critic
    """

    def __init__(
        self,
        method: str,
        env,                          # vectorised gym env (n_envs)
        policy_kwargs: dict,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 256,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        learning_rate: float = 3e-4,
        # concept_actor_critic weights
        lambda_v: float = 0.5,        # concept critic loss weight
        lambda_s: float = 0.5,        # supervised anchor weight
        # training settings
        training_mode: str = "two_phase",   # 'two_phase' | 'end_to_end'
        normalize_advantage: bool = True,
        seed: int = 0,
        device: str = "auto",
        verbose: int = 1,
    ):
        assert method in ("no_concept", "vanilla_freeze", "concept_actor_critic"), (
            f"Unknown method: {method}"
        )
        assert training_mode in ("two_phase", "end_to_end"), (
            f"training_mode must be 'two_phase' or 'end_to_end', got '{training_mode}'"
        )

        self.method = method
        self.training_mode = training_mode
        self.env = env
        self.n_envs = env.num_envs
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.lambda_v = lambda_v
        self.lambda_s = lambda_s
        self.normalize_advantage = normalize_advantage
        self.seed = seed
        self.verbose = verbose

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # temporal_encoding: 'gru', 'stacked', or 'none'
        temporal_encoding = policy_kwargs.pop("temporal_encoding", "none")

        # ---- Policy ----
        self.policy = ActorCriticPolicy(
            **policy_kwargs,
            method=method,
            temporal_encoding=temporal_encoding,
            device=self.device,
        ).to(self.device)
        self.policy.update_lr(learning_rate)

        # ---- Rollout buffer ----
        obs_space   = env.observation_space
        concept_dim = policy_kwargs["concept_dim"]

        # hidden_dim only matters for GRU; use 1 otherwise to avoid large allocations
        if temporal_encoding == "gru":
            hidden_dim = ConceptActorCritic.HIDDEN_DIM
        else:
            hidden_dim = 1

        if hasattr(obs_space, "spaces"):
            obs_shape = {k: v.shape for k, v in obs_space.spaces.items()}
        else:
            obs_shape = obs_space.shape

        self.rollout_buffer = RolloutBuffer(
            buffer_size=n_steps,
            obs_shape=obs_shape,
            concept_dim=concept_dim,
            action_dim=1,
            hidden_dim=hidden_dim,
            n_envs=self.n_envs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=self.device,
        )

        self.temporal_encoding = temporal_encoding
        self.hidden_dim = hidden_dim

        # ---- Tracking ----
        self.num_timesteps = 0
        self.episode_rewards: List[List[float]] = [[] for _ in range(self.n_envs)]
        self.episode_reward_history: List[float] = []
        self._last_obs = None
        self._last_episode_starts = np.ones((self.n_envs,), dtype=bool)
        self._last_hidden = torch.zeros(self.n_envs, hidden_dim, device=self.device)

    # ==================================================================
    # learn
    # ==================================================================

    def learn(
        self,
        total_timesteps: int,
        query_num_times: int = 1,
        query_labels_per_time: int = 500,
    ) -> "PPO":
        """
        Main training loop.

        Label queries happen evenly spaced across training.
        """
        query_interval = total_timesteps // max(query_num_times, 1)
        next_query_at  = query_interval
        query_count    = 0

        obs = self.env.reset()
        self._last_obs = obs
        self._last_episode_starts = np.ones((self.n_envs,), dtype=bool)
        self._last_hidden = torch.zeros(
            self.n_envs, ConceptActorCritic.HIDDEN_DIM, device=self.device
        )

        iteration = 0
        t_start = time.time()

        while self.num_timesteps < total_timesteps:
            # ---- Query concept labels ----
            if (
                self.method != "no_concept"
                and query_count < query_num_times
                and self.num_timesteps >= next_query_at - 1
            ):
                if self.verbose:
                    print(
                        f"[step {self.num_timesteps}] querying {query_labels_per_time} labels "
                        f"(query {query_count+1}/{query_num_times})"
                    )
                labeled_obs, labeled_concepts = self._collect_labels(
                    query_labels_per_time
                )
                self.train_concepts(labeled_obs, labeled_concepts)
                query_count += 1
                next_query_at += query_interval

            # ---- Collect rollout ----
            self.collect_rollouts()
            iteration += 1

            # ---- Train policy ----
            policy_stats = self.train_policy()

            # ---- Logging ----
            if self.verbose and len(self.episode_reward_history) > 0:
                recent = self.episode_reward_history[-100:]
                elapsed = time.time() - t_start
                fps = int(self.num_timesteps / max(elapsed, 1e-6))
                print(
                    f"[iter {iteration}] "
                    f"timesteps={self.num_timesteps}/{total_timesteps}  "
                    f"mean_ep_reward={np.mean(recent):.2f}  "
                    f"n_episodes={len(self.episode_reward_history)}  "
                    f"fps={fps}  "
                    f"pg_loss={policy_stats.get('pg_loss', 0):.4f}  "
                    f"vf_loss={policy_stats.get('vf_loss', 0):.4f}"
                )

        return self

    # ==================================================================
    # collect_rollouts
    # ==================================================================

    def collect_rollouts(self) -> None:
        """
        Run policy for n_steps, fill rollout buffer.
        Carries GRU hidden state across steps for concept_actor_critic.
        Resets h_t at episode boundaries.
        """
        self.policy.set_training_mode(False)
        self.rollout_buffer.reset()

        obs = self._last_obs
        episode_starts = self._last_episode_starts.copy()
        h_t = self._last_hidden.clone()

        for _ in range(self.n_steps):
            obs_tensor = _obs_to_tensor(obs, self.device)

            with torch.no_grad():
                # Reset hidden state at episode boundaries
                if self.method == "concept_actor_critic":
                    reset_mask = torch.as_tensor(
                        episode_starts, dtype=torch.float32, device=self.device
                    ).unsqueeze(1)
                    h_t = h_t * (1.0 - reset_mask)
                    # Full forward pass
                    features = self.policy.extract_features(obs_tensor)
                    c_t, h_new, concept_dists, V_c = self.policy.concept_net(
                        features, h_t
                    )
                    latent = self.policy.mlp_extractor(c_t)
                    action_logits = self.policy.action_net(latent)
                    dist = torch.distributions.Categorical(logits=action_logits)
                    actions   = dist.sample()
                    log_probs = dist.log_prob(actions)
                    values    = self.policy.value_net(latent).flatten()

                    concept_log_prob = (
                        self.policy.concept_net.concept_log_probs(concept_dists, c_t)
                        .cpu().numpy()
                    )
                    concept_value = V_c
                    h_t = h_new
                else:
                    actions, values, log_probs, h_new = self.policy.forward(
                        obs_tensor
                    )
                    h_new = None  # not used
                    concept_log_prob = None
                    concept_value = None

            # Step environment
            np_actions = actions.cpu().numpy()
            next_obs, rewards, dones, infos = self.env.step(np_actions)

            # Track episode rewards
            for i in range(self.n_envs):
                self.episode_rewards[i].append(rewards[i])
                if dones[i]:
                    ep_rew = sum(self.episode_rewards[i])
                    self.episode_reward_history.append(ep_rew)
                    self.episode_rewards[i] = []

            # Collect ground-truth concepts from each env
            concepts = self._get_concepts_from_infos(infos)

            # Store
            self.rollout_buffer.add(
                obs=obs if not isinstance(obs, dict)
                    else {k: v.copy() for k, v in obs.items()},
                concept=concepts,
                action=np_actions.reshape(self.n_envs, 1),
                reward=rewards,
                episode_start=episode_starts.astype(np.float32),
                value=values,
                log_prob=log_probs,
                hidden_state=(
                    h_t.cpu().numpy()
                    if self.method == "concept_actor_critic" else None
                ),
                concept_value=concept_value,
                concept_log_prob=concept_log_prob,
            )

            self.num_timesteps += self.n_envs
            obs = next_obs
            episode_starts = dones.copy()

            if self.method == "concept_actor_critic":
                # reset hidden state for done envs after storing
                reset_mask = torch.as_tensor(
                    dones, dtype=torch.float32, device=self.device
                ).unsqueeze(1)
                h_t = h_t * (1.0 - reset_mask)

        self._last_obs = obs
        self._last_episode_starts = episode_starts
        self._last_hidden = h_t

        # Compute GAE
        with torch.no_grad():
            obs_tensor = _obs_to_tensor(obs, self.device)
            if self.method == "concept_actor_critic":
                reset_mask = torch.as_tensor(
                    episode_starts, dtype=torch.float32, device=self.device
                ).unsqueeze(1)
                h_t_final = h_t * (1.0 - reset_mask)
                features = self.policy.extract_features(obs_tensor)
                c_t, _, _, V_c_last = self.policy.concept_net(features, h_t_final)
                latent = self.policy.mlp_extractor(c_t)
                last_values = self.policy.value_net(latent).flatten()
                last_concept_values = V_c_last.flatten()
            else:
                _, last_values, _, _ = self.policy.forward(obs_tensor)
                last_concept_values = torch.zeros(self.n_envs, device=self.device)

        self.rollout_buffer.compute_returns_and_advantage(
            last_values, dones=episode_starts.astype(np.float32)
        )
        if self.method == "concept_actor_critic":
            self.rollout_buffer.compute_concept_returns_and_advantage(
                last_concept_values, dones=episode_starts.astype(np.float32)
            )

    # ==================================================================
    # train_policy
    # ==================================================================

    def train_policy(self) -> dict:
        """
        Standard PPO clipped surrogate loss.

        Optimizer selection:
          two_phase   — optimizer_exclude_concept: concept net frozen, updated only in
                        train_concepts(). Mirrors LICORICE vanilla_freeze training.
          end_to_end  — full optimizer: policy gradient flows through concept net,
                        updating it jointly with the actor/critic heads.
          no_concept  — always uses full optimizer (no concept net to freeze).
        """
        self.policy.set_training_mode(True)

        if self.method != "no_concept" and self.training_mode == "two_phase":
            optimizer = self.policy.optimizer_exclude_concept
        else:
            optimizer = self.policy.optimizer

        pg_losses, vf_losses, ent_losses = [], [], []

        for _ in range(self.n_epochs):
            for batch in self.rollout_buffer.get(self.batch_size):
                obs     = batch["observations"]
                actions = batch["actions"].long().flatten()
                h_prev  = batch["hidden_states"] if self.method == "concept_actor_critic" else None

                _, values, log_prob, entropy, _, _, _ = self.policy.evaluate_actions(
                    obs, actions, h_prev
                )

                advantages = batch["advantages"]
                if self.normalize_advantage and advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                old_log_prob = batch["log_probs"]
                ratio = torch.exp(log_prob - old_log_prob)

                pg_loss1 = advantages * ratio
                pg_loss2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                pg_loss  = -torch.min(pg_loss1, pg_loss2).mean()

                vf_loss  = F.mse_loss(values, batch["returns"])

                if entropy is None:
                    ent_loss = -torch.mean(-log_prob)
                else:
                    ent_loss = -entropy.mean()

                loss = pg_loss + self.vf_coef * vf_loss + self.ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())

        return {
            "pg_loss": float(np.mean(pg_losses)),
            "vf_loss": float(np.mean(vf_losses)),
            "ent_loss": float(np.mean(ent_losses)),
        }

    # ==================================================================
    # train_concepts
    # ==================================================================

    def train_concepts(
        self,
        labeled_obs,
        labeled_concepts: np.ndarray,
        n_epochs: int = 50,
        batch_size: int = 64,
    ) -> dict:
        """
        Concept training — method-specific.

        vanilla_freeze:
            supervised CrossEntropy/MSE on labeled samples.

        concept_actor_critic:
            PPO-style concept actor loss  +  concept critic loss  +  supervised anchor.
        """
        if self.method == "no_concept":
            return {}

        self.policy.set_training_mode(True)
        concept_net = self.policy.concept_net
        optimizer   = self.policy.optimizer

        n_labeled = labeled_concepts.shape[0]
        concept_losses = []

        for epoch in range(n_epochs):
            idx = np.random.permutation(n_labeled)
            epoch_losses = []

            for start in range(0, n_labeled, batch_size):
                bidx = idx[start: start + batch_size]
                if isinstance(labeled_obs, dict):
                    obs_b = {k: torch.as_tensor(
                        labeled_obs[k][bidx], dtype=torch.float32
                    ).to(self.device) for k in labeled_obs}
                else:
                    obs_b = torch.as_tensor(
                        labeled_obs[bidx], dtype=torch.float32
                    ).to(self.device)
                c_b = torch.as_tensor(
                    labeled_concepts[bidx], dtype=torch.float32
                ).to(self.device)

                features = self.policy.extract_features(obs_b)

                if self.method == "vanilla_freeze":
                    logits = concept_net.get_logits(features)
                    loss   = concept_net.compute_loss(logits, c_b)

                elif self.method == "concept_actor_critic":
                    # ---- Concept actor-critic loss (from rollout buffer) ----
                    ac_loss = torch.tensor(0.0, device=self.device)
                    if self.rollout_buffer.full:
                        buf_batch = next(iter(self._get_concept_batches(batch_size)))
                        h_buf    = buf_batch["hidden_states"]
                        obs_buf  = buf_batch["observations"]
                        feat_buf = self.policy.extract_features(obs_buf)

                        c_pred, h_new_buf, concept_dists, V_c = concept_net(
                            feat_buf, h_buf
                        )
                        c_adv = buf_batch["concept_advantages"]
                        if c_adv.numel() > 1:
                            c_adv = (c_adv - c_adv.mean()) / (c_adv.std() + 1e-8)

                        old_clp = buf_batch["concept_log_probs"]
                        new_clp = concept_net.concept_log_probs(concept_dists, c_pred)
                        ratio   = torch.exp(new_clp - old_clp)

                        ac_loss1 = c_adv * ratio
                        ac_loss2 = c_adv * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                        actor_loss = -torch.min(ac_loss1, ac_loss2).mean()

                        c_ret = buf_batch["concept_returns"]
                        critic_loss = F.mse_loss(V_c.flatten(), c_ret)
                        ac_loss = actor_loss + self.lambda_v * critic_loss

                    # ---- Supervised anchor ----
                    logits, _ = concept_net.get_logits(features)
                    sup_loss   = concept_net.compute_concept_loss(logits, c_b)

                    loss = ac_loss + self.lambda_s * sup_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                optimizer.step()
                epoch_losses.append(loss.item())

            concept_losses.append(float(np.mean(epoch_losses)))

        if self.verbose:
            print(
                f"  [{self.method}] concept training: "
                f"initial_loss={concept_losses[0]:.4f}  "
                f"final_loss={concept_losses[-1]:.4f} "
                f"over {n_epochs} epochs"
            )
        return {"concept_loss_history": concept_losses}

    # ==================================================================
    # helpers
    # ==================================================================

    def _get_concept_batches(self, batch_size: int):
        """Tiny wrapper so we can pull one batch from buffer during concept training."""
        return self.rollout_buffer.get(batch_size)

    def _collect_labels(
        self, n_samples: int
    ) -> Tuple[object, np.ndarray]:
        """
        Collect n_samples (obs, concept) pairs by rolling out the current policy
        in the vectorised env.  Ground-truth concepts come from env.get_concept()
        (accessed via the info dict or the env wrapper attribute).
        """
        obs_list: List = []
        concept_list: List[np.ndarray] = []

        # Use a fresh single-env rollout: reset env and step
        obs = self._last_obs
        h_t = torch.zeros(
            self.n_envs, ConceptActorCritic.HIDDEN_DIM, device=self.device
        )
        collected = 0

        self.policy.set_training_mode(False)
        while collected < n_samples:
            obs_t = _obs_to_tensor(obs, self.device)
            with torch.no_grad():
                if self.method == "concept_actor_critic":
                    actions, h_new = self.policy.predict(obs_t, h_t)
                    h_t = h_new
                else:
                    actions, _ = self.policy.predict(obs_t)

            next_obs, _, dones, infos = self.env.step(actions.cpu().numpy())
            concepts = self._get_concepts_from_infos(infos)

            # Randomly subsample to keep memory manageable
            for i in range(self.n_envs):
                if collected >= n_samples:
                    break
                if isinstance(obs, dict):
                    obs_list.append({k: obs[k][i] for k in obs})
                else:
                    obs_list.append(obs[i])
                concept_list.append(concepts[i])
                collected += 1
                if dones[i]:
                    h_t[i] = 0.0

            obs = next_obs

        # Stack observations
        if isinstance(obs_list[0], dict):
            labeled_obs = {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
        else:
            labeled_obs = np.stack(obs_list)

        labeled_concepts = np.stack(concept_list)
        return labeled_obs, labeled_concepts

    def _get_concepts_from_infos(self, infos) -> np.ndarray:
        """
        Try to extract ground-truth concepts from step infos.
        Supports:
          - info dict containing 'concept' key
          - vectorised envs exposing .get_attr('get_concept')
        Falls back to zeros if unavailable.
        """
        concept_dim = self.rollout_buffer.concept_dim
        concepts = np.zeros((self.n_envs, concept_dim), dtype=np.float32)

        # Try info dict
        if isinstance(infos, (list, tuple)) and isinstance(infos[0], dict):
            for i, info in enumerate(infos):
                if "concept" in info:
                    concepts[i] = info["concept"]
            return concepts

        # Try get_attr on VecEnv
        try:
            raw = self.env.get_attr("current_concept")
            for i, c in enumerate(raw):
                if c is not None:
                    concepts[i] = np.asarray(c, dtype=np.float32)
        except Exception:
            pass

        return concepts

    # ==================================================================
    # evaluation helpers
    # ==================================================================

    def evaluate(
        self, n_episodes: int = 20, deterministic: bool = True
    ) -> Tuple[float, float]:
        """Run n_episodes in env, return (mean_reward, std_reward)."""
        rewards = []
        obs = self.env.reset()
        h_t = torch.zeros(
            self.n_envs, ConceptActorCritic.HIDDEN_DIM, device=self.device
        )
        ep_rewards = [0.0] * self.n_envs
        done_count = 0
        self.policy.set_training_mode(False)

        while done_count < n_episodes:
            obs_t = _obs_to_tensor(obs, self.device)
            with torch.no_grad():
                if self.method == "concept_actor_critic":
                    actions, h_t = self.policy.predict(obs_t, h_t, deterministic)
                else:
                    actions, _ = self.policy.predict(obs_t, deterministic=deterministic)

            obs, r, dones, _ = self.env.step(actions.cpu().numpy())
            for i in range(self.n_envs):
                ep_rewards[i] += r[i]
                if dones[i]:
                    rewards.append(ep_rewards[i])
                    ep_rewards[i] = 0.0
                    done_count += 1
                    h_t[i] = 0.0

        return float(np.mean(rewards)), float(np.std(rewards))
