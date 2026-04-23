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
    return torch.as_tensor(np.asarray(obs), dtype=torch.float32).to(device)


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
      - gvf
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
        concept_ac_epochs: Optional[int] = None,  # epochs for concept AC update; defaults to n_epochs
        # training settings
        training_mode: str = "two_phase",   # 'two_phase' | 'end_to_end'
        normalize_advantage: bool = True,
        seed: int = 0,
        device: str = "auto",
        verbose: int = 1,
    ):
        assert method in ("no_concept", "vanilla_freeze", "concept_actor_critic", "gvf"), (
            f"Unknown method: {method}"
        )
        assert training_mode in ("two_phase", "end_to_end", "joint"), (
            f"training_mode must be 'two_phase', 'end_to_end', or 'joint', got '{training_mode}'"
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
        self.concept_ac_epochs = concept_ac_epochs if concept_ac_epochs is not None else n_epochs
        self.normalize_advantage = normalize_advantage
        self.seed = seed
        self.verbose = verbose
        concept_dim = int(policy_kwargs["concept_dim"])
        gvf_pairing_from_kwargs = policy_kwargs.pop("gvf_pairing", None)
        if gvf_pairing_from_kwargs is not None:
            self.gvf_pairing = list(gvf_pairing_from_kwargs)
        else:
            self.gvf_pairing = list(range(concept_dim))

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # temporal_encoding: 'gru', 'stacked', or 'none'
        temporal_encoding = policy_kwargs.pop("temporal_encoding", "none")
        # concept_names is used for logging only; not a policy parameter
        concept_names_kwarg = policy_kwargs.pop("concept_names", None)

        # -------------------------------------------------------------------
        # THE FIX: Dynamically extract the ground truth from the environment
        # -------------------------------------------------------------------
        if hasattr(self.env, "task_types"):
            policy_kwargs["task_types"] = self.env.task_types
        if hasattr(self.env, "num_classes"):
            policy_kwargs["num_classes"] = self.env.num_classes
        if hasattr(self.env, "temporal_concepts"):
            policy_kwargs["temporal_concepts"] = self.env.temporal_concepts
            
        # Optional but highly recommended: Auto-sync the logger names too!
        if hasattr(self.env, "concept_names") and concept_names_kwarg is None:
            concept_names_kwarg = self.env.concept_names

        # ---- Policy ----
        self.policy = ActorCriticPolicy(
            **policy_kwargs,
            method=method,
            temporal_encoding=temporal_encoding,
            gvf_pairing=self.gvf_pairing,
        ).to(self.device)
        self.policy.update_lr(learning_rate)

        assert all(
            isinstance(pairing, (int, np.integer)) and 0 <= int(pairing) < concept_dim
            for pairing in self.gvf_pairing
        ), "gvf_pairing must contain valid concept indices in [0, concept_dim - 1]"

        # ---- Rollout buffer ----
        obs_shape   = policy_kwargs["obs_shape"]
        concept_dim = policy_kwargs["concept_dim"]

        # hidden_dim only matters for GRU; use 1 otherwise to avoid large allocations
        if temporal_encoding == "gru":
            hidden_dim = ConceptActorCritic.HIDDEN_DIM
        else:
            hidden_dim = 1

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

        # concept_names for logging (optional, falls back to indices)
        self.concept_names: List[str] = concept_names_kwarg if concept_names_kwarg is not None else (
            [f"c{i}" for i in range(policy_kwargs["concept_dim"])]
        )

        # ---- Tracking ----
        self.num_timesteps = 0
        self.episode_rewards: List[List[float]] = [[] for _ in range(self.n_envs)]
        self.episode_reward_history: List[float] = []
        # concept_acc_log: list of (timestep, {concept_name: metric})
        self.concept_acc_log: List[Tuple[int, dict]] = []
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

        Concept labels are queried once at the start of training only.
        """
        query_interval = total_timesteps // max(query_num_times, 1)
        next_query_at  = 0
        query_count    = 0

        obs, _ = self.env.reset()
        self._last_obs = obs
        self._last_episode_starts = np.ones((self.n_envs,), dtype=bool)
        self._last_hidden = torch.zeros(
            self.n_envs, self.hidden_dim, device=self.device
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

            # ---- Joint concept supervision from rollout buffer (every iteration) ----
            if self.method != "no_concept" and self.training_mode == "joint":
                if self.temporal_encoding == "gru":
                    # BPTT: process buffer as a sequence so gradients flow
                    # through the GRU across time steps
                    self._train_concepts_bptt(n_epochs=self.n_epochs)
                else:
                    buf = self.rollout_buffer
                    T, N = buf.buffer_size, buf.n_envs
                    if buf.obs_is_dict:
                        obs_flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in buf.observations.items()}
                    else:
                        obs_flat = buf.observations.reshape(T * N, *buf.obs_shape)
                    con_flat = buf.concepts.reshape(T * N, buf.concept_dim)
                    self.train_concepts(obs_flat, con_flat, n_epochs=self.n_epochs, batch_size=self.batch_size)

            # ---- Train concept actor-critic (every iteration, mirrors train_policy) ----
            gvf_stats = self.train_gvf()
            concept_ac_stats = self.train_concept_actor_critic()

            # ---- Concept accuracy tracking (every 10 iters) ----
            concept_loss_log = {}
            if self.method != "no_concept" and iteration % 10 == 0:
                acc = self._compute_concept_accuracy_from_buffer()
                if acc:
                    self.concept_acc_log.append((self.num_timesteps, acc))
                    # Also compute raw MSE per concept for logging
                    concept_loss_log = self._compute_concept_mse_from_buffer()

            # ---- Logging ----
            if self.verbose and len(self.episode_reward_history) > 0:
                recent = self.episode_reward_history[-100:]
                elapsed = time.time() - t_start
                fps = int(self.num_timesteps / max(elapsed, 1e-6))
                extra = ""
                if concept_ac_stats:
                    extra = (
                        f"  ca_loss={concept_ac_stats.get('concept_actor_loss', 0):.4f}"
                        f"  cc_loss={concept_ac_stats.get('concept_critic_loss', 0):.4f}"
                        f"  ce_loss={concept_ac_stats.get('concept_ent_loss', 0):.4f}"
                    )
                if gvf_stats:
                    extra += f"  gvf_loss={gvf_stats.get('gvf_loss', 0):.4f}"
                if concept_loss_log:
                    mse_str = "  ".join(
                        f"{n}={v:.5f}" for n, v in concept_loss_log.items()
                    )
                    extra += f"  mse=[{mse_str}]"
                print(
                    f"[iter {iteration}] "
                    f"timesteps={self.num_timesteps}/{total_timesteps}  "
                    f"mean_ep_reward={np.mean(recent):.2f}  "
                    f"n_episodes={len(self.episode_reward_history)}  "
                    f"fps={fps}  "
                    f"pg_loss={policy_stats.get('pg_loss', 0):.4f}  "
                    f"vf_loss={policy_stats.get('vf_loss', 0):.4f}"
                    f"{extra}"
                )

        return self

    # ==================================================================
    # collect_rollouts
    # ==================================================================

    def collect_rollouts(self) -> None:
        """
        Run policy for n_steps, fill rollout buffer.
        Carries GRU hidden state across steps for concept_actor_critic/gvf.
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
                # c_t = None
                # h_prev_for_buffer = h_t.clone() if self.temporal_encoding == "gru" else None
                if self.method == "concept_actor_critic":
                    # Reset hidden state at episode boundaries
                    if self.temporal_encoding == "gru":
                        reset_mask = torch.as_tensor(
                            episode_starts, dtype=torch.float32, device=self.device
                        ).unsqueeze(1)
                        h_t = h_t * (1.0 - reset_mask)
                    # Save h_PREV (pre-GRU) for buffer storage — must be before concept_net call
                    h_prev_for_buffer = h_t.clone()
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
                    if h_new is not None:
                        h_t = h_new
                elif self.method == "vanilla_freeze" and self.temporal_encoding == "gru":
                    # vanilla_freeze + GRU: track h_t so BPTT training has valid sequences
                    reset_mask = torch.as_tensor(
                        episode_starts, dtype=torch.float32, device=self.device
                    ).unsqueeze(1)
                    h_t = h_t * (1.0 - reset_mask)
                    h_prev_for_buffer = h_t.clone()
                    features = self.policy.extract_features(obs_tensor)
                    c_t, h_new = self.policy.concept_net(features, h_t)
                    latent = self.policy.mlp_extractor(c_t)
                    action_logits = self.policy.action_net(latent)
                    dist = torch.distributions.Categorical(logits=action_logits)
                    actions   = dist.sample()
                    log_probs = dist.log_prob(actions)
                    values    = self.policy.value_net(latent).flatten()
                    concept_log_prob = None
                    concept_value    = None
                    if h_new is not None:
                        h_t = h_new
                else:
                    h_prev_for_buffer = h_t.clone()
                    actions, values, log_probs, h_new = self.policy.forward(obs_tensor)
                    h_new = None
                    concept_log_prob = None
                    concept_value    = None

            # Collect ground-truth concepts for CURRENT observation (before stepping)
            concepts = self._get_current_concepts()

            # Collect concept_reward_active mask from env (e.g. TMaze fires only
            # at the junction).  Used both for concept AC reward and for
            # filtering the concept accuracy evaluation metric.
            try:
                eval_mask = np.array(
                    self.env.get_attr("concept_reward_active"), dtype=np.float32
                )
            except Exception:
                eval_mask = np.ones(self.n_envs, dtype=np.float32)

            concept_reward = None
            if self.method == "concept_actor_critic" and c_t is not None:
                concept_reward = self._compute_concept_reward(
                    c_t.cpu().numpy(), concepts
                ) * eval_mask

            # Step environment
            np_actions = actions.cpu().numpy()
            next_obs, rewards, terminated, truncated, infos = self.env.step(np_actions)
            dones = np.logical_or(terminated, truncated)

            # Track episode rewards
            for i in range(self.n_envs):
                self.episode_rewards[i].append(rewards[i])
                if dones[i]:
                    ep_rew = sum(self.episode_rewards[i])
                    self.episode_reward_history.append(ep_rew)
                    self.episode_rewards[i] = []

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
                hidden_state=(h_prev_for_buffer.cpu().numpy() if h_prev_for_buffer is not None else None),
                concept_value=concept_value,
                concept_log_prob=concept_log_prob,
                concept_reward=concept_reward,
                concept_eval_mask=eval_mask,
            )

            self.num_timesteps += self.n_envs
            obs = next_obs
            episode_starts = dones.copy()

            if self.temporal_encoding == "gru":
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
            elif self.method == "vanilla_freeze" and self.temporal_encoding == "gru":
                reset_mask = torch.as_tensor(
                    episode_starts, dtype=torch.float32, device=self.device
                ).unsqueeze(1)
                h_t_final = h_t * (1.0 - reset_mask)
                features = self.policy.extract_features(obs_tensor)
                c_t, _ = self.policy.concept_net(features, h_t_final)
                latent = self.policy.mlp_extractor(c_t)
                last_values = self.policy.value_net(latent).flatten()
                last_concept_values = torch.zeros(self.n_envs, device=self.device)
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
        self.policy.set_training_mode(True)

        if self.method != "no_concept" and self.training_mode in ("two_phase", "joint"):
            optimizer = self.policy.optimizer_exclude_concept
        else:
            optimizer = self.policy.optimizer

        pg_losses, vf_losses, ent_losses = [], [], []
        seq_len = 8  # Match this to your buffer's seq_len

        for _ in range(self.n_epochs):
            for batch in self.rollout_buffer.get(self.batch_size):
                obs     = batch["observations"]
                actions = batch["actions"].long().flatten()
                h_prev  = batch["hidden_states"] if self.temporal_encoding == "gru" else None

                _, values, log_prob, entropy, _, _, _ = self.policy.evaluate_actions(
                    obs, actions, h_prev
                )

                _, values, log_prob, entropy, _, _, _ = self.policy.evaluate_actions(
                    obs, actions, h_prev
                )

                # --- STANDARD PPO LOSS ---
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
    # train_gvf
    # ==================================================================

    def train_gvf(self) -> dict:
        """
        Train GVF heads to predict discounted cumulant expectations.

        Each GVF head j is paired with one concept index via gvf_pairing[j].
        The paired concept value in rollout_buffer.concepts is treated as the
        cumulant c_t, and the training target is:

          G_t = c_t + gamma * (1 - done_t) * G_{t+1}

        where done_t is inferred from episode starts at the next step.
        """
        if self.method != "gvf":
            print("Warning: train_gvf called but method is not gvf")
            return {}
        if not self.rollout_buffer.full:
            print("Warning: train_gvf called but rollout buffer is not full")
            return {}
        if self.gvf_pairing is None or len(self.gvf_pairing) == 0:
            print("Warning: train_gvf called but gvf_pairing is not set")
            return {}
        if self.policy.concept_net is None or not hasattr(self.policy.concept_net, "get_gvf_logits"):
            print("Warning: train_gvf called but concept_net does not have get_gvf_logits")
            return {}

        self.policy.set_training_mode(True)
        optimizer = self.policy.optimizer_concept_and_features
        if optimizer is None:
            return {}

        T = self.rollout_buffer.buffer_size
        N = self.rollout_buffer.n_envs
        num_gvf = len(self.gvf_pairing)
        concept_dim = self.rollout_buffer.concept_dim

        # Convert pairing to concept indices (STRICTLY 0-based).
        pairing_indices = []
        for pairing in self.gvf_pairing:
            idx = int(pairing)
            if idx < 0 or idx >= concept_dim:
                raise ValueError(
                    f"Invalid gvf_pairing index {pairing}; expected 0-based concept index in "
                    f"[0, {concept_dim - 1}]."
                )
            pairing_indices.append(idx)

        concepts = self.rollout_buffer.concepts  # [T, N, concept_dim]
        episode_starts = self.rollout_buffer.episode_starts  # [T, N]

        # Build discounted targets G_t for each GVF head and env.
        gvf_targets = np.zeros((T, N, num_gvf), dtype=np.float32)
        running = np.zeros((N, num_gvf), dtype=np.float32)
        for step in reversed(range(T)):
            if step < T - 1:
                # If next step starts a new episode, bootstrap is zero.
                running *= (1.0 - episode_starts[step + 1])[:, None]
            # Two-step index → [N, num_gvf]; concepts[step,:,idx] is [num_gvf,N] in numpy.
            cumulants_t = concepts[step][:, pairing_indices]
            running = cumulants_t + self.gamma * running
            gvf_targets[step] = running

        # Match RolloutBuffer flattening order: [T, N, ...] -> [T*N, ...].
        gvf_targets_flat = gvf_targets.swapaxes(0, 1).reshape(T * N, num_gvf)

        if self.rollout_buffer.obs_is_dict:
            obs_flat = {
                k: self.rollout_buffer.observations[k].swapaxes(0, 1).reshape(T * N, *v.shape[2:])
                for k, v in self.rollout_buffer.observations.items()
            }
        else:
            obs_flat = self.rollout_buffer.observations.swapaxes(0, 1).reshape(
                T * N, *self.rollout_buffer.observations.shape[2:]
            )
        hidden_flat = self.rollout_buffer.hidden_states.swapaxes(0, 1).reshape(T * N, self.hidden_dim)

        gvf_losses = []
        concept_net = self.policy.concept_net
        total = T * N
        for _ in range(self.n_epochs):
            perm = np.random.permutation(total)
            for start in range(0, total, self.batch_size):
                bidx = perm[start: start + self.batch_size]

                if isinstance(obs_flat, dict):
                    obs_b = {
                        k: torch.as_tensor(obs_flat[k][bidx], dtype=torch.float32, device=self.device)
                        for k in obs_flat
                    }
                else:
                    obs_b = torch.as_tensor(obs_flat[bidx], dtype=torch.float32, device=self.device)

                h_prev = None
                if self.temporal_encoding == "gru":
                    h_prev = torch.as_tensor(
                        hidden_flat[bidx], dtype=torch.float32, device=self.device
                    )
                target_batch = torch.as_tensor(
                    gvf_targets_flat[bidx], dtype=torch.float32, device=self.device
                )

                features = self.policy.extract_features(obs_b)
                gvf_logits, _ = concept_net.get_gvf_logits(features, h_prev)
                gvf_pred = torch.cat([g.squeeze(-1).unsqueeze(1) for g in gvf_logits], dim=1)
                loss = F.mse_loss(gvf_pred, target_batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.features_extractor.parameters()) +
                    list(concept_net.parameters()),
                    self.max_grad_norm,
                )
                optimizer.step()
                gvf_losses.append(loss.item())

        return {"gvf_loss": float(np.mean(gvf_losses))} if gvf_losses else {}

    # ==================================================================
    # train_gvf
    # ==================================================================

    def train_concept_actor_critic(self) -> dict:
        """
        PPO-style update for the concept actor and concept critic.

        Mirrors train_policy() exactly:
          - called every iteration after collect_rollouts()
          - iterates over the full rollout buffer for n_epochs
          - uses clipped surrogate on concept log-probs weighted by concept advantages
          - fits concept critic V_c to concept returns

        This makes the concept actor-critic symmetric with the task actor-critic:
          task actor   ← updated by value critic advantages    (train_policy)
          concept actor ← updated by concept critic advantages (train_concept_actor_critic)
        """
        if self.method != "concept_actor_critic":
            return {}

        # Use BPTT sequence training for GRU so gradients flow across time steps
        if self.temporal_encoding == "gru":
            return self._train_concept_ac_bptt()

        self.policy.set_training_mode(True)
        concept_net = self.policy.concept_net
        optimizer = self.policy.optimizer_concept_and_features

        actor_losses, critic_losses, ent_losses = [], [], []
        seq_len = 8

        for _ in range(self.n_epochs):
            for batch in self.rollout_buffer.get(self.batch_size):
                obs    = batch["observations"]
                
                # 1. Bulk extract features
                features = self.policy.extract_features(obs)
                
                # --- BPTT SEQUENCE UNROLLING ---
                if self.temporal_encoding == "gru":
                    B_seq = self.batch_size // seq_len
                    features_seq = features.view(B_seq, seq_len, -1)
                    
                    # 2. Get initial hidden state
                    h_prev = batch["hidden_states"].view(B_seq, seq_len, -1)[:, 0, :].contiguous()
                    
                    V_c_list, new_clp_list, ent_list = [], [], []
                    
                    # 3. Unroll
                    for t in range(seq_len):
                        feat_t = features_seq[:, t]
                        c_p, h_prev, d_p, v_p = concept_net(feat_t, h_prev)
                        
                        clp_t = concept_net.concept_log_probs(d_p, c_p)
                        # We calculate entropy here and apply the negative sign to match standard logic
                        ent_t = -torch.stack([d.entropy() for d in d_p], dim=1).mean()
                        
                        V_c_list.append(v_p)
                        new_clp_list.append(clp_t)
                        ent_list.append(ent_t)
                        
                    # 4. Restack
                    V_c = torch.stack(V_c_list, dim=1).flatten()
                    new_clp = torch.stack(new_clp_list, dim=1).flatten()
                    concept_ent_loss = torch.stack(ent_list).mean()
                else:
                    h_prev = None
                    c_pred, _, concept_dists, V_c = concept_net(features, h_prev)
                    new_clp = concept_net.concept_log_probs(concept_dists, c_pred)
                    concept_ent_loss = -torch.stack([d.entropy() for d in concept_dists], dim=1).mean()

                # ---- Concept actor loss (PPO-clipped) ----
                c_adv = batch["concept_advantages"]
                if self.normalize_advantage and c_adv.numel() > 1:
                    c_adv = (c_adv - c_adv.mean()) / (c_adv.std() + 1e-8)

                old_clp = batch["concept_log_probs"]
                ratio   = torch.exp(new_clp - old_clp)

                ac_loss1 = c_adv * ratio
                ac_loss2 = c_adv * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                actor_loss = -torch.min(ac_loss1, ac_loss2).mean()

                # ---- Concept critic loss (PPO-style value clipping) ----
                # Mirrors standard PPO: prevent V_c from jumping more than clip_range
                # per update by taking the max of clipped and unclipped MSE losses.
                c_ret = batch["concept_returns"]
                V_c_flat = V_c.flatten()
                V_c_old  = batch["concept_values"]
                V_c_clipped = V_c_old + torch.clamp(
                    V_c_flat - V_c_old, -self.clip_range, self.clip_range
                )
                critic_loss = torch.max(
                    F.mse_loss(V_c_flat,    c_ret),
                    F.mse_loss(V_c_clipped, c_ret),
                ).mean()

                # ---- Concept actor entropy (mirrors ent_coef * ent_loss in train_policy) ----
                concept_ent_loss = -torch.stack(
                    [d.entropy() for d in concept_dists], dim=1
                ).mean()

                loss = actor_loss + self.lambda_v * critic_loss + self.ent_coef * concept_ent_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.features_extractor.parameters()) +
                    list(concept_net.parameters()),
                    self.max_grad_norm,
                )
                optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                ent_losses.append(concept_ent_loss.item())

        return {
            "concept_actor_loss":  float(np.mean(actor_losses)),
            "concept_critic_loss": float(np.mean(critic_losses)),
            "concept_ent_loss":    float(np.mean(ent_losses)),
        }
    
    # ==================================================================
    # train_concepts  (supervised anchor — called at label query times)
    # ==================================================================

    def train_concepts(
        self,
        labeled_obs,
        labeled_concepts: np.ndarray,
        n_epochs: int = 50,
        batch_size: int = 64,
    ) -> dict:
        """
        Supervised concept training on labeled samples (called at query times only).

        vanilla_freeze:       CrossEntropy/MSE on labeled samples.
        concept_actor_critic: supervised anchor only — the AC update runs every
                              iteration via train_concept_actor_critic().
        """
        if self.method == "no_concept":
            return {}

        self.policy.set_training_mode(True)
        concept_net = self.policy.concept_net
        # Both methods: update features_extractor + concept_net only.
        # mlp_extractor / action_net / value_net have zero concept gradients and
        # are updated by train_policy instead.
        optimizer = self.policy.optimizer_concept_and_features

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
                    logits, _ = concept_net.get_logits(features, None)
                    loss   = concept_net.compute_loss(logits, c_b)
                else:
                    # concept_actor_critic: supervised anchor weighted by lambda_s
                    logits, _ = concept_net.get_logits(features)
                    loss = self.lambda_s * concept_net.compute_concept_loss(logits, c_b)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.features_extractor.parameters()) +
                    list(concept_net.parameters()),
                    self.max_grad_norm,
                )
                optimizer.step()
                epoch_losses.append(loss.item())

            concept_losses.append(float(np.mean(epoch_losses)))

        if self.verbose:
            print(
                f"  [{self.method}] supervised anchor: "
                f"initial_loss={concept_losses[0]:.4f}  "
                f"final_loss={concept_losses[-1]:.4f} "
                f"over {n_epochs} epochs"
            )
        return {"concept_loss_history": concept_losses}

    # ==================================================================
    # helpers
    # ==================================================================

    # ==================================================================
    # BPTT sequence training helpers
    # ==================================================================

    def _train_concepts_bptt(self, n_epochs: int = 1) -> dict:
        """
        Supervised concept training via BPTT over the full rollout sequence.

        Replaces random-batch train_concepts for GRU methods in joint mode.
        Processes all N envs in parallel as sequences of T steps, resetting
        h_t at episode boundaries.  Gradients flow back through the GRU
        across time so it can learn to latch and maintain temporal concepts.
        """
        if not self.rollout_buffer.full:
            return {}

        self.policy.set_training_mode(True)
        buf = self.rollout_buffer
        T, N = buf.buffer_size, buf.n_envs
        concept_net = self.policy.concept_net
        optimizer = self.policy.optimizer_concept_and_features
        epoch_losses = []

        for _ in range(n_epochs):
            h_t = torch.zeros(N, concept_net.hidden_dim, device=self.device)
            step_losses: List[torch.Tensor] = []

            for t in range(T):
                if buf.obs_is_dict:
                    obs_t = {k: torch.as_tensor(buf.observations[k][t], dtype=torch.float32).to(self.device)
                             for k in buf.observations}
                else:
                    obs_t = torch.as_tensor(buf.observations[t], dtype=torch.float32).to(self.device)

                c_true_t = torch.as_tensor(buf.concepts[t], dtype=torch.float32).to(self.device)
                ep_start = torch.as_tensor(
                    buf.episode_starts[t], dtype=torch.float32, device=self.device
                ).unsqueeze(1)
                h_t = h_t * (1.0 - ep_start)

                features = self.policy.extract_features(obs_t)

                if self.method == "vanilla_freeze":
                    logits, h_t = concept_net.get_logits(features, h_t)
                    loss_t = concept_net.compute_loss(logits, c_true_t)
                else:
                    logits, h_t = concept_net.get_logits(features, h_t)
                    loss_t = self.lambda_s * concept_net.compute_concept_loss(logits, c_true_t)

                step_losses.append(loss_t)

            total_loss = torch.stack(step_losses).mean()
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.features_extractor.parameters()) +
                list(concept_net.parameters()),
                self.max_grad_norm,
            )
            optimizer.step()
            epoch_losses.append(total_loss.item())

        return {"concept_bptt_loss": float(np.mean(epoch_losses))}

    def _train_concept_ac_bptt(self) -> dict:
        """
        PPO-style concept actor-critic update via BPTT over the rollout sequence.

        Replaces the random-batch train_concept_actor_critic for GRU methods.
        Same loss as the batch version but computed over the full T-step sequence
        so the GRU receives gradients through time.
        """
        if not self.rollout_buffer.full:
            return {}

        self.policy.set_training_mode(True)
        buf = self.rollout_buffer
        T, N = buf.buffer_size, buf.n_envs
        concept_net = self.policy.concept_net
        optimizer = self.policy.optimizer_concept_and_features
        actor_losses, critic_losses, ent_losses = [], [], []

        for _ in range(self.concept_ac_epochs):
            h_t = torch.zeros(N, concept_net.hidden_dim, device=self.device)
            step_actor: List[torch.Tensor] = []
            step_critic: List[torch.Tensor] = []
            step_ent: List[torch.Tensor] = []

            for t in range(T):
                if buf.obs_is_dict:
                    obs_t = {k: torch.as_tensor(buf.observations[k][t], dtype=torch.float32).to(self.device)
                             for k in buf.observations}
                else:
                    obs_t = torch.as_tensor(buf.observations[t], dtype=torch.float32).to(self.device)

                ep_start = torch.as_tensor(
                    buf.episode_starts[t], dtype=torch.float32, device=self.device
                ).unsqueeze(1)
                h_t = h_t * (1.0 - ep_start)

                features = self.policy.extract_features(obs_t)
                c_pred, h_t, concept_dists, V_c = concept_net(features, h_t)

                # Concept actor loss (PPO-clipped)
                c_adv = torch.as_tensor(buf.concept_advantages[t], dtype=torch.float32, device=self.device)
                if self.normalize_advantage and c_adv.numel() > 1:
                    c_adv = (c_adv - c_adv.mean()) / (c_adv.std() + 1e-8)

                old_clp = torch.as_tensor(buf.concept_log_probs[t], dtype=torch.float32, device=self.device)
                new_clp = concept_net.concept_log_probs(concept_dists, c_pred)
                ratio   = torch.exp(new_clp - old_clp)
                ac_l1   = c_adv * ratio
                ac_l2   = c_adv * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                step_actor.append(-torch.min(ac_l1, ac_l2).mean())

                # Concept critic loss (clipped)
                c_ret    = torch.as_tensor(buf.concept_returns[t], dtype=torch.float32, device=self.device)
                V_c_flat = V_c.flatten()
                V_c_old  = torch.as_tensor(buf.concept_values[t], dtype=torch.float32, device=self.device)
                V_clipped = V_c_old + torch.clamp(V_c_flat - V_c_old, -self.clip_range, self.clip_range)
                step_critic.append(torch.max(
                    F.mse_loss(V_c_flat, c_ret),
                    F.mse_loss(V_clipped, c_ret),
                ))

                # Concept entropy
                step_ent.append(
                    -torch.stack([d.entropy() for d in concept_dists], dim=1).mean()
                )

            total_loss = (
                torch.stack(step_actor).mean()
                + self.lambda_v  * torch.stack(step_critic).mean()
                + self.ent_coef  * torch.stack(step_ent).mean()
            )
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.features_extractor.parameters()) +
                list(concept_net.parameters()),
                self.max_grad_norm,
            )
            optimizer.step()

            actor_losses.append(torch.stack(step_actor).mean().item())
            critic_losses.append(torch.stack(step_critic).mean().item())
            ent_losses.append(torch.stack(step_ent).mean().item())

        return {
            "concept_actor_loss":  float(np.mean(actor_losses)),
            "concept_critic_loss": float(np.mean(critic_losses)),
            "concept_ent_loss":    float(np.mean(ent_losses)),
        }

    def _gather_concept_preds_from_buffer(self):
        """
        Run the concept network over the full rollout buffer (in order for GRU,
        chunked for non-GRU) and return (c_pred_np, c_true_np, eval_mask_np)
        already filtered to eval_mask > 0.

        Used by both accuracy and MSE logging so they share the same samples
        and the same GRU continuity treatment.
        """
        if not self.rollout_buffer.full:
            return None, None, None
        self.policy.set_training_mode(False)

        T = self.rollout_buffer.buffer_size
        N = self.n_envs

        with torch.no_grad():
            buf = self.rollout_buffer
            all_pred, all_true = [], []

            if self.temporal_encoding == "gru":
                # Sequential evaluation for any GRU method: carry h_t in
                # timestep order so the GRU sees proper temporal context.
                # Both vanilla_freeze+gru and concept_ac+gru need this —
                # chunked evaluation with h_prev=None breaks GRU continuity.
                h_t = torch.zeros(N, self.hidden_dim, device=self.device)
                for step in range(T):
                    if buf.obs_is_dict:
                        obs_t = {
                            k: torch.as_tensor(buf.observations[k][step],
                                               dtype=torch.float32).to(self.device)
                            for k in buf.observations
                        }
                    else:
                        obs_t = torch.as_tensor(
                            buf.observations[step], dtype=torch.float32
                        ).to(self.device)

                    ep_start = torch.as_tensor(
                        buf.episode_starts[step], dtype=torch.float32, device=self.device
                    ).unsqueeze(1)
                    h_t = h_t * (1.0 - ep_start)

                    features = self.policy.extract_features(obs_t)
                    if self.method == "vanilla_freeze":
                        c_pred, h_t = self.policy.concept_net(features, h_t)
                    else:
                        c_pred, h_t, _, _ = self.policy.concept_net(features, h_t)
                    all_pred.append(c_pred.cpu())
                    all_true.append(
                        torch.as_tensor(buf.concepts[step], dtype=torch.float32)
                    )
            else:
                # Non-GRU: process full buffer in order in chunks.
                T_buf, N_buf = buf.buffer_size, buf.n_envs
                chunk = 256
                for start in range(0, T_buf * N_buf, chunk):
                    end = min(start + chunk, T_buf * N_buf)
                    if buf.obs_is_dict:
                        obs_flat = {
                            k: torch.as_tensor(
                                buf.observations[k].reshape(T_buf * N_buf, *buf.obs_shape[k])[start:end],
                                dtype=torch.float32
                            ).to(self.device)
                            for k in buf.observations
                        }
                    else:
                        obs_flat = torch.as_tensor(
                            buf.observations.reshape(T_buf * N_buf, *buf.obs_shape)[start:end],
                            dtype=torch.float32
                        ).to(self.device)
                    c_true_chunk = torch.as_tensor(
                        buf.concepts.reshape(T_buf * N_buf, buf.concept_dim)[start:end],
                        dtype=torch.float32
                    )
                    features = self.policy.extract_features(obs_flat)
                    if self.method == "vanilla_freeze":
                        c_pred_chunk, _ = self.policy.concept_net(features, None)
                    else:
                        c_pred_chunk, _, _, _ = self.policy.concept_net(features, None)
                    all_pred.append(c_pred_chunk.cpu())
                    all_true.append(c_true_chunk)

            c_pred_np = torch.cat(all_pred, dim=0).numpy()   # [T*N, concept_dim]
            c_true_np = torch.cat(all_true, dim=0).numpy()

        self.policy.set_training_mode(True)

        # eval_mask: [T*N] — 1.0 only at steps where concept accuracy is
        # decision-relevant (e.g. at the junction in TMaze).  All-ones for
        # envs that don't expose concept_reward_active.
        eval_mask_np = self.rollout_buffer.concept_eval_mask.reshape(-1)
        keep = eval_mask_np > 0
        return c_pred_np[keep], c_true_np[keep], eval_mask_np[keep]

    def _compute_concept_accuracy_from_buffer(self) -> dict:
        """
        Per-concept metric over the rollout buffer, masked to decision-relevant
        steps (classification → accuracy ↑, regression → MSE ↓).
        """
        c_pred_np, c_true_np, mask = self._gather_concept_preds_from_buffer()
        if c_pred_np is None or len(c_pred_np) == 0:
            return {}
        metrics = {}
        for i, (name, tt) in enumerate(zip(self.concept_names, self.policy.task_types)):
            p = c_pred_np[:, i]
            t = c_true_np[:, i]
            if tt == "classification":
                # Append _accuracy to classification tasks
                metrics[f"{name}_acc"] = float(np.mean(np.round(p) == np.round(t)))
            else:
                # Append _mse to regression tasks
                metrics[f"{name}_mse"] = float(np.mean((p - t) ** 2))
        return metrics

    def _compute_concept_mse_from_buffer(self) -> dict:
        """
        Raw MSE per regression concept over the same buffer samples used for
        accuracy (sequential GRU eval, eval_mask applied).
        """
        c_pred_np, c_true_np, _ = self._gather_concept_preds_from_buffer()
        if c_pred_np is None or len(c_pred_np) == 0:
            return {}
        mse = {}
        for i, (name, tt) in enumerate(zip(self.concept_names, self.policy.task_types)):
            if tt == "regression":
                mse[name] = float(np.mean((c_pred_np[:, i] - c_true_np[:, i]) ** 2))
        return mse

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
            self.n_envs, self.hidden_dim, device=self.device
        )
        collected = 0

        self.policy.set_training_mode(False)
        while collected < n_samples:
            # Get concept for CURRENT observation before stepping
            concepts = self._get_current_concepts()

            obs_t = _obs_to_tensor(obs, self.device)
            with torch.no_grad():
                if self.method in ("concept_actor_critic", "gvf"):
                    actions, h_new = self.policy.predict(obs_t, h_t)
                    if h_new is not None:
                        h_t = h_new
                else:
                    actions, _ = self.policy.predict(obs_t)

            next_obs, _, terminated, truncated, infos = self.env.step(actions.cpu().numpy())
            dones = np.logical_or(terminated, truncated)

            for i in range(self.n_envs):
                if collected >= n_samples:
                    break
                if isinstance(obs, dict):
                    obs_list.append({k: obs[k][i] for k in obs})
                else:
                    obs_list.append(obs[i])
                concept_list.append(concepts[i])
                collected += 1
                if dones[i] and h_t is not None:
                    h_t[i] = 0.0

            obs = next_obs

        # Stack observations
        if isinstance(obs_list[0], dict):
            labeled_obs = {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
        else:
            labeled_obs = np.stack(obs_list)

        labeled_concepts = np.stack(concept_list)
        return labeled_obs, labeled_concepts

    def _get_current_concepts(self) -> np.ndarray:
        """
        Get ground-truth concepts for the CURRENT observation from each sub-env.
        Must be called BEFORE env.step() to get concepts matching the current obs.
        Uses get_attr for SubprocVectorEnv compatibility.
        """
        concept_dim = self.rollout_buffer.concept_dim
        try:
            raw = self.env.get_attr("current_concept")
            return np.stack([np.asarray(c, dtype=np.float32) for c in raw])
        except Exception:
            return np.zeros((self.n_envs, concept_dim), dtype=np.float32)

    def _compute_concept_reward(
        self, c_pred: np.ndarray, c_truth: np.ndarray
    ) -> np.ndarray:
        """
        Compute per-step concept accuracy reward r_c in [0, 1].

        For regression: errors are normalised by the running std of the concept
        values seen in the current rollout buffer, so r_c is informative
        regardless of the raw magnitude of the concept (e.g. tiny velocities
        in MountainCar would otherwise give exp(-MSE) ≈ 1 always, killing the
        gradient signal to the concept actor-critic).

        r_c = exp(-(normalised_error)^2), where
          normalised_error = (c_pred - c_truth) / (std(c_truth in buffer) + eps)
        """
        task_types = self.policy.task_types
        rewards = np.zeros(c_pred.shape[0], dtype=np.float32)

        # Use buffer concepts for std estimation when available
        if self.rollout_buffer.full:
            # concepts stored as [T, N, concept_dim] — flatten to [T*N, concept_dim]
            buf_concepts = self.rollout_buffer.concepts.reshape(-1, len(task_types))
        else:
            buf_concepts = None

        for i, tt in enumerate(task_types):
            if tt == "classification":
                rewards += (np.round(c_pred[:, i]) == np.round(c_truth[:, i])).astype(np.float32)
            else:
                error = c_pred[:, i] - c_truth[:, i]
                if buf_concepts is not None:
                    scale = float(buf_concepts[:, i].std()) + 1e-6
                else:
                    scale = float(np.abs(c_truth[:, i]).mean()) + 1e-6
                normalised_error = error / scale
                rewards += np.exp(-(normalised_error ** 2))
        rewards /= len(task_types)
        return rewards

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
        obs, _ = self.env.reset()
        h_t = torch.zeros(
            self.n_envs, self.hidden_dim, device=self.device
        )
        ep_rewards = [0.0] * self.n_envs
        done_count = 0
        self.policy.set_training_mode(False)

        while done_count < n_episodes:
            obs_t = _obs_to_tensor(obs, self.device)
            with torch.no_grad():
                if self.method in ("concept_actor_critic", "gvf"):
                    actions, h_new = self.policy.predict(obs_t, h_t, deterministic)
                    if h_new is not None:
                        h_t = h_new
                else:
                    actions, _ = self.policy.predict(obs_t, deterministic=deterministic)

            obs, r, terminated, truncated, _ = self.env.step(actions.cpu().numpy())
            dones = np.logical_or(terminated, truncated)
            for i in range(self.n_envs):
                ep_rewards[i] += r[i]
                if dones[i]:
                    rewards.append(ep_rewards[i])
                    ep_rewards[i] = 0.0
                    done_count += 1
                    if h_t is not None:
                        h_t[i] = 0.0

        return float(np.mean(rewards)), float(np.std(rewards))
