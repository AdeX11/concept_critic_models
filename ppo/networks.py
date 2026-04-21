"""
networks.py — Concept bottleneck network architectures.

Two concept modules:
  1. FlexibleMultiTaskNetwork  — LICORICE-style single-step CBM (no temporal state)
  2. ConceptActorCritic        — concept actor + concept critic (new method)

ConceptActorCritic supports three temporal encodings (temporal_encoding arg):
  'gru'     — GRUCell carries hidden state across steps (network-level temporal)
  'stacked' — no GRU; temporal info comes from frame-stacked observations (env-level)
  'none'    — no temporal encoding at all (ablation baseline)

For 'stacked' and 'none', ConceptActorCritic uses plain linear heads identical in
architecture to FlexibleMultiTaskNetwork, but is trained with the actor-critic
discounted concept reward signal rather than single-step supervised loss.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


# ---------------------------------------------------------------------------
# FlexibleMultiTaskNetwork
# ---------------------------------------------------------------------------

class FlexibleMultiTaskNetwork(nn.Module):
    """
    Baseline concept bottleneck module used in vanilla_freeze.

    For each concept:
      - classification: nn.Linear(feature_dim, K)  → argmax → integer class
      - regression:     nn.Linear(feature_dim, 1)  → scalar

    forward() returns [B, n_concepts] (float tensor of argmax / scalar predictions).
    get_logits() returns list of raw output tensors (for loss computation).
    """

    def __init__(
        self,
        feature_dim: int,
        task_types: List[str],
        num_classes: List[int],
    ):
        super().__init__()
        self.task_types = task_types
        self.num_classes = num_classes
        self.n_concepts = len(task_types)

        self.heads = nn.ModuleList()
        for task_type, n_cls in zip(task_types, num_classes):
            if task_type == "classification":
                assert n_cls is not None and n_cls > 1, (
                    "num_classes must be > 1 for classification"
                )
                self.heads.append(nn.Linear(feature_dim, n_cls))
            elif task_type == "regression":
                self.heads.append(nn.Linear(feature_dim, 1))
            else:
                raise ValueError(f"Unknown task_type '{task_type}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, feature_dim]
        returns: [B, n_concepts]  (argmax for classification, scalar for regression)
        """
        outputs = []
        for head, task_type in zip(self.heads, self.task_types):
            out = head(x)
            if task_type == "classification":
                out = out.argmax(dim=1).float()
            else:
                out = out.squeeze(dim=1)
            outputs.append(out)
        return torch.stack(outputs, dim=1)  # [B, n_concepts]

    def get_logits(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns raw logits / scalars — one tensor per concept."""
        return [head(x) for head in self.heads]

    def compute_loss(
        self, logits: List[torch.Tensor], ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """
        logits: list of [B, K] or [B, 1]
        ground_truth: [B, n_concepts]
        """
        losses = []
        for idx, (logit, task_type) in enumerate(zip(logits, self.task_types)):
            truth = ground_truth[:, idx]
            if task_type == "classification":
                loss = F.cross_entropy(logit, truth.long())
            else:
                loss = F.mse_loss(logit.squeeze(-1), truth.float())
            losses.append(loss)
        return sum(losses) / len(losses)

    def compute_metric(
        self, preds: torch.Tensor, ground_truth: torch.Tensor
    ) -> float:
        """
        preds, ground_truth: [B, n_concepts]
        Returns mean metric over all concepts (accuracy for cls, MSE for reg).
        """
        metrics = self.compute_all_metrics(preds, ground_truth)
        return sum(metrics) / len(metrics)

    def compute_all_metrics(
        self, preds: torch.Tensor, ground_truth: torch.Tensor
    ) -> List[float]:
        """Returns per-concept metric list."""
        metrics = []
        for idx, task_type in enumerate(self.task_types):
            pred = preds[:, idx]
            truth = ground_truth[:, idx]
            if task_type == "classification":
                metric = (pred.round().long() == truth.long()).float().mean().item()
            else:
                metric = F.mse_loss(pred.float(), truth.float()).item()
            metrics.append(metric)
        return metrics


# ---------------------------------------------------------------------------
# ConceptActorCritic
# ---------------------------------------------------------------------------

class ConceptActorCritic(nn.Module):
    """
    Concept actor + concept critic (new method).

    temporal_encoding controls how temporal information enters:
      'gru'     — GRUCell(feature_dim, hidden_dim=256); heads operate on h_t
      'stacked' — no GRU; temporal info comes from frame-stacked observations (env-level)
      'none'    — no GRU; heads operate directly on features [B, feature_dim]
                  no temporal information (ablation)

    All three variants are trained with the actor-critic discounted concept reward
    signal, distinguishing them from FlexibleMultiTaskNetwork regardless of
    temporal encoding.

    Action-conditional concept value estimation:
      The concept critic is Q_c(h_t, a_t) rather than V_c(h_t).
      critic_head takes concat(head_input, a_t_one_hot) so it can answer
      "given state history h_t and the action I'm about to take, what is the
      expected discounted concept accuracy?"

      Scope limitation: this makes the VALUE FUNCTION action-conditional, not the
      concept PREDICTIONS themselves.  The concept actor c_t = f(h_t) is still a
      function of observation history only and does not predict what would happen
      under a candidate action.  Making the concept actor truly action-conditional
      (e.g. "if I go right, will I collide?") would require conditioning actor
      heads on the current action, which creates a circular dependency with the
      task policy π(a|c_t).  That is a deeper architectural change outside this
      fix's scope.

      Consequence: V_c cannot be computed until after action sampling (in
      collect_rollouts). forward() returns V_c=None; callers obtain Q_c via
      compute_concept_value_from_head(head_input, action_one_hot) after sampling.
      train_concept_actor_critic and evaluate_actions pass stored actions directly
      to forward(actions=...) so Q_c is computed in one shot there.

    Fix #2 — One-hot policy representation (no false ordinality):
      Previously, classification concepts were squashed to a float class-index
      scalar via the STE (0, 1, 2, ..., K-1).  This imposes a false ordinal
      structure: the policy MLP sees "left"(3) as numerically closer to "down"(2)
      than to "right"(1) for obstacle_move_direction, which has no ordinal
      meaning.  The fix keeps the full K-dimensional one-hot STE vector for each
      classification concept and concatenates them.  The policy MLP input grows
      from n_concepts scalars to policy_dim = sum(K_i) floats.  Regression
      concepts remain as scalars.

      concept_slices tracks (start, end) offsets in the policy_dim vector per
      concept, used by decode_concept_vector and compute_all_metrics to decode
      back to integer class predictions without rerunning the network.
      concept_log_probs does NOT use concept_slices — it iterates by concept
      index over the stored [B, n_concepts] concept_actions directly.

    forward(features, h_prev, actions=None) → (c_t, h_t, concept_dists, V_c)
      c_t:           [B, policy_dim]  — one-hot cat for classification, scalar for regression
      h_t:           [B, hidden_dim] for 'gru', else None
      concept_dists: list[Distribution]
      V_c:           [B, 1] if actions provided, else None
    """

    HIDDEN_DIM = 256

    def __init__(
        self,
        feature_dim: int,
        task_types: List[str],
        num_classes: List[int],
        n_actions: int = 7,
        temporal_encoding: str = "gru",
    ):
        super().__init__()
        assert temporal_encoding in ("gru", "stacked", "none"), (
            f"temporal_encoding must be 'gru', 'stacked', or 'none', got '{temporal_encoding}'"
        )
        self.task_types = task_types
        self.num_classes = num_classes
        self.n_concepts = len(task_types)
        self.n_actions = n_actions
        self.temporal_encoding = temporal_encoding

        if temporal_encoding == "gru":
            self.hidden_dim = self.HIDDEN_DIM
            self.gru = nn.GRUCell(feature_dim, self.hidden_dim)
            head_input_dim = self.hidden_dim
        else:
            # 'stacked' or 'none': heads read CNN features directly, no GRU
            self.hidden_dim = feature_dim
            self.gru = None
            head_input_dim = feature_dim

        # Fix #2: compute per-concept slice offsets in the policy vector.
        # Each classification concept occupies K_i dimensions (one-hot),
        # each regression concept occupies 1 dimension (scalar).
        # policy_dim replaces the old concept_dim as the MLP input width.
        concept_slices: List[Tuple[int, int]] = []
        offset = 0
        for task_type, n_cls in zip(task_types, num_classes):
            if task_type == "classification":
                concept_slices.append((offset, offset + n_cls))
                offset += n_cls
            else:
                concept_slices.append((offset, offset + 1))
                offset += 1
        # Register as buffer so it moves with the module (e.g. .to(device)) but
        # is not a learnable parameter.
        self.register_buffer(
            "_concept_slices",
            torch.tensor(concept_slices, dtype=torch.long),
        )
        self.concept_slices: List[Tuple[int, int]] = concept_slices
        self.policy_dim: int = offset  # total policy MLP input width

        # Actor heads — one per concept (unchanged from before)
        self.actor_heads = nn.ModuleList()
        for task_type, n_cls in zip(task_types, num_classes):
            if task_type == "classification":
                self.actor_heads.append(nn.Linear(head_input_dim, n_cls))
            else:  # regression — mu + log_std
                head = nn.ModuleDict({
                    "mu":      nn.Linear(head_input_dim, 1),
                    "log_std": nn.Linear(head_input_dim, 1),
                })
                self.actor_heads.append(head)

        # Fix #1: action-conditional concept critic Q_c(h_t, a_t).
        # Input is concat(head_input, a_t_one_hot), so width = head_input_dim + n_actions.
        # The old passive V_c(h_t) = Linear(head_input_dim, 1) ignored the action,
        # preventing the critic from expressing "given I take action a, what is the
        # expected concept accuracy?"  Increasing the input by n_actions fixes this
        # without changing the number of layers or other hyperparameters.
        self.critic_head = nn.Linear(head_input_dim + n_actions, 1)

    # ------------------------------------------------------------------

    def _get_head_input(
        self,
        features: torch.Tensor,
        h_prev: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns (head_input, h_new).
        For 'gru': runs GRUCell, returns (h_t, h_t).
        For 'stacked'/'none': returns (features, None).
        """
        if self.temporal_encoding == "gru":
            B = features.size(0)
            if h_prev is None:
                h_prev = torch.zeros(B, self.hidden_dim, device=features.device)
            h_t = self.gru(features, h_prev)
            return h_t, h_t
        else:
            return features, None

    # ------------------------------------------------------------------

    def compute_concept_value_from_head(
        self,
        head_input: torch.Tensor,
        action_one_hot: torch.Tensor,
    ) -> torch.Tensor:
        """
        Q_c(head_input, a_t) — action-conditional concept value.

        Called separately from forward() during rollout, after the action has been
        sampled, so that V_c is not computed before we know what action was taken.
        head_input is h_t (GRU output) for temporal_encoding='gru', or CNN features
        for 'stacked'/'none'.  action_one_hot is [B, n_actions] float.

        Relationship to GAE: concept_values stored in the buffer are Q_c values,
        and compute_concept_returns_and_advantage in buffer.py treats them as
        standard V values in the Bellman backup.  This is valid because for each
        stored (s_t, a_t) pair, Q_c(s_t, a_t) is the correct bootstrap target
        under the current policy.
        """
        return self.critic_head(torch.cat([head_input, action_one_hot], dim=-1))

    def forward(
        self,
        features: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], list, Optional[torch.Tensor]]:
        """
        features: [B, feature_dim]
        h_prev:   [B, hidden_dim] or None  (only used for 'gru')
        actions:  [B] long tensor of action indices, or None.
                  When provided, computes Q_c(h_t, a_t) and returns it as V_c.
                  When None (rollout, before action sampling), V_c = None and the
                  caller must invoke compute_concept_value_from_head after sampling.

        returns:
          c_t:           [B, policy_dim]  — Fix #2: one-hot STE per classification concept
                                            (not scalar class indices).  No false ordinality.
          h_t:           [B, hidden_dim] for 'gru', else None
          concept_dists: list[Distribution]
          V_c:           [B, 1] if actions provided, else None  — Fix #1
        """
        head_input, h_t = self._get_head_input(features, h_prev)

        # Fix #1: compute Q_c only when we have the action.
        # During rollout, actions=None because the policy has not yet sampled.
        # During training (train_concept_actor_critic, evaluate_actions), the stored
        # action is passed in so Q_c is computed in one forward pass.
        if actions is not None:
            action_one_hot = F.one_hot(
                actions.long().flatten(), self.n_actions
            ).float().to(features.device)           # [B, n_actions]
            V_c = self.critic_head(torch.cat([head_input, action_one_hot], dim=-1))
        else:
            V_c = None

        c_t_list = []
        concept_dists = []

        for head, task_type, n_cls in zip(self.actor_heads, self.task_types, self.num_classes):
            if task_type == "classification":
                logits = head(head_input)                           # [B, K]
                dist = Categorical(logits=logits)

                # Fix #2: Straight-through estimator — one-hot forward, softmax backward.
                # Previously this collapsed to a scalar class index via dot product with
                # [0,1,...,K-1], imposing a false ordinal structure on non-ordinal
                # categories (e.g. obstacle_move_direction: 0=stayed,1=right,...,4=up).
                # Now we keep the full K-dimensional one-hot vector, which is what the
                # policy MLP should receive.  The downstream MLP input dimension grows
                # from n_concepts to policy_dim = sum(K_i), tracked via concept_slices.
                hard_idx  = logits.argmax(dim=1)                    # [B]
                soft      = F.softmax(logits, dim=1)                # [B, K]
                one_hot_h = F.one_hot(hard_idx, n_cls).float()      # [B, K]
                # STE: hard one-hot in the forward pass, soft gradient in the backward pass.
                # This preserves the categorical semantics while allowing gradient flow
                # through the softmax, following Bengio et al. (2013).
                c = (one_hot_h - soft).detach() + soft              # [B, K]
                c_t_list.append(c)
                concept_dists.append(dist)
            else:
                mu      = head["mu"](head_input).squeeze(-1)        # [B]
                log_std = head["log_std"](head_input).squeeze(-1)   # [B]
                log_std = torch.clamp(log_std, -4.0, 2.0)
                dist = Normal(mu, log_std.exp())
                # Regression concepts remain scalars — ordinal structure is valid here.
                c_t_list.append(mu.unsqueeze(-1))                   # [B, 1]
                concept_dists.append(dist)

        # Concatenate all concept representations into the policy vector.
        # Shape: [B, policy_dim].  policy_dim = sum(K_i for cls) + count(reg).
        # concept_slices[i] = (start, end) gives the slice for concept i.
        c_t = torch.cat(c_t_list, dim=1)
        return c_t, h_t, concept_dists, V_c

    # ------------------------------------------------------------------

    def decode_concept_vector(self, c_t: torch.Tensor) -> torch.Tensor:
        """
        Project the policy-format concept vector [B, policy_dim] back into
        ground-truth-comparable concept space [B, n_concepts].

        For classification: argmax of the one-hot slice → integer class (float).
        For regression: scalar at the slice start (unchanged).

        This is used ONLY for logging and evaluation callers
        (_compute_concept_accuracy_from_buffer, _compute_concept_mse_from_buffer,
        compare._evaluate_concepts).  It must NOT be used for _compute_concept_reward
        or PPO training — those use the stored sampled concept_actions instead,
        for reward/ratio consistency.
        """
        decoded = []
        for (start, end), task_type in zip(self.concept_slices, self.task_types):
            if task_type == "classification":
                decoded.append(c_t[:, start:end].argmax(dim=1).float())
            else:
                decoded.append(c_t[:, start])
        return torch.stack(decoded, dim=1)  # [B, n_concepts]

    # ------------------------------------------------------------------

    def get_logits(
        self,
        features: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """Returns raw logits (for supervised loss) and new hidden state."""
        head_input, h_t = self._get_head_input(features, h_prev)
        logits = []
        for head, task_type in zip(self.actor_heads, self.task_types):
            if task_type == "classification":
                logits.append(head(head_input))
            else:
                logits.append(head["mu"](head_input))
        return logits, h_t

    # ------------------------------------------------------------------

    def compute_concept_loss(
        self, logits: List[torch.Tensor], ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """Supervised anchor loss on labeled concept samples."""
        losses = []
        for idx, (logit, task_type) in enumerate(zip(logits, self.task_types)):
            truth = ground_truth[:, idx]
            if task_type == "classification":
                loss = F.cross_entropy(logit, truth.long())
            else:
                loss = F.mse_loss(logit.squeeze(-1), truth.float())
            losses.append(loss)
        return sum(losses) / len(losses)

    def compute_metric(
        self, preds: torch.Tensor, ground_truth: torch.Tensor
    ) -> float:
        metrics = self.compute_all_metrics(preds, ground_truth)
        return sum(metrics) / len(metrics)

    def compute_all_metrics(
        self, preds: torch.Tensor, ground_truth: torch.Tensor
    ) -> List[float]:
        """
        preds:        [B, policy_dim] one-hot format (Fix #2 output of forward())
                      OR [B, n_concepts] legacy scalar format.
        ground_truth: [B, n_concepts] integer class labels.

        For classification: decodes argmax from the concept's one-hot slice using
        concept_slices.  For regression: reads the single scalar at the slice start.
        This makes the metric computation consistent with the new policy representation
        without changing how ground-truth labels are stored in the buffer.
        """
        is_one_hot = (preds.shape[1] == self.policy_dim and self.policy_dim != self.n_concepts)
        metrics = []
        for idx, (task_type, (start, end)) in enumerate(
            zip(self.task_types, self.concept_slices)
        ):
            truth = ground_truth[:, idx]
            if task_type == "classification":
                if is_one_hot:
                    # Decode integer class from the one-hot slice for this concept.
                    pred = preds[:, start:end].argmax(dim=1).long()
                else:
                    # Legacy path: scalar float predictions (e.g. from FlexibleMultiTaskNetwork
                    # or older checkpoints).  Round to nearest integer class.
                    pred = preds[:, idx].round().long()
                metric = (pred == truth.long()).float().mean().item()
            else:
                pred = preds[:, start] if is_one_hot else preds[:, idx]
                metric = F.mse_loss(pred.float(), truth.float()).item()
            metrics.append(metric)
        return metrics

    def sample_concept_actions(
        self,
        concept_dists: list,
    ) -> torch.Tensor:
        """
        Sample one action per concept from the current concept distributions.

        Returns [B, n_concepts] float — one sampled value per concept.
        Classification entries are integer class indices (stored as float for
        buffer uniformity); regression entries are float samples from Normal.

        These samples are what the PPO importance ratio is computed over: both
        old_log_prob (at collection time) and new_log_prob (at training time) are
        evaluated at the SAME stored sample, ensuring the ratio measures policy
        change on a fixed action.
        """
        samples = []
        for dist, task_type in zip(concept_dists, self.task_types):
            if task_type == "classification":
                # Sample integer class index from Categorical distribution.
                samples.append(dist.sample().float())   # [B]
            else:
                # Sample from Normal(mu, sigma).
                samples.append(dist.sample())            # [B]
        return torch.stack(samples, dim=1)  # [B, n_concepts]

    def concept_log_probs(
        self,
        concept_dists: list,
        concept_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sum of log-probs evaluated at stored concept_actions.

        concept_actions: [B, n_concepts] — the sampled concept actions from the
        rollout, stored in the buffer.  Each column is one concept's action:
        integer class for classification, float value for regression.

        This method iterates by concept index (not concept_slices) because
        concept_actions is already in per-concept [B, n_concepts] layout.
        Classification entries must be cast to long for Categorical.log_prob().

        returns: [B] — summed log prob across all concepts.
        """
        log_p = []
        for i, (dist, task_type) in enumerate(zip(concept_dists, self.task_types)):
            c = concept_actions[:, i]
            if task_type == "classification":
                lp = dist.log_prob(c.long())
            else:
                lp = dist.log_prob(c)
            log_p.append(lp)
        return torch.stack(log_p, dim=1).sum(dim=1)  # [B]
