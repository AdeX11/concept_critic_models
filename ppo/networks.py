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
      'stacked' — no GRU; heads operate directly on features [B, feature_dim]
                  temporal info comes from frame-stacked observations (env-level)
      'none'    — no GRU; heads operate directly on features [B, feature_dim]
                  no temporal information (ablation)

    All three variants are trained with the actor-critic discounted concept reward
    signal, distinguishing them from FlexibleMultiTaskNetwork regardless of
    temporal encoding.

    forward(features, h_prev) → (c_t, h_t, concept_dists, V_c)
      c_t:           [B, n_concepts]
      h_t:           [B, hidden_dim] for 'gru', else None
      concept_dists: list[Distribution]
      V_c:           [B, 1]
    """

    HIDDEN_DIM = 256

    def __init__(
        self,
        feature_dim: int,
        task_types: List[str],
        num_classes: List[int],
        temporal_encoding: str = "gru",
    ):
        super().__init__()
        assert temporal_encoding in ("gru", "stacked", "none"), (
            f"temporal_encoding must be 'gru', 'stacked', or 'none', got '{temporal_encoding}'"
        )
        self.task_types = task_types
        self.num_classes = num_classes
        self.n_concepts = len(task_types)
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

        # Actor heads — one per concept
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

        # Concept critic head
        self.critic_head = nn.Linear(head_input_dim, 1)

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

    def forward(
        self,
        features: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], list, torch.Tensor]:
        """
        features: [B, feature_dim]
        h_prev:   [B, hidden_dim] or None  (only used for 'gru')
        returns:
          c_t:           [B, n_concepts]
          h_t:           [B, hidden_dim] for 'gru', else None
          concept_dists: list[Distribution]
          V_c:           [B, 1]
        """
        head_input, h_t = self._get_head_input(features, h_prev)
        V_c = self.critic_head(head_input)  # [B, 1]

        c_t_list = []
        concept_dists = []

        for head, task_type in zip(self.actor_heads, self.task_types):
            if task_type == "classification":
                logits = head(head_input)                      # [B, K]
                dist = Categorical(logits=logits)
                # Straight-through estimator: argmax forward, softmax backward
                hard = logits.argmax(dim=1).float()            # [B]
                soft = F.softmax(logits, dim=1)                # [B, K]
                k = logits.size(1)
                one_hot = F.one_hot(hard.long(), k).float()
                c = (one_hot - soft).detach() + soft
                c = (c * torch.arange(k, device=features.device).float()).sum(dim=1)
                c_t_list.append(c)
                concept_dists.append(dist)
            else:
                mu      = head["mu"](head_input).squeeze(-1)       # [B]
                log_std = head["log_std"](head_input).squeeze(-1)  # [B]
                log_std = torch.clamp(log_std, -4.0, 2.0)
                dist = Normal(mu, log_std.exp())
                c_t_list.append(mu)
                concept_dists.append(dist)

        c_t = torch.stack(c_t_list, dim=1)  # [B, n_concepts]
        return c_t, h_t, concept_dists, V_c

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

    def concept_log_probs(
        self,
        concept_dists: list,
        c_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sum of log-probs across concepts.
        c_t: [B, n_concepts]
        returns: [B]
        """
        log_p = []
        for idx, (dist, task_type) in enumerate(zip(concept_dists, self.task_types)):
            c = c_t[:, idx]
            if task_type == "classification":
                lp = dist.log_prob(c.long())
            else:
                lp = dist.log_prob(c)
            log_p.append(lp)
        return torch.stack(log_p, dim=1).sum(dim=1)  # [B]
