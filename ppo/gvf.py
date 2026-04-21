"""
gvf.py — Concept bottleneck with optional general value function heads.

ConceptNetwork matches FlexibleMultiTaskNetwork for concept prediction
(classification / regression heads, logits, and supervised losses), and adds
num_gvf linear heads that predict scalar general value functions from the same
features. forward() returns concept predictions concatenated with GVF scalars
along the feature dimension: [B, n_concepts + num_gvf].
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GVFConceptNetwork(nn.Module):
    """
    Like FlexibleMultiTaskNetwork, plus num_gvf scalar value heads (GVFs).
    Supports temporal encoding matching ConceptActorCritic:
      - 'gru': heads consume recurrent hidden state
      - 'stacked'/'none': heads consume input features directly

    For each concept:
      - classification: nn.Linear(feature_dim, K) → argmax → float class index
      - regression:     nn.Linear(feature_dim, 1)  → scalar

    Each GVF: nn.Linear(feature_dim, 1) → squeeze → scalar; forward stacks these
    after concept outputs: [B, n_concepts + num_gvf].

    get_logits() returns raw outputs for concept heads only (same as
    FlexibleMultiTaskNetwork). Use get_gvf_logits() for raw GVF outputs.
    """

    HIDDEN_DIM = 256

    def __init__(
        self,
        feature_dim: int,
        task_types: List[str],
        num_classes: List[int],
        gvf_pairing: List[int],
        temporal_encoding: str = "gru",
    ):
        super().__init__()
        assert temporal_encoding in ("gru", "stacked", "none"), (
            f"temporal_encoding must be 'gru', 'stacked', or 'none', got '{temporal_encoding}'"
        )
        self.task_types = task_types
        self.num_classes = num_classes
        self.n_concepts = len(task_types)
        self.num_gvf = len(gvf_pairing)
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

        self.heads = nn.ModuleList()
        for task_type, n_cls in zip(task_types, num_classes):
            if task_type == "classification":
                assert n_cls is not None and n_cls > 1, (
                    "num_classes must be > 1 for classification"
                )
                self.heads.append(nn.Linear(head_input_dim, n_cls))
            elif task_type == "regression":
                self.heads.append(nn.Linear(head_input_dim, 1))
            else:
                raise ValueError(f"Unknown task_type '{task_type}'")

        self.gvf_heads = nn.ModuleList(
            [nn.Linear(head_input_dim, 1) for _ in range(self.num_gvf)]
        )

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

    def forward(
        self,
        x: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: [B, feature_dim]
        h_prev: [B, hidden_dim] or None (for 'gru')
        returns:
          pred: [B, n_concepts + num_gvf]
          first n_concepts columns: argmax (cls) or scalar (reg)
          last num_gvf columns: GVF scalar predictions
          h_t: [B, hidden_dim] for 'gru', else None
        """
        head_input, h_t = self._get_head_input(x, h_prev)
        outputs: List[torch.Tensor] = []
        for head, task_type in zip(self.heads, self.task_types):
            out = head(head_input)
            if task_type == "classification":
                out = out.argmax(dim=1).float()
            else:
                out = out.squeeze(dim=1)
            outputs.append(out)
        concept_part = torch.stack(outputs, dim=1)

        if self.num_gvf == 0:
            return concept_part, h_t

        gvf_scalars = [head(head_input).squeeze(dim=1) for head in self.gvf_heads]
        gvf_part = torch.stack(gvf_scalars, dim=1)
        return torch.cat([concept_part, gvf_part], dim=1), h_t

    def get_logits(
        self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """Returns raw logits/scalars for concept heads and new hidden state."""
        head_input, h_t = self._get_head_input(x, h_prev)
        return [head(head_input) for head in self.heads], h_t

    def get_gvf_logits(
        self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """Returns raw GVF outputs and new hidden state."""
        head_input, h_t = self._get_head_input(x, h_prev)
        return [head(head_input) for head in self.gvf_heads], h_t

    def compute_concept_loss(
        self, logits: List[torch.Tensor], ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """
        logits: list of [B, K] or [B, 1] for concepts only
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

    def _concept_preds(self, preds: torch.Tensor) -> torch.Tensor:
        """Use first n_concept columns if preds include GVF outputs from forward()."""
        if preds.size(1) > self.n_concepts:
            return preds[:, : self.n_concepts]
        return preds

    def compute_metric(
        self, preds: torch.Tensor, ground_truth: torch.Tensor
    ) -> float:
        """preds may be [B, n_concepts] or full forward() [B, n_concepts + num_gvf]."""
        metrics = self.compute_all_metrics(preds, ground_truth)
        return sum(metrics) / len(metrics)

    def compute_all_metrics(
        self, preds: torch.Tensor, ground_truth: torch.Tensor
    ) -> List[float]:
        """Returns per-concept metric list."""
        preds = self._concept_preds(preds)
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
