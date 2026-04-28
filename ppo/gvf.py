"""
gvf.py - Concept bottleneck module with general value function heads.

GVFConceptNetwork mirrors FlexibleMultiTaskNetwork for concept prediction and
adds scalar GVF heads that learn discounted cumulants derived from selected
ground-truth concepts. The policy receives predicted concepts concatenated with
GVF scalars: [B, n_concepts + num_gvf].
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GVFConceptNetwork(nn.Module):
    """
    Concept bottleneck with optional recurrent state and GVF scalar heads.

    For concepts:
      - classification: Linear(head_input, K) -> argmax class index
      - regression: Linear(head_input, 1) -> scalar

    For each GVF:
      - Linear(head_input, 1) -> scalar discounted-cumulant prediction
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
        self.gvf_pairing = list(gvf_pairing)
        self.num_gvf = len(gvf_pairing)
        self.temporal_encoding = temporal_encoding

        if temporal_encoding == "gru":
            self.hidden_dim = self.HIDDEN_DIM
            self.gru = nn.GRUCell(feature_dim, self.hidden_dim)
            head_input_dim = self.hidden_dim
        else:
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
        if self.temporal_encoding == "gru":
            batch_size = features.size(0)
            if h_prev is None:
                h_prev = torch.zeros(batch_size, self.hidden_dim, device=features.device)
            h_t = self.gru(features, h_prev)
            return h_t, h_t
        return features, None

    def forward(
        self,
        x: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Return (predictions, h_new).

        predictions contains concept predictions followed by GVF predictions.
        """
        head_input, h_t = self._get_head_input(x, h_prev)
        concept_outputs: List[torch.Tensor] = []
        for head, task_type in zip(self.heads, self.task_types):
            out = head(head_input)
            if task_type == "classification":
                out = out.argmax(dim=1).float()
            else:
                out = out.squeeze(dim=1)
            concept_outputs.append(out)
        concept_part = torch.stack(concept_outputs, dim=1)

        if self.num_gvf == 0:
            return concept_part, h_t

        gvf_scalars = [head(head_input).squeeze(dim=1) for head in self.gvf_heads]
        gvf_part = torch.stack(gvf_scalars, dim=1)
        return torch.cat([concept_part, gvf_part], dim=1), h_t

    def get_logits(
        self,
        x: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """Return raw concept logits/scalars and recurrent state."""
        head_input, h_t = self._get_head_input(x, h_prev)
        return [head(head_input) for head in self.heads], h_t

    def get_gvf_logits(
        self,
        x: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """Return raw GVF scalar predictions and recurrent state."""
        head_input, h_t = self._get_head_input(x, h_prev)
        return [head(head_input) for head in self.gvf_heads], h_t

    def compute_concept_loss(
        self,
        logits: List[torch.Tensor],
        ground_truth: torch.Tensor,
    ) -> torch.Tensor:
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
        if preds.size(1) > self.n_concepts:
            return preds[:, : self.n_concepts]
        return preds

    def compute_metric(self, preds: torch.Tensor, ground_truth: torch.Tensor) -> float:
        metrics = self.compute_all_metrics(preds, ground_truth)
        return sum(metrics) / len(metrics)

    def compute_all_metrics(
        self,
        preds: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> List[float]:
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
