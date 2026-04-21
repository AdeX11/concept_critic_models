"""
base.py - Lightweight shared interfaces for concept environments.
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SupportsConceptEnv(Protocol):
    task_types: List[str]
    num_classes: List[int]
    concept_names: List[str]
    temporal_concepts: List[int]

    def get_concept(self) -> np.ndarray:
        ...


class ConceptMetadataMixin:
    """
    Shared helpers for environments exposing concept annotations.
    """

    current_concept: np.ndarray
    task_types: List[str]
    num_classes: List[int]
    concept_names: List[str]
    temporal_concepts: List[int]

    def get_concept(self) -> np.ndarray:
        return self.current_concept.copy()

    def _validate_concept_spec(self) -> None:
        n = len(self.task_types)
        if len(self.num_classes) != n or len(self.concept_names) != n:
            raise ValueError(
                "task_types, num_classes, and concept_names must have equal length"
            )
        if any(idx < 0 or idx >= n for idx in self.temporal_concepts):
            raise ValueError("temporal_concepts contains an out-of-range index")
