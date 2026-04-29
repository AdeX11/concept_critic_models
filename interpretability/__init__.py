"""
interpretability — Tools for analysing concept bottleneck actor-critic models.

Modules
-------
intervention   : Override concept values and run counterfactual replays.
ablation       : Measure per-concept importance via zero-ablation.
attribution    : Decompose action logits into per-concept contributions.
causal_graph   : Discover inter-concept causal dependencies across time.
visualization  : Plotting utilities for all analysis results.
"""

from .intervention import (
    intervene_on_concept,
    intervene_on_concepts,
    run_with_intervention,
    counterfactual_replay,
)
from .ablation import (
    ablate_and_evaluate,
    ablation_sweep,
    compute_importance_scores,
)
from .attribution import (
    attribute_action_logits,
    concept_sensitivity_matrix,
)
from .causal_graph import (
    discover_concept_dependencies,
    build_causal_adjacency_matrix,
)
from .visualization import (
    plot_ablation_heatmap,
    plot_attribution_bars,
    plot_causal_graph,
)

__all__ = [
    # intervention
    "intervene_on_concept",
    "intervene_on_concepts",
    "run_with_intervention",
    "counterfactual_replay",
    # ablation
    "ablate_and_evaluate",
    "ablation_sweep",
    "compute_importance_scores",
    # attribution
    "attribute_action_logits",
    "concept_sensitivity_matrix",
    # causal graph
    "discover_concept_dependencies",
    "build_causal_adjacency_matrix",
    # visualization
    "plot_ablation_heatmap",
    "plot_attribution_bars",
    "plot_causal_graph",
]