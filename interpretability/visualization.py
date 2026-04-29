"""
visualization.py — Plotting utilities for concept bottleneck interpretability results.

Produces publication-ready matplotlib/seaborn figures from the data structures
returned by ablation.py, attribution.py, and causal_graph.py.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    import os
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Ablation visualisation
# ---------------------------------------------------------------------------

def plot_ablation_heatmap(
    sweep_result: Dict,
    save_path: Optional[str] = None,
    *,
    figsize: Tuple[float, float] = (10, 4),
    title: str = "Concept Ablation Impact",
) -> plt.Figure:
    """
    Horizontal bar chart showing per-concept importance.

    Parameters
    ----------
    sweep_result : dict from ablation_sweep()
    save_path : optional file path to save the figure
    figsize, title : display options

    Returns
    -------
    matplotlib Figure
    """
    from .ablation import compute_importance_scores

    scores = compute_importance_scores(sweep_result)
    importance_list = scores["importance_scores"]
    names = [e["name"] for e in importance_list]
    values = [e["score"] for e in importance_list]
    ablated_means = [e["ablated_mean"] for e in importance_list]
    baseline = scores["baseline_mean"]

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["#e74c3c" if v > 0.3 else "#f39c12" if v > 0.1 else "#95a5a6" for v in values]
    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.axvline(x=0, color="white", linewidth=0.5)

    # Annotate with ablated mean
    for bar, am in zip(bars, ablated_means):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"μ={am:.1f}",
            va="center",
            fontsize=9,
            color="#ecf0f1",
        )

    ax.set_xlabel("Importance score (0 = no impact, 1 = maximal)")
    ax.set_title(f"{title}\nbaseline return = {baseline:.1f}")
    ax.set_xlim(-0.05, 1.15)
    ax.invert_yaxis()

    # Legend
    red_patch = mpatches.Patch(color="#e74c3c", label="High impact (>0.3)")
    yellow_patch = mpatches.Patch(color="#f39c12", label="Medium impact (>0.1)")
    grey_patch = mpatches.Patch(color="#95a5a6", label="Low impact")
    ax.legend(handles=[red_patch, yellow_patch, grey_patch], loc="lower right")

    fig.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[visualization] saved → {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Attribution visualisation
# ---------------------------------------------------------------------------

def plot_attribution_bars(
    attribution_result: Dict,
    save_path: Optional[str] = None,
    *,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Concept → Action Logit Sensitivity",
) -> plt.Figure:
    """
    Grouped bar chart: for each action, show ∂logit/∂concept for each concept.

    Parameters
    ----------
    attribution_result : dict from attribute_action_logits() or concept_sensitivity_matrix()
    save_path : optional file path
    figsize : auto-computed if None
    title : plot title

    Returns
    -------
    matplotlib Figure
    """
    # Accept both single-step and multi-step results
    if "sensitivity_matrix" in attribution_result:
        sens = attribution_result["sensitivity_matrix"]
    elif "mean_sensitivity" in attribution_result:
        sens = attribution_result["mean_sensitivity"]
    else:
        raise ValueError("attribution_result must contain 'sensitivity_matrix' or 'mean_sensitivity'")

    concept_names = attribution_result.get("concept_names", [f"c{i}" for i in range(sens.shape[1])])
    action_names = attribution_result.get("action_names", [f"A{i}" for i in range(sens.shape[0])])

    n_actions, n_concepts = sens.shape
    if figsize is None:
        figsize = (max(8, n_concepts * 1.2), max(4, n_actions * 1.0))

    x = np.arange(n_concepts)
    width = 0.8 / n_actions

    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.cm.tab10
    for i in range(n_actions):
        offset = (i - n_actions / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            sens[i],
            width,
            label=action_names[i],
            color=cmap(i % 10),
            edgecolor="white",
            linewidth=0.5,
        )
        # Annotate bars with value if significant
        for bar, val in zip(bars, sens[i]):
            if abs(val) > 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02 * np.sign(val) if val > 0 else bar.get_height() - 0.08,
                    f"{val:.2f}",
                    ha="center",
                    fontsize=7,
                    color="#ecf0f1",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(concept_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("∂ action_logit / ∂ concept")
    ax.set_title(title)
    ax.axhline(y=0, color="white", linewidth=0.5)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[visualization] saved → {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Causal graph visualisation
# ---------------------------------------------------------------------------

def plot_causal_graph(
    causal_result: Dict,
    save_path: Optional[str] = None,
    *,
    figsize: Tuple[float, float] = (8, 7),
    title: str = "Inter-Concept Causal Dependencies",
    cmap: str = "YlOrRd",
) -> plt.Figure:
    """
    Heatmap of the K×K causal adjacency matrix.

    Parameters
    ----------
    causal_result : dict from discover_concept_dependencies()
    save_path : optional file path
    figsize, title, cmap : display options

    Returns
    -------
    matplotlib Figure
    """
    adjacency = causal_result["adjacency_matrix"]
    concept_names = causal_result.get("concept_names", [f"c{i}" for i in range(adjacency.shape[0])])

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(adjacency, aspect="auto", cmap=cmap, vmin=0)

    # Annotate cells
    for i in range(adjacency.shape[0]):
        for j in range(adjacency.shape[1]):
            val = adjacency[i, j]
            text_color = "white" if val > adjacency.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=text_color)

    ax.set_xticks(range(len(concept_names)))
    ax.set_yticks(range(len(concept_names)))
    ax.set_xticklabels(concept_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(concept_names, fontsize=9)

    ax.set_xlabel("Target concept (affected)")
    ax.set_ylabel("Source concept (intervened on)")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean |Δ target| when source is set to 0", fontsize=9)

    fig.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[visualization] saved → {save_path}")

    return fig