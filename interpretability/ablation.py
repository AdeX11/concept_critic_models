"""
ablation.py — Measure per-concept importance by zero-ablation.

For each concept, we zero it out (set to 0.0) and measure the drop in episodic
return compared to a baseline run with all concepts intact.  A large drop means
the policy depends heavily on that concept.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ppo.policy import ActorCriticPolicy
from .intervention import run_with_intervention


# ---------------------------------------------------------------------------
# Single-concept ablation
# ---------------------------------------------------------------------------

def ablate_and_evaluate(
    policy: ActorCriticPolicy,
    env_factory,
    concept_idx: int,
    *,
    n_episodes: int = 10,
    seed: int = 42,
    max_steps: int = 200,
    deterministic: bool = True,
    device: torch.device = torch.device("cpu"),
    temporal_encoding: str = "none",
) -> Tuple[float, float]:
    """
    Zero-ablate one concept and measure average episodic return.

    Parameters
    ----------
    policy : ActorCriticPolicy
    env_factory : callable returning a fresh env (e.g. lambda: make_single_env(...))
    concept_idx : int — which concept to ablate
    n_episodes : number of evaluation episodes
    seed : base seed (incremented per episode)
    max_steps, deterministic, device, temporal_encoding : standard rollout params

    Returns
    -------
    (mean_return, std_return) across n_episodes
    """
    returns: List[float] = []
    for ep in range(n_episodes):
        env = env_factory()
        result = run_with_intervention(
            policy,
            env,
            overrides={concept_idx: 0.0},
            seed=seed + ep,
            max_steps=max_steps,
            deterministic=deterministic,
            device=device,
            temporal_encoding=temporal_encoding,
        )
        returns.append(result["total_reward"])
    return float(np.mean(returns)), float(np.std(returns))


# ---------------------------------------------------------------------------
# Full ablation sweep across all concepts
# ---------------------------------------------------------------------------

def ablation_sweep(
    policy: ActorCriticPolicy,
    env_factory,
    *,
    n_episodes: int = 10,
    seed: int = 42,
    max_steps: int = 200,
    deterministic: bool = True,
    device: torch.device = torch.device("cpu"),
    temporal_encoding: str = "none",
    concept_names: Optional[List[str]] = None,
) -> Dict:
    """
    Run zero-ablation across every concept, plus a baseline with all concepts intact.

    Parameters
    ----------
    policy, env_factory, n_episodes, seed, max_steps, deterministic, device,
    temporal_encoding : same as ablate_and_evaluate
    concept_names : optional list of human-readable concept names

    Returns
    -------
    result : dict with keys:
        baseline_mean, baseline_std,
        per_concept : list of {name, mean, std, delta_mean}
    """
    n_concepts = policy.concept_dim

    # --- Baseline (no ablation) ---
    baseline_returns: List[float] = []
    for ep in range(n_episodes):
        env = env_factory()
        result = run_with_intervention(
            policy,
            env,
            overrides={},
            seed=seed + ep,
            max_steps=max_steps,
            deterministic=deterministic,
            device=device,
            temporal_encoding=temporal_encoding,
        )
        baseline_returns.append(result["total_reward"])

    baseline_mean = float(np.mean(baseline_returns))
    baseline_std = float(np.std(baseline_returns))

    # --- Per-concept ablation ---
    per_concept: List[Dict] = []
    for c_idx in range(n_concepts):
        mean_r, std_r = ablate_and_evaluate(
            policy, env_factory, c_idx,
            n_episodes=n_episodes,
            seed=seed + 1000,  # offset to avoid correlation with baseline seeds
            max_steps=max_steps,
            deterministic=deterministic,
            device=device,
            temporal_encoding=temporal_encoding,
        )
        name = concept_names[c_idx] if concept_names else f"concept_{c_idx}"
        per_concept.append({
            "index": c_idx,
            "name": name,
            "mean": mean_r,
            "std": std_r,
            "delta_mean": mean_r - baseline_mean,
        })

    return {
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "per_concept": per_concept,
    }


# ---------------------------------------------------------------------------
# Importance scores (normalised)
# ---------------------------------------------------------------------------

def compute_importance_scores(sweep_result: Dict) -> Dict:
    """
    Convert ablation sweep result into normalised importance scores.

    importance[i] = (baseline_mean - ablated_mean[i]) / baseline_mean
    clamped to [0, 1] so 0 = no impact, 1 = maximal impact.

    Parameters
    ----------
    sweep_result : dict returned by ablation_sweep()

    Returns
    -------
    dict with keys:
        baseline_mean, importance_scores (list of {name, score})
    """
    baseline = sweep_result["baseline_mean"]
    if abs(baseline) < 1e-6:
        baseline = 1.0  # avoid division by zero

    scores = []
    for entry in sweep_result["per_concept"]:
        raw_drop = baseline - entry["mean"]
        # Clamp: negative drop (ablated is better) → 0 importance
        #        large positive drop → up to 1.0 importance
        score = max(0.0, min(1.0, raw_drop / abs(baseline)))
        scores.append({
            "index": entry["index"],
            "name": entry["name"],
            "score": score,
            "ablated_mean": entry["mean"],
            "delta": -entry["delta_mean"],  # positive = performance dropped
        })

    return {
        "baseline_mean": baseline,
        "importance_scores": scores,
    }