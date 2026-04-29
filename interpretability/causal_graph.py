"""
causal_graph.py — Discover inter-concept causal dependencies across time.

For concept bottleneck models with temporal encoding (GRU), we can measure
how intervening on concept A at time t affects concept B at time t+1.

This builds a K×K causal adjacency matrix where entry [i,j] measures:
  "changing concept i changes the predicted value of concept j at the next step"

Uses the intervention engine from intervention.py as the causal probe.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ppo.policy import ActorCriticPolicy
from .intervention import _obs_to_tensor, _get_latent_with_override


# ---------------------------------------------------------------------------
# Single intervention pair
# ---------------------------------------------------------------------------

def measure_concept_influence(
    policy: ActorCriticPolicy,
    env,
    source_idx: int,
    target_idx: int,
    intervention_value: float,
    *,
    n_steps: int = 200,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
    temporal_encoding: str = "none",
) -> Dict:
    """
    Measure how overriding source concept affects target concept at next step.

    At each step:
      1. Run baseline forward pass → record baseline target value
      2. Run with source overridden → record intervened target value
      3. Step environment with baseline action

    The "influence" is mean(|delta_target|) across steps.

    Parameters
    ----------
    policy, env, n_steps, seed, device, temporal_encoding : rollout config
    source_idx : int — which concept to override
    target_idx : int — which concept to measure effect on
    intervention_value : float — value to set source concept to

    Returns
    -------
    dict with keys: source_idx, target_idx, intervention_value,
                    mean_abs_delta, all_deltas, all_baseline, all_intervened
    """
    if policy.method == "no_concept":
        raise ValueError("causal graph requires a concept method")

    policy.set_training_mode(False)

    hidden_dim = getattr(policy.concept_net, "HIDDEN_DIM", 256) if policy.concept_net is not None else 256
    h_base = torch.zeros(1, hidden_dim, device=device) if temporal_encoding == "gru" else None
    h_int = torch.zeros(1, hidden_dim, device=device) if temporal_encoding == "gru" else None

    obs, _ = env.reset(seed=seed)

    deltas: List[float] = []
    baseline_vals: List[float] = []
    intervened_vals: List[float] = []

    for step in range(n_steps):
        obs_t = _obs_to_tensor(obs, device)
        features = policy.extract_features(obs_t)

        # --- Baseline pass ---
        latent_base, h_base_new, c_base, _ = _get_latent_with_override(
            policy, features, h_base, {},
        )
        base_val = float(c_base[0, target_idx].item())

        # --- Intervened pass ---
        latent_int, h_int_new, _, c_int = _get_latent_with_override(
            policy, features, h_int, {source_idx: intervention_value},
        )
        int_val = float(c_int[0, target_idx].item())

        delta = abs(int_val - base_val)
        deltas.append(delta)
        baseline_vals.append(base_val)
        intervened_vals.append(int_val)

        # Step environment with baseline action
        action_logits = policy.action_net(latent_base)
        action = int(action_logits.argmax(dim=1).item())
        obs, _, done, truncated, _ = env.step(action)

        if temporal_encoding == "gru":
            h_base = h_base_new
            h_int = h_int_new
            if done or truncated:
                h_base = torch.zeros_like(h_base)
                h_int = torch.zeros_like(h_int)

        if done or truncated:
            obs, _ = env.reset()

    env.close()

    return {
        "source_idx": source_idx,
        "target_idx": target_idx,
        "intervention_value": intervention_value,
        "mean_abs_delta": float(np.mean(deltas)),
        "all_deltas": deltas,
        "all_baseline": baseline_vals,
        "all_intervened": intervened_vals,
    }


# ---------------------------------------------------------------------------
# Build full K×K adjacency matrix
# ---------------------------------------------------------------------------

def discover_concept_dependencies(
    policy: ActorCriticPolicy,
    env_factory,
    *,
    n_steps: int = 200,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
    temporal_encoding: str = "none",
    concept_names: Optional[List[str]] = None,
) -> Dict:
    """
    Discover causal dependencies between all pairs of concepts.

    For each ordered pair (source, target):
      override source → 0.0 and measure |delta in target|

    Also includes diagonal entries (self-influence).

    Parameters
    ----------
    policy, env_factory, n_steps, seed, device, temporal_encoding : config
    concept_names : optional list of concept names

    Returns
    -------
    dict with keys:
        adjacency_matrix [K, K], concept_names,
        source_concept, target_concept (aligned),
        per_pair (list of detailed results)
    """
    n_concepts = policy.concept_dim
    env = env_factory()

    c_names = concept_names or getattr(env, "concept_names", None) or [f"c{i}" for i in range(n_concepts)]
    env.close()

    adjacency = np.zeros((n_concepts, n_concepts))
    per_pair: List[Dict] = []

    for src in range(n_concepts):
        for tgt in range(n_concepts):
            env = env_factory()
            result = measure_concept_influence(
                policy, env, src, tgt, intervention_value=0.0,
                n_steps=n_steps,
                seed=seed + src * n_concepts + tgt,
                device=device,
                temporal_encoding=temporal_encoding,
            )
            adjacency[src, tgt] = result["mean_abs_delta"]
            per_pair.append(result)
            env.close()

    return {
        "adjacency_matrix": adjacency,
        "concept_names": c_names,
        "source_concept": c_names,
        "target_concept": c_names,
        "per_pair": per_pair,
    }


# ---------------------------------------------------------------------------
# High-level wrapper (alias)
# ---------------------------------------------------------------------------

def build_causal_adjacency_matrix(
    policy: ActorCriticPolicy,
    env_factory,
    *,
    n_steps: int = 200,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
    temporal_encoding: str = "none",
    concept_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Convenience wrapper returning just the adjacency matrix.

    Returns K×K numpy array where entry [i,j] is the mean |delta|
    in concept j when concept i is set to 0.0.
    """
    result = discover_concept_dependencies(
        policy, env_factory,
        n_steps=n_steps,
        seed=seed,
        device=device,
        temporal_encoding=temporal_encoding,
        concept_names=concept_names,
    )
    return result["adjacency_matrix"]