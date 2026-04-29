"""
attribution.py — Decompose action logits into per-concept contributions.

Since the policy is:
  action_logits = action_net( mlp_extractor( c_t ) )

and c_t = [concept_0, concept_1, ..., concept_{K-1}], we can use a first-order
Taylor expansion around c_t to attribute each action logit to individual
concept dimensions.

The sensitivity matrix S[i, j] = ∂ action_logit_i / ∂ c_j
tells us: "increasing concept j by +1 changes action logit i by S[i,j] units."
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ppo.policy import ActorCriticPolicy


# ---------------------------------------------------------------------------
# Logit attribution for a single observation
# ---------------------------------------------------------------------------

def attribute_action_logits(
    policy: ActorCriticPolicy,
    obs_t: torch.Tensor,
    h_prev: Optional[torch.Tensor] = None,
    *,
    concept_names: Optional[List[str]] = None,
    action_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute per-concept contribution to each action logit via gradient attribution.

    For each action i and concept j, computes:
        sensitivity[i, j] = ∂ action_logit_i / ∂ c_j

    Parameters
    ----------
    policy : ActorCriticPolicy
    obs_t : [1, *obs_shape] observation tensor
    h_prev : optional GRU hidden state [1, hidden_dim]
    concept_names : optional list of concept names
    action_names : optional list of action names

    Returns
    -------
    dict with keys:
        action_logits, concept_values, sensitivity_matrix,
        concept_names, action_names,
        top_contributions_each_action (list of (concept_name, sensitivity) per action)
    """
    if policy.method == "no_concept":
        raise ValueError("attribution requires a concept method (vanilla_freeze or concept_actor_critic)")

    policy.set_training_mode(False)

    # We need c_t with grad enabled
    features = policy.extract_features(obs_t)

    if policy.method == "vanilla_freeze":
        c_t, h_new = policy.concept_net(features, h_prev)
    elif policy.method == "concept_actor_critic":
        c_t, h_new, _, _ = policy.concept_net(features, h_prev)
    else:
        raise ValueError(f"Unknown method: {policy.method}")

    # Make c_t a leaf with grad
    c_t.requires_grad_(True)
    if c_t.grad is not None:
        c_t.grad.zero_()

    latent = policy.mlp_extractor(c_t)
    action_logits = policy.action_net(latent)  # [1, n_actions]

    n_actions = action_logits.shape[1]
    n_concepts = c_t.shape[1]

    sensitivity = torch.zeros(n_actions, n_concepts)

    for a in range(n_actions):
        grad = torch.autograd.grad(
            action_logits[0, a],
            c_t,
            retain_graph=True,
            create_graph=False,
        )[0]
        sensitivity[a, :] = grad.detach().squeeze(0)

    action_logits_np = action_logits.detach().squeeze(0).cpu().numpy()
    concept_vals = c_t.detach().squeeze(0).cpu().numpy()
    sens_np = sensitivity.cpu().numpy()

    # Build human-readable names
    c_names = concept_names or [f"c{i}" for i in range(n_concepts)]
    a_names = action_names or [f"action_{i}" for i in range(n_actions)]

    # Top contributions per action
    top_contributions: List[List[Dict]] = []
    for a in range(n_actions):
        contributions = []
        for c in range(n_concepts):
            contributions.append({
                "concept_name": c_names[c],
                "concept_idx": c,
                "sensitivity": float(sens_np[a, c]),
                "weighted": float(sens_np[a, c] * concept_vals[c]),
            })
        contributions.sort(key=lambda x: abs(x["sensitivity"]), reverse=True)
        top_contributions.append(contributions)

    return {
        "action_logits": action_logits_np,
        "concept_values": concept_vals,
        "sensitivity_matrix": sens_np,
        "concept_names": c_names,
        "action_names": a_names,
        "top_contributions_each_action": top_contributions,
    }


# ---------------------------------------------------------------------------
# Full sensitivity matrix across multiple observations
# ---------------------------------------------------------------------------

def concept_sensitivity_matrix(
    policy: ActorCriticPolicy,
    env,
    *,
    n_steps: int = 100,
    seed: int = 42,
    deterministic: bool = True,
    device: torch.device = torch.device("cpu"),
    temporal_encoding: str = "none",
) -> Dict:
    """
    Accumulate the sensitivity matrix S[i,j] = ∂action_logit_i / ∂c_j
    averaged over n_steps of a rollout.

    Parameters
    ----------
    policy, env, n_steps, seed, deterministic, device, temporal_encoding :
        rollout configuration

    Returns
    -------
    dict with keys:
        mean_sensitivity [n_actions, n_concepts],
        std_sensitivity,
        concept_names,
        action_names,
        per_step_sensitivities (list of [n_actions, n_concepts] arrays)
    """
    from .intervention import _obs_to_tensor

    policy.set_training_mode(False)

    hidden_dim = getattr(policy.concept_net, "HIDDEN_DIM", 256) if policy.concept_net is not None else 256
    h_t = torch.zeros(1, hidden_dim, device=device) if temporal_encoding == "gru" else None

    obs, _ = env.reset(seed=seed)

    concept_names = getattr(env, "concept_names", None)
    if concept_names is None:
        concept_names = [f"c{i}" for i in range(policy.concept_dim)]

    action_enum = getattr(getattr(env, "unwrapped", env), "actions", None)
    if action_enum is not None:
        try:
            action_names = [action_enum(i).name for i in range(len(action_enum))]
        except Exception:
            action_names = [f"action_{i}" for i in range(policy.n_actions)]
    else:
        action_names = [f"action_{i}" for i in range(policy.n_actions)]

    all_sens: List[np.ndarray] = []

    for step in range(n_steps):
        obs_t = _obs_to_tensor(obs, device)
        result = attribute_action_logits(
            policy, obs_t, h_t,
            concept_names=concept_names,
            action_names=action_names,
        )
        all_sens.append(result["sensitivity_matrix"])

        # Step the environment
        features = policy.extract_features(obs_t)
        if policy.method == "vanilla_freeze":
            _, h_t = policy.concept_net(features, h_t)
        elif policy.method == "concept_actor_critic":
            _, h_t, _, _ = policy.concept_net(features, h_t)

        latent = policy.mlp_extractor(
            policy.concept_net(features, h_t)[0] if policy.method == "vanilla_freeze"
            else policy.concept_net(features, h_t)[0]
        )
        logits = policy.action_net(latent)
        if deterministic:
            action = int(logits.argmax(dim=1).item())
        else:
            action = int(torch.distributions.Categorical(logits=logits).sample().item())

        obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            obs, _ = env.reset()

    env.close()

    stacked = np.stack(all_sens, axis=0)  # [n_steps, n_actions, n_concepts]
    mean_sens = stacked.mean(axis=0)
    std_sens = stacked.std(axis=0)

    return {
        "mean_sensitivity": mean_sens,
        "std_sensitivity": std_sens,
        "concept_names": concept_names,
        "action_names": action_names,
        "per_step_sensitivities": all_sens,
        "n_steps": n_steps,
    }