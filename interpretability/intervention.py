"""
intervention.py — Override concept bottleneck values and run counterfactual replays.

All interventions operate on c_t (the [B, n_concepts] tensor) after the concept
network produces it but before it enters mlp_extractor.  This is the cleanest
possible causal intervention because there is no skip connection around c_t
— every action must pass through these 7 numbers.
"""

from __future__ import annotations

import copy
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ppo.policy import ActorCriticPolicy


# ---------------------------------------------------------------------------
# Low-level concept manipulation
# ---------------------------------------------------------------------------

def intervene_on_concept(
    c_t: torch.Tensor,
    concept_idx: int,
    new_value: float,
) -> torch.Tensor:
    """
    Override a single concept dimension in the bottleneck.

    Parameters
    ----------
    c_t : [B, n_concepts] — original concept vector from concept_net
    concept_idx : int — which concept to override (0-based)
    new_value : float — forced value for that concept

    Returns
    -------
    c_t_modified : [B, n_concepts] — same tensor with concept_idx overwritten
    """
    c_mod = c_t.clone()
    c_mod[:, concept_idx] = new_value
    return c_mod


def intervene_on_concepts(
    c_t: torch.Tensor,
    overrides: Dict[int, float],
) -> torch.Tensor:
    """
    Override multiple concept dimensions simultaneously.

    Parameters
    ----------
    c_t : [B, n_concepts]
    overrides : dict mapping concept_index → forced_value

    Returns
    -------
    c_t_modified : [B, n_concepts]
    """
    c_mod = c_t.clone()
    for idx, val in overrides.items():
        c_mod[:, idx] = val
    return c_mod


# ---------------------------------------------------------------------------
# Observation helpers (mirror replay.py)
# ---------------------------------------------------------------------------

def _obs_to_tensor(obs, device: torch.device) -> torch.Tensor:
    if isinstance(obs, dict):
        return {
            k: torch.as_tensor(np.expand_dims(v, 0), dtype=torch.float32, device=device)
            for k, v in obs.items()
        }
    return torch.as_tensor(np.expand_dims(obs, 0), dtype=torch.float32, device=device)


def _render_frame(obs, env=None) -> np.ndarray:
    if isinstance(obs, dict):
        obs = obs["images"]
    arr = np.asarray(obs)
    if arr.ndim == 3 and arr.shape[0] >= 3:
        arr = arr[:3].transpose(1, 2, 0)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    if env is not None and hasattr(env, "render"):
        rendered = env.render()
        if rendered is not None:
            return np.asarray(rendered, dtype=np.uint8)
    return np.zeros((84, 84, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Single-step intervention forward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def _get_latent_with_override(
    policy: ActorCriticPolicy,
    features: torch.Tensor,
    h_prev: Optional[torch.Tensor],
    overrides: Dict[int, float],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Run _get_latent but override c_t before it hits mlp_extractor.

    Returns (latent, h_new, c_t_original, c_t_modified).
    """
    # Run the full _get_latent to get original c_t
    features_clone = features
    if policy.method == "no_concept":
        raise ValueError("intervention requires a concept method (vanilla_freeze or concept_actor_critic)")
    elif policy.method == "vanilla_freeze":
        c_t, h_new = policy.concept_net(features_clone, h_prev)
        c_t_original = c_t.clone()
        c_t_mod = intervene_on_concepts(c_t, overrides)
        latent = policy.mlp_extractor(c_t_mod)
    elif policy.method == "concept_actor_critic":
        c_t, h_new, concept_dists, V_c = policy.concept_net(features_clone, h_prev)
        c_t_original = c_t.clone()
        c_t_mod = intervene_on_concepts(c_t, overrides)
        latent = policy.mlp_extractor(c_t_mod)
    else:
        raise ValueError(f"Unknown method: {policy.method}")

    return latent, h_new, c_t_original, c_t_mod


# ---------------------------------------------------------------------------
# Full rollout with persistent intervention
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_with_intervention(
    policy: ActorCriticPolicy,
    env,
    overrides: Dict[int, float],
    *,
    seed: int = 42,
    max_steps: int = 200,
    deterministic: bool = True,
    device: torch.device = torch.device("cpu"),
    temporal_encoding: str = "none",
) -> Dict:
    """
    Run one episode with a persistent concept override active at every step.

    Parameters
    ----------
    policy : ActorCriticPolicy
    env : gym-like env with get_concept() method
    overrides : dict {concept_idx: forced_value}
    seed, max_steps, deterministic, device, temporal_encoding : standard rollout params

    Returns
    -------
    result : dict with keys:
        total_reward, steps, actions, original_concepts, modified_concepts,
        ground_truth_concepts, concept_names
    """
    policy.set_training_mode(False)
    env.reset(seed=seed)

    hidden_dim = getattr(policy.concept_net, "HIDDEN_DIM", 256) if policy.concept_net is not None else 256
    h_t = torch.zeros(1, hidden_dim, device=device) if temporal_encoding == "gru" else None

    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    step = 0

    actions: List[int] = []
    original_concepts: List[np.ndarray] = []
    modified_concepts: List[np.ndarray] = []
    ground_truth: List[np.ndarray] = []
    frames: List[np.ndarray] = []

    concept_names = getattr(env, "concept_names", [f"c{i}" for i in range(7)])

    while step < max_steps:
        obs_t = _obs_to_tensor(obs, device)
        features = policy.extract_features(obs_t)

        latent, h_new, c_orig, c_mod = _get_latent_with_override(
            policy, features, h_t, overrides,
        )

        action_logits = policy.action_net(latent)
        if deterministic:
            action = int(action_logits.argmax(dim=1).item())
        else:
            action = int(torch.distributions.Categorical(logits=action_logits).sample().item())

        truth = env.get_concept() if hasattr(env, "get_concept") else None

        actions.append(action)
        original_concepts.append(c_orig.squeeze(0).cpu().numpy().copy())
        modified_concepts.append(c_mod.squeeze(0).cpu().numpy().copy())
        if truth is not None:
            ground_truth.append(truth.copy())
        frames.append(_render_frame(obs, env=env))

        obs, reward, done, truncated, _ = env.step(action)
        total_reward += float(reward)
        step += 1

        if temporal_encoding == "gru":
            h_t = h_new
            if done or truncated:
                h_t = torch.zeros_like(h_t)

        if done or truncated:
            break

    env.close()
    return {
        "total_reward": total_reward,
        "steps": step,
        "actions": actions,
        "original_concepts": original_concepts,
        "modified_concepts": modified_concepts,
        "ground_truth_concepts": ground_truth,
        "concept_names": concept_names,
        "overrides": overrides,
        "frames": frames,
    }


# ---------------------------------------------------------------------------
# Counterfactual replay
# ---------------------------------------------------------------------------

def action_label(env, action: int) -> str:
    enum_obj = getattr(getattr(env, "unwrapped", env), "actions", None)
    if enum_obj is not None:
        try:
            return f"{action}:{enum_obj(action).name}"
        except Exception:
            pass
    return str(action)


def _annotate_counterfactual_frame(
    frame: np.ndarray,
    step: int,
    orig_action: int,
    cf_action: int,
    orig_concepts: np.ndarray,
    cf_concepts: np.ndarray,
    truth: Optional[np.ndarray],
    concept_names: List[str],
    action_label_fn: Callable[[int], str],
) -> Image.Image:
    image = Image.fromarray(frame)
    font = ImageFont.load_default()
    padding = 8
    line_h = 14

    lines = [
        f"Step {step}  |  Orig action: {action_label_fn(orig_action)}  →  CF action: {action_label_fn(cf_action)}",
        "",
        f"{'Concept':<22} {'Orig':>6} {'CF':>6} {'Truth':>6}",
        "-" * 44,
    ]

    n_show = min(len(concept_names), 7)
    for i in range(n_show):
        name = concept_names[i][:20]
        o_val = orig_concepts[i] if orig_concepts is not None else float("nan")
        c_val = cf_concepts[i] if cf_concepts is not None else float("nan")
        t_val = truth[i] if truth is not None else float("nan")
        marker = " ◀" if abs(o_val - c_val) > 0.01 else ""
        lines.append(f"{name:<22} {o_val:6.1f} {c_val:6.1f} {t_val:6.1f}{marker}")

    overlay_h = padding * 2 + line_h * len(lines)
    canvas = Image.new("RGB", (image.width, image.height + overlay_h), color=(18, 18, 18))
    canvas.paste(image, (0, 0))

    draw = ImageDraw.Draw(canvas)
    y = image.height + padding
    for line in lines:
        draw.text((padding, y), line, fill=(235, 235, 235), font=font)
        y += line_h

    return canvas


def counterfactual_replay(
    policy: ActorCriticPolicy,
    env,
    overrides: Dict[int, float],
    *,
    seed: int = 42,
    max_steps: int = 200,
    deterministic: bool = True,
    device: torch.device = torch.device("cpu"),
    temporal_encoding: str = "none",
    output_gif: Optional[str] = None,
    fps: int = 4,
) -> Dict:
    """
    Run a counterfactual replay comparing original vs intervened behavior.

    At every step both the original and counterfactual actions are computed.
    The agent follows the ORIGINAL action (so the trajectory is the same),
    but the counterfactual "what-if" action is recorded alongside.

    Parameters
    ----------
    policy, env, overrides, seed, max_steps, deterministic, device, temporal_encoding :
        Same as run_with_intervention.
    output_gif : optional path to save an annotated side-by-side GIF
    fps : frames per second for GIF

    Returns
    -------
    result : dict with keys:
        steps, actions_original, actions_counterfactual,
        concepts_original, concepts_counterfactual, ground_truth,
        concept_names, overrides, divergence_steps
    """
    policy.set_training_mode(False)

    hidden_dim = getattr(policy.concept_net, "HIDDEN_DIM", 256) if policy.concept_net is not None else 256
    h_orig = torch.zeros(1, hidden_dim, device=device) if temporal_encoding == "gru" else None
    h_cf = torch.zeros(1, hidden_dim, device=device) if temporal_encoding == "gru" else None

    obs, _ = env.reset(seed=seed)
    step = 0

    actions_orig: List[int] = []
    actions_cf: List[int] = []
    concepts_orig: List[np.ndarray] = []
    concepts_cf: List[np.ndarray] = []
    ground_truth: List[np.ndarray] = []
    divergence_steps: List[int] = []
    annotated_frames: List[Image.Image] = []

    concept_names = getattr(env, "concept_names", [f"c{i}" for i in range(7)])
    act_label = lambda a: action_label(env, a)

    while step < max_steps:
        obs_t = _obs_to_tensor(obs, device)
        features = policy.extract_features(obs_t)

        # --- Original pass ---
        latent_orig, h_orig_new, c_orig, _ = _get_latent_with_override(
            policy, features, h_orig, {},
        )
        logits_orig = policy.action_net(latent_orig)
        if deterministic:
            action_orig = int(logits_orig.argmax(dim=1).item())
        else:
            action_orig = int(torch.distributions.Categorical(logits=logits_orig).sample().item())

        # --- Counterfactual pass ---
        latent_cf, h_cf_new, _, c_cf = _get_latent_with_override(
            policy, features, h_cf, overrides,
        )
        logits_cf = policy.action_net(latent_cf)
        if deterministic:
            action_cf = int(logits_cf.argmax(dim=1).item())
        else:
            action_cf = int(torch.distributions.Categorical(logits=logits_cf).sample().item())

        truth = env.get_concept() if hasattr(env, "get_concept") else None

        actions_orig.append(action_orig)
        actions_cf.append(action_cf)
        concepts_orig.append(c_orig.squeeze(0).cpu().numpy().copy())
        concepts_cf.append(c_cf.squeeze(0).cpu().numpy().copy())
        if truth is not None:
            ground_truth.append(truth.copy())

        if action_orig != action_cf:
            divergence_steps.append(step)

        # Build annotated frame
        frame = _render_frame(obs, env=env)
        annotated_frames.append(
            _annotate_counterfactual_frame(
                frame, step, action_orig, action_cf,
                c_orig.squeeze(0).cpu().numpy(),
                c_cf.squeeze(0).cpu().numpy(),
                truth, concept_names, act_label,
            )
        )

        # Advance environment using ORIGINAL action
        obs, reward, done, truncated, _ = env.step(action_orig)
        step += 1

        if temporal_encoding == "gru":
            h_orig = h_orig_new
            h_cf = h_cf_new
            if done or truncated:
                h_orig = torch.zeros_like(h_orig)
                h_cf = torch.zeros_like(h_cf)

        if done or truncated:
            break

    env.close()

    # Save GIF if requested
    if output_gif is not None and annotated_frames:
        os.makedirs(os.path.dirname(output_gif) or ".", exist_ok=True)
        duration_ms = int(1000 / max(fps, 1))
        annotated_frames[0].save(
            output_gif,
            save_all=True,
            append_images=annotated_frames[1:],
            duration=duration_ms,
            loop=0,
        )
        print(f"[counterfactual_replay] saved {len(annotated_frames)} frames → {output_gif}")

    return {
        "steps": step,
        "actions_original": actions_orig,
        "actions_counterfactual": actions_cf,
        "concepts_original": concepts_orig,
        "concepts_counterfactual": concepts_cf,
        "ground_truth": ground_truth,
        "concept_names": concept_names,
        "overrides": overrides,
        "divergence_steps": divergence_steps,
    }