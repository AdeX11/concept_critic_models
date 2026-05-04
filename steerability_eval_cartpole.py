"""
steerability_eval_cartpole.py — Concept intervention experiment for CartPole.

For each trained model, runs N_EPISODES episodes under three conditions
(separate episodes, fresh env each time):

  baseline  — no intervention; agent uses its own concept encoding
  correct   — all 4 concepts forced to ground truth at every step
  flipped   — all concepts ground truth EXCEPT the temporal concepts
              (cart_vel at index 1, pole_ang_vel at index 3), which are negated

Metrics per model:
  reward_baseline      mean episode reward under normal inference
  reward_correct       mean reward if the agent acted on ground-truth concepts
  reward_flipped       mean reward if temporal concepts were wrong
  correct_change_rate  fraction of steps where action changed (per-episode avg)
  flip_change_rate     fraction of steps where action changed (per-episode avg)
  steerability_score   normalized improvement score, clipped to [-1,1]
  causal_sensitivity   reward_baseline - reward_flipped

Usage:
  python steerability_eval_cartpole.py
  python steerability_eval_cartpole.py --n_episodes 500 --results_dir /path/to/results
"""

import argparse
import os
import sys
import json

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.cartpole import make_single_cartpole_env
from ppo.policy import ActorCriticPolicy


# ---------------------------------------------------------------------------
# Models to evaluate
# ---------------------------------------------------------------------------

MODELS = [
    # ---- PPO baseline ----
    {
        "tag":        "none_cartpole_seed42",
        "concept_net":"none",
        "temporal":   "none",
        "label":      "None (PPO baseline)",
    },
    # ---- CBM × {gru,none} × {frozen,coupled} ----
    {
        "tag":        "cbm_gru_online_frozen_cartpole_seed42",
        "concept_net":"cbm",
        "temporal":   "gru",
        "label":      "CBM | GRU | online | frozen",
    },
    {
        "tag":        "cbm_gru_online_coupled_cartpole_seed42",
        "concept_net":"cbm",
        "temporal":   "gru",
        "label":      "CBM | GRU | online | coupled",
    },
    {
        "tag":        "cbm_none_online_frozen_cartpole_seed42",
        "concept_net":"cbm",
        "temporal":   "none",
        "label":      "CBM | No memory | online | frozen",
    },
    {
        "tag":        "cbm_none_online_coupled_cartpole_seed42",
        "concept_net":"cbm",
        "temporal":   "none",
        "label":      "CBM | No memory | online | coupled",
    },
    # ---- Concept-AC × {gru,none} × {online,none} × {frozen,coupled} ----
    {
        "tag":        "concept_ac_gru_online_frozen_cartpole_seed42",
        "concept_net":"concept_ac",
        "temporal":   "gru",
        "label":      "Concept-AC | GRU | online | frozen",
    },
    {
        "tag":        "concept_ac_gru_online_coupled_cartpole_seed42",
        "concept_net":"concept_ac",
        "temporal":   "gru",
        "label":      "Concept-AC | GRU | online | coupled",
    },
    {
        "tag":        "concept_ac_gru_none_frozen_cartpole_seed42",
        "concept_net":"concept_ac",
        "temporal":   "gru",
        "label":      "Concept-AC | GRU | none | frozen",
    },
    {
        "tag":        "concept_ac_gru_none_coupled_cartpole_seed42",
        "concept_net":"concept_ac",
        "temporal":   "gru",
        "label":      "Concept-AC | GRU | none | coupled",
    },
    {
        "tag":        "concept_ac_none_online_frozen_cartpole_seed42",
        "concept_net":"concept_ac",
        "temporal":   "none",
        "label":      "Concept-AC | No memory | online | frozen",
    },
    {
        "tag":        "concept_ac_none_online_coupled_cartpole_seed42",
        "concept_net":"concept_ac",
        "temporal":   "none",
        "label":      "Concept-AC | No memory | online | coupled",
    },
    {
        "tag":        "concept_ac_none_none_frozen_cartpole_seed42",
        "concept_net":"concept_ac",
        "temporal":   "none",
        "label":      "Concept-AC | No memory | none | frozen",
    },
    {
        "tag":        "concept_ac_none_none_coupled_cartpole_seed42",
        "concept_net":"concept_ac",
        "temporal":   "none",
        "label":      "Concept-AC | No memory | none | coupled",
    },
]

# CartPole temporal concepts (indices 1 and 3): cart_vel, pole_ang_vel
TEMPORAL_INDICES = [1, 3]

# ---------------------------------------------------------------------------
# Policy loader
# ---------------------------------------------------------------------------

def load_policy(model_dir: str, concept_net: str, temporal: str,
                device: torch.device) -> ActorCriticPolicy:
    """Reconstruct policy architecture and load saved weights."""
    # n_stack=1: GRU handles temporal encoding, single frame per step
    # flicker_prob=0.0: evaluation uses clean frames (model was trained with flicker)
    env = make_single_cartpole_env(seed=0, n_stack=1, flicker_prob=0.0)
    # CartPole uses Dict obs space — .shape is None; extract sub-shapes
    obs_space = env.observation_space
    if hasattr(obs_space, "spaces"):
        obs_shape = {k: v.shape for k, v in obs_space.spaces.items()}
    else:
        obs_shape = obs_space.shape
    n_actions   = env.action_space.n
    task_types  = env.task_types
    num_classes = env.num_classes
    concept_dim = len(task_types)
    env.close()

    policy = ActorCriticPolicy(
        obs_shape        = obs_shape,
        n_actions        = n_actions,
        concept_net      = concept_net,
        task_types       = task_types,
        num_classes      = num_classes,
        concept_dim      = concept_dim,
        temporal_encoding= temporal,
        features_dim     = 512,
        net_arch         = [64, 64],
        device           = str(device),
    )

    model_path = os.path.join(model_dir, "model.pt")
    state = torch.load(model_path, map_location=device)
    policy.load_state_dict(state)
    policy.to(device)
    policy.eval()
    return policy


# ---------------------------------------------------------------------------
# Concept override builders
# ---------------------------------------------------------------------------

def build_correct_override(true_concept: np.ndarray, n_concepts: int,
                           device: torch.device) -> torch.Tensor:
    """All 4 concepts set to ground truth."""
    return torch.tensor(true_concept, dtype=torch.float32, device=device).unsqueeze(0)


def build_flipped_override(true_concept: np.ndarray, n_concepts: int,
                           device: torch.device) -> torch.Tensor:
    """All concepts = ground truth, but temporal concepts (cart_vel, pole_ang_vel)
    are negated — reversing their direction while preserving magnitude."""
    c = true_concept.copy()
    for idx in TEMPORAL_INDICES:
        c[idx] = -c[idx]
    return torch.tensor(c, dtype=torch.float32, device=device).unsqueeze(0)


# ---------------------------------------------------------------------------
# Single episode run with optional concept override at every step
# ---------------------------------------------------------------------------

def run_episode(policy: ActorCriticPolicy, env, device: torch.device,
                concept_override_fn=None) -> dict:
    """
    Runs one episode. If concept_override_fn is not None, calls it with the
    true concept at each step and passes the resulting tensor to policy.predict
    as concept_override.

    Returns dict with 'reward', 'length', 'actions' (list of ints).
    """
    obs, info = env.reset()
    true_concept = info["concept"].copy()
    n_concepts = len(true_concept)

    h = None
    done = False
    ep_reward = 0.0
    ep_length = 0
    actions = []

    while not done:
        if isinstance(obs, dict):
            obs_t = {k: torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
                     for k, v in obs.items()}
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        if concept_override_fn is not None:
            override = concept_override_fn(true_concept, n_concepts, device)
            action, h = policy.predict(obs_t, h, deterministic=True,
                                       concept_override=override)
        else:
            action, h = policy.predict(obs_t, h, deterministic=True)

        a = int(action.item())
        actions.append(a)

        obs, r, terminated, truncated, info = env.step(a)
        ep_reward += float(r)
        ep_length += 1
        done = terminated or truncated

        if not done:
            true_concept = info["concept"].copy()

    return {"reward": ep_reward, "length": ep_length, "actions": actions}


# ---------------------------------------------------------------------------
# Single model evaluation
# ---------------------------------------------------------------------------

def evaluate_steerability(policy: ActorCriticPolicy, concept_net: str,
                          n_episodes: int, seed: int,
                          device: torch.device) -> dict:
    """
    Runs n_episodes episodes under each of three conditions:
      baseline, correct override, flipped override.

    Returns dict of aggregate metrics.
    """
    # ---- baseline ----
    env = make_single_cartpole_env(seed=seed, n_stack=1, flicker_prob=0.0)
    baseline_rewards = []
    baseline_lengths = []
    baseline_all_actions = []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        result = run_episode(policy, env, device, concept_override_fn=None)
        baseline_rewards.append(result["reward"])
        baseline_lengths.append(result["length"])
        baseline_all_actions.append(result["actions"])
    env.close()

    r_base = float(np.mean(baseline_rewards))
    len_base = float(np.mean(baseline_lengths))

    if concept_net == "none":
        return {
            "reward_baseline":      r_base,
            "reward_correct":       None,
            "reward_flipped":       None,
            "length_baseline":      len_base,
            "length_correct":       None,
            "length_flipped":       None,
            "correct_change_rate":  None,
            "flip_change_rate":     None,
            "steerability_score":   None,
            "causal_sensitivity":   None,
            "n_episodes":           n_episodes,
        }

    # ---- correct override ----
    env = make_single_cartpole_env(seed=seed, n_stack=1, flicker_prob=0.0)
    correct_rewards = []
    correct_lengths = []
    correct_change_rates = []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        result = run_episode(policy, env, device,
                             concept_override_fn=build_correct_override)
        correct_rewards.append(result["reward"])
        correct_lengths.append(result["length"])
        base_acts = baseline_all_actions[ep]
        min_len = min(len(base_acts), len(result["actions"]))
        changes = sum(1 for i in range(min_len) if result["actions"][i] != base_acts[i])
        correct_change_rates.append(changes / max(min_len, 1))
    env.close()

    # ---- flipped override ----
    env = make_single_cartpole_env(seed=seed, n_stack=1, flicker_prob=0.0)
    flipped_rewards = []
    flipped_lengths = []
    flipped_change_rates = []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        result = run_episode(policy, env, device,
                             concept_override_fn=build_flipped_override)
        flipped_rewards.append(result["reward"])
        flipped_lengths.append(result["length"])
        base_acts = baseline_all_actions[ep]
        min_len = min(len(base_acts), len(result["actions"]))
        changes = sum(1 for i in range(min_len) if result["actions"][i] != base_acts[i])
        flipped_change_rates.append(changes / max(min_len, 1))
    env.close()

    r_correct = float(np.mean(correct_rewards))
    r_flipped = float(np.mean(flipped_rewards))
    len_correct = float(np.mean(correct_lengths))
    len_flipped = float(np.mean(flipped_lengths))

    # Steerability score: normalized improvement over baseline.
    # CartPole max reward = 500 (episode truncation limit, +1 per step alive).
    max_possible = 500.0
    max_observed = max(r_base, r_correct, r_flipped)
    ceiling = max(max_observed, max_possible)
    denom = ceiling - r_base
    steer = (r_correct - r_base) / denom if abs(denom) > 1e-6 else 0.0
    steer = float(np.clip(steer, -1.0, 1.0))

    return {
        "reward_baseline":      r_base,
        "reward_correct":       r_correct,
        "reward_flipped":       r_flipped,
        "length_baseline":      len_base,
        "length_correct":       len_correct,
        "length_flipped":       len_flipped,
        "correct_change_rate":  float(np.mean(correct_change_rates)),
        "flip_change_rate":     float(np.mean(flipped_change_rates)),
        "steerability_score":   steer,
        "causal_sensitivity":   r_base - r_flipped,
        "n_episodes":           n_episodes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="/glade/derecho/scratch/adadelek/results/cartpole_full")
    parser.add_argument("--n_episodes",  type=int, default=200)
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = args.results_dir

    print(f"\nSteerability Evaluation — CartPole")
    print(f"Episodes per model: {args.n_episodes}  |  seed: {args.seed}")
    print(f"Intervention targets: temporal concepts cart_vel (idx 1), pole_ang_vel (idx 3) — negated")
    print("=" * 80)

    header = (
        f"{'Model':<42} {'R_base':>7} {'R_cor':>7} {'R_flip':>7} "
        f"{'Cor%':>6} {'Flip%':>6} {'Steer':>6} {'Sens':>6}"
    )
    print(header)
    print("-" * 80)

    all_results = {}

    for m in MODELS:
        tag = m["tag"]
        model_dir = os.path.join(args.results_dir, tag)

        if not os.path.isdir(model_dir):
            print(f"  [skip] {tag} — directory not found")
            continue

        policy = load_policy(model_dir, m["concept_net"], m["temporal"], device)
        metrics = evaluate_steerability(policy, m["concept_net"],
                                        args.n_episodes, args.seed, device)

        def fmt(v, pct=False):
            if v is None:
                return "  n/a"
            if pct:
                return f"{v*100:5.1f}%"
            return f"{v:7.3f}"

        print(
            f"{m['label']:<42} "
            f"{fmt(metrics['reward_baseline'])} "
            f"{fmt(metrics['reward_correct'])} "
            f"{fmt(metrics['reward_flipped'])} "
            f"{fmt(metrics['correct_change_rate'], pct=True)} "
            f"{fmt(metrics['flip_change_rate'], pct=True)} "
            f"{fmt(metrics['steerability_score'])} "
            f"{fmt(metrics['causal_sensitivity'])}"
        )

        all_results[tag] = {"label": m["label"], **metrics}

    print("=" * 80)

    out_path = os.path.join(out_dir, "steerability_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()