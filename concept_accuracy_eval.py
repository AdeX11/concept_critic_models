"""
concept_accuracy_eval.py — Per-phase concept accuracy evaluation for TMaze.

Loads each trained concept model and runs N deterministic episodes, computing
cue accuracy split across three phases:

  cue_phase      (pos 0–2)  : cue is visible in obs — tests encoding
  blank_corridor (pos 3–9)  : cue is hidden — tests memory maintenance
  junction       (pos 10)   : decision step — tests recall under pressure

This avoids the two problems with the training-buffer accuracy metric:
  1. Tiny sample: only ~12 junction steps per 16-step buffer fill
  2. Phase conflation: high cue-phase accuracy inflates the overall number

Usage:
  python concept_accuracy_eval.py
  python concept_accuracy_eval.py --n_episodes 500 --results_dir /path/to/results
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.tmaze import CUE_STEPS, make_single_tmaze_env
from ppo.policy import ActorCriticPolicy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = "/glade/derecho/scratch/adadelek/results/tmaze_full"
CORRIDOR_LEN = 10
CONCEPT_NAMES = ["cue", "at_junction"]
PHASES = ["cue_phase", "blank_corridor", "junction"]

PAT_CONCEPT = re.compile(
    r"^(?P<concept_net>cbm|concept_ac)"
    r"_(?P<temporal>none|stacked|gru)"
    r"_(?P<supervision>online|none|queried)"
    r"_(?P<freeze>frozen|coupled)"
    r"_tmaze_seed(?P<seed>\d+)$"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_inner_env(env):
    """Unwrap to the underlying TMazeEnv."""
    inner = env
    while hasattr(inner, "env"):
        inner = inner.env
    return inner


def _phase(pos: int) -> str:
    if pos < CUE_STEPS:
        return "cue_phase"
    if pos < CORRIDOR_LEN:
        return "blank_corridor"
    return "junction"


def load_policy(model_dir: str, concept_net: str, temporal: str,
                device: torch.device) -> ActorCriticPolicy:
    n_stack = 4 if temporal == "stacked" else 1
    env = make_single_tmaze_env(seed=0, n_stack=n_stack)
    policy = ActorCriticPolicy(
        obs_shape        = env.observation_space.shape,
        n_actions        = env.action_space.n,
        concept_net      = concept_net,
        task_types       = env.task_types,
        num_classes      = env.num_classes,
        concept_dim      = len(env.task_types),
        temporal_encoding= temporal,
        features_dim     = 128,
        net_arch         = [64, 64],
        device           = str(device),
    )
    env.close()
    state = torch.load(os.path.join(model_dir, "model.pt"), map_location=device)
    policy.load_state_dict(state)
    policy.to(device)
    policy.eval()
    return policy


# ---------------------------------------------------------------------------
# Per-phase evaluation
# ---------------------------------------------------------------------------

def evaluate(policy: ActorCriticPolicy, concept_net_type: str, temporal: str,
             n_episodes: int, seed: int, device: torch.device) -> dict:
    """
    Run n_episodes deterministic episodes. At each step, record the concept
    prediction and ground-truth concept, tagged by phase.
    Returns per-phase accuracy and episode reward stats.
    """
    n_stack = 4 if temporal == "stacked" else 1
    env = make_single_tmaze_env(seed=seed, n_stack=n_stack)
    inner = _get_inner_env(env)

    # Storage: phase → concept_name → list of (pred, truth) values
    preds  = {ph: {c: [] for c in CONCEPT_NAMES} for ph in PHASES}
    truths = {ph: {c: [] for c in CONCEPT_NAMES} for ph in PHASES}
    ep_rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        h_t = None
        ep_r = 0.0
        done = False

        while not done:
            pos     = inner._pos
            true_cue     = float(inner._cue)
            at_junc = float(pos == CORRIDOR_LEN)
            phase   = _phase(pos)

            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                features = policy.extract_features(obs_t)
                if concept_net_type == "concept_ac":
                    c_t, h_new, _, _ = policy.concept_net(features, h_t)
                else:  # cbm
                    c_t, h_new = policy.concept_net(features, h_t)
                latent = policy.mlp_extractor(c_t)
                action = policy.action_net(latent).argmax(dim=1)

            c_np = c_t.cpu().numpy().flatten()
            for i, cname in enumerate(CONCEPT_NAMES):
                preds[phase][cname].append(c_np[i])
                truths[phase][cname].append([true_cue, at_junc][i])

            if h_new is not None:
                h_t = h_new

            obs, r, terminated, truncated, _ = env.step(int(action.item()))
            ep_r += r
            done = terminated or truncated

        ep_rewards.append(ep_r)
        h_t = None  # reset hidden state for next episode

    env.close()

    # Compute per-phase accuracy for each concept
    phase_results = {}
    for phase in PHASES:
        phase_results[phase] = {}
        for cname in CONCEPT_NAMES:
            p = np.array(preds[phase][cname])
            t = np.array(truths[phase][cname])
            if len(p) == 0:
                phase_results[phase][cname] = None
                continue
            acc = float(np.mean(np.round(p) == np.round(t)))
            phase_results[phase][cname] = {
                "accuracy":  acc,
                "n_samples": int(len(p)),
            }

    return {
        "mean_reward": float(np.mean(ep_rewards)),
        "std_reward":  float(np.std(ep_rewards)),
        "n_episodes":  n_episodes,
        "phases":      phase_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--n_episodes",  type=int, default=500)
    parser.add_argument("--seed",        type=int, default=1)
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    all_results = {}

    header = f"{'Run':<52} {'Reward':>7}  {'cue_phase':>9}  {'blank_corr':>10}  {'junction':>8}"
    print(f"\nPer-phase concept (cue) accuracy — {args.n_episodes} deterministic episodes")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for name in sorted(os.listdir(args.results_dir)):
        run_dir = os.path.join(args.results_dir, name)
        if not os.path.isdir(run_dir):
            continue
        m = PAT_CONCEPT.match(name)
        if m is None:
            continue
        if not os.path.exists(os.path.join(run_dir, "model.pt")):
            continue

        concept_net = m.group("concept_net")
        temporal    = m.group("temporal")

        policy  = load_policy(run_dir, concept_net, temporal, device)
        results = evaluate(policy, concept_net, temporal,
                           args.n_episodes, args.seed, device)
        all_results[name] = results

        def fmt(phase):
            v = results["phases"][phase]["cue"]
            return f"{v['accuracy']:.3f}" if v else "  n/a"

        print(
            f"{name:<52} {results['mean_reward']:>7.3f}  "
            f"{fmt('cue_phase'):>9}  {fmt('blank_corridor'):>10}  {fmt('junction'):>8}"
        )

    print("=" * len(header))

    out_path = os.path.join(args.results_dir, "concept_accuracy_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
