"""
steerability_eval.py — Concept intervention experiment for TMaze.

For each trained model, runs N_EPISODES episodes. At the junction step the agent
faces three conditions simultaneously (same corridor walk, three action queries):

  baseline  — no intervention; agent uses its own concept encoding
  correct   — cue concept forced to ground truth value
  flipped   — cue concept forced to the opposite of ground truth

Metrics per model:
  reward_baseline      mean episode reward under normal inference
  reward_correct       mean reward if the agent acted on the correct concept
  reward_flipped       mean reward if the agent acted on the wrong concept
  correct_change_rate  fraction of episodes where action changed under correct override
  flip_change_rate     fraction of episodes where action changed under flipped override
  steerability_score   (reward_correct - reward_baseline) normalised to [0,1]
  causal_sensitivity   reward_baseline - reward_flipped

Usage:
  python steerability_eval.py
  python steerability_eval.py --n_episodes 500 --results_dir /path/to/results
"""

import argparse
import os
import sys
import json

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.tmaze import make_single_tmaze_env
from ppo.policy import ActorCriticPolicy


# ---------------------------------------------------------------------------
# Models to evaluate
# ---------------------------------------------------------------------------

MODELS = [
    {
        "tag":        "concept_ac_gru_online_frozen_tmaze_seed42",
        "concept_net":"concept_ac",
        "temporal":   "gru",
        "label":      "Concept-AC | GRU | online | frozen",
    },
    {
        "tag":        "concept_ac_gru_none_frozen_tmaze_seed42",
        "concept_net":"concept_ac",
        "temporal":   "gru",
        "label":      "Concept-AC | GRU | none | frozen",
    },
    {
        "tag":        "concept_ac_gru_online_coupled_tmaze_seed42",
        "concept_net":"concept_ac",
        "temporal":   "gru",
        "label":      "Concept-AC | GRU | online | coupled",
    },
    {
        "tag":        "cbm_gru_online_frozen_tmaze_seed42",
        "concept_net":"cbm",
        "temporal":   "gru",
        "label":      "CBM | GRU | online | frozen",
    },
    {
        "tag":        "none_tmaze_seed42",
        "concept_net":"none",
        "temporal":   "none",
        "label":      "None (PPO baseline)",
    },
]


# ---------------------------------------------------------------------------
# Policy loader
# ---------------------------------------------------------------------------

def load_policy(model_dir: str, concept_net: str, temporal: str,
                device: torch.device) -> ActorCriticPolicy:
    """Reconstruct policy architecture and load saved weights."""
    # TMaze env metadata
    env = make_single_tmaze_env(seed=0, n_stack=1)
    obs_shape   = env.observation_space.shape
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
        features_dim     = 128,
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
# Reward inference from action + cue
# ---------------------------------------------------------------------------

def infer_reward(action: int, cue: int, n_steps: int) -> float:
    """
    TMaze reward at junction:
      +1.0 if action matches cue,  -1.0 otherwise
      plus -0.01 per step taken (time penalty)
    """
    time_penalty = -0.01 * n_steps
    if action == 1:          # choose LEFT
        outcome = 1.0 if cue == 0 else -1.0
    elif action == 2:        # choose RIGHT
        outcome = 1.0 if cue == 1 else -1.0
    else:                    # forward at junction = treated as wrong
        outcome = 1.0 if cue == 0 else -1.0
    return outcome + time_penalty


# ---------------------------------------------------------------------------
# Single model evaluation
# ---------------------------------------------------------------------------

def evaluate_steerability(policy: ActorCriticPolicy, concept_net: str,
                           n_episodes: int, seed: int,
                           device: torch.device) -> dict:
    """
    Runs n_episodes episodes. At the junction, queries the policy under
    three conditions (baseline / correct override / flipped override)
    without modifying the trajectory — all three actions are computed from
    the same GRU hidden state at the junction.

    Returns a dict of aggregate metrics.
    """
    env = make_single_tmaze_env(seed=seed, n_stack=1)
    n_concepts = len(env.task_types)

    rewards_base    = []
    rewards_correct = []
    rewards_flipped = []
    correct_changed = []
    flip_changed    = []
    junction_actions_base = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        h = None
        done = False
        n_steps = 0
        junction_reached = False
        ep_cue = int(env.env._cue) if hasattr(env, "env") else int(env._cue)

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            n_steps += 1

            # Detect junction from observation (at_junction is obs[3])
            at_junction = obs[3] > 0.5

            if at_junction and concept_net != "none":
                junction_reached = True

                # Baseline action (no intervention)
                action_base, h_base = policy.predict(obs_t, h, deterministic=True)
                a_base = int(action_base.item())

                # Build correct override: [true_cue, 1.0 (at_junction)]
                c_correct = torch.zeros(1, n_concepts, device=device)
                c_correct[0, 0] = float(ep_cue)        # cue concept
                c_correct[0, 1] = 1.0                   # at_junction concept

                # Build flipped override: opposite cue
                c_flipped = c_correct.clone()
                c_flipped[0, 0] = float(1 - ep_cue)

                action_cor, _ = policy.predict(obs_t, h, deterministic=True,
                                               concept_override=c_correct)
                action_fli, _ = policy.predict(obs_t, h, deterministic=True,
                                               concept_override=c_flipped)
                a_cor = int(action_cor.item())
                a_fli = int(action_fli.item())

                r_base    = infer_reward(a_base, ep_cue, n_steps)
                r_correct = infer_reward(a_cor,  ep_cue, n_steps)
                r_flipped = infer_reward(a_fli,  ep_cue, n_steps)

                rewards_base.append(r_base)
                rewards_correct.append(r_correct)
                rewards_flipped.append(r_flipped)
                correct_changed.append(int(a_cor != a_base))
                flip_changed.append(int(a_fli != a_base))
                junction_actions_base.append(a_base)

                # Step the env with baseline action to close the episode
                obs, _, terminated, truncated, _ = env.step(a_base)
                done = terminated or truncated
                h = h_base

            else:
                # Normal step (corridor walk)
                action, h = policy.predict(obs_t, h, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(int(action.item()))
                done = terminated or truncated

        if concept_net == "none" or not junction_reached:
            # For none model we just collect actual episode reward
            # Re-run cleanly
            pass

    env.close()

    if concept_net == "none" or len(rewards_base) == 0:
        # For the none baseline: run normally and record actual rewards
        env2 = make_single_tmaze_env(seed=seed, n_stack=1)
        ep_rewards = []
        for ep in range(n_episodes):
            obs, info = env2.reset()
            h = None
            done = False
            ep_r = 0.0
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action, h = policy.predict(obs_t, h, deterministic=True)
                obs, r, terminated, truncated, _ = env2.step(int(action.item()))
                ep_r += r
                done = terminated or truncated
            ep_rewards.append(ep_r)
        env2.close()
        mean_r = float(np.mean(ep_rewards))
        return {
            "reward_baseline":     mean_r,
            "reward_correct":      None,
            "reward_flipped":      None,
            "correct_change_rate": None,
            "flip_change_rate":    None,
            "steerability_score":  None,
            "causal_sensitivity":  None,
            "n_episodes":          n_episodes,
        }

    r_base    = float(np.mean(rewards_base))
    r_correct = float(np.mean(rewards_correct))
    r_flipped = float(np.mean(rewards_flipped))

    max_reward = 0.89   # +1.0 - 0.01*11 steps (straight walk)

    denom = max_reward - r_base
    steer = (r_correct - r_base) / denom if abs(denom) > 1e-6 else 0.0
    steer = float(np.clip(steer, -1.0, 1.0))

    return {
        "reward_baseline":     r_base,
        "reward_correct":      r_correct,
        "reward_flipped":      r_flipped,
        "correct_change_rate": float(np.mean(correct_changed)),
        "flip_change_rate":    float(np.mean(flip_changed)),
        "steerability_score":  steer,
        "causal_sensitivity":  r_base - r_flipped,
        "n_episodes":          n_episodes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="/glade/derecho/scratch/adadelek/results/tmaze_full")
    parser.add_argument("--n_episodes",  type=int, default=200)
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = args.results_dir

    print(f"\nSteerability Evaluation — TMaze")
    print(f"Episodes per model: {args.n_episodes}  |  seed: {args.seed}")
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
