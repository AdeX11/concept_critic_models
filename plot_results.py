"""
plot_results.py — Load completed training runs and produce comparison plots.

Reads results saved by train.py from results/<method>_<training_mode>_<temporal_encoding>_<env>_seed<seed>/
Produces the same outputs as compare.py but without re-running training.

Usage:
  # Plot all runs in results/ for a specific env
  python plot_results.py --env lunar_lander

  # Plot specific methods only
  python plot_results.py --env lunar_lander --methods no_concept vanilla_freeze concept_actor_critic

  # Load from a different results dir
  python plot_results.py --env lunar_lander --results_dir /path/to/results
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHODS = ["no_concept", "vanilla_freeze", "concept_actor_critic"]

METHOD_LABELS = {
    "no_concept":           "No Concept (PPO)",
    "vanilla_freeze":       "Vanilla Freeze (CBM)",
    "concept_actor_critic": "Concept Actor-Critic",
}

METHOD_COLORS = {
    "no_concept":           "#1f77b4",
    "vanilla_freeze":       "#ff7f0e",
    "concept_actor_critic": "#2ca02c",
}

# Directory name format: <method>_<training_mode>_<temporal_encoding>_<env>_seed<seed>
DIR_PATTERN = re.compile(
    r"^(?P<method>[^_]+(?:_[^_]+)*)_"
    r"(?P<training_mode>two_phase|end_to_end)_"
    r"(?P<temporal_encoding>gru|stacked|none)_"
    r"(?P<env>[^_]+(?:_[^_]+)*)_"
    r"seed(?P<seed>\d+)$"
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def discover_runs(results_dir: str, env: str, methods: List[str]) -> Dict:
    """
    Scan results_dir for completed runs matching env and methods.

    Returns nested dict:
      runs[method][training_mode][temporal_encoding][seed] = {
          'rewards': np.ndarray,
          'mean_reward': float,
          'std_reward': float,
          'run_dir': str,
      }
    """
    runs: Dict = {}

    if not os.path.isdir(results_dir):
        print(f"[plot] results_dir not found: {results_dir}")
        return runs

    for name in sorted(os.listdir(results_dir)):
        run_dir = os.path.join(results_dir, name)
        if not os.path.isdir(run_dir):
            continue

        m = DIR_PATTERN.match(name)
        if m is None:
            continue

        method           = m.group("method")
        training_mode    = m.group("training_mode")
        temporal_encoding = m.group("temporal_encoding")
        run_env          = m.group("env")
        seed             = int(m.group("seed"))

        if run_env != env:
            continue
        if method not in methods:
            continue

        rewards_path = os.path.join(run_dir, "rewards.npy")
        if not os.path.exists(rewards_path):
            print(f"[plot] skipping {name} — no rewards.npy")
            continue

        rewards = np.load(rewards_path)

        # Read eval.txt if available
        mean_r, std_r = 0.0, 0.0
        eval_path = os.path.join(run_dir, "eval.txt")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                for line in f:
                    if line.startswith("mean_reward="):
                        mean_r = float(line.strip().split("=")[1])
                    elif line.startswith("std_reward="):
                        std_r = float(line.strip().split("=")[1])

        runs.setdefault(method, {})
        runs[method].setdefault(training_mode, {})
        runs[method][training_mode].setdefault(temporal_encoding, {})
        runs[method][training_mode][temporal_encoding][seed] = {
            "rewards":     rewards,
            "mean_reward": mean_r,
            "std_reward":  std_r,
            "run_dir":     run_dir,
        }
        print(f"[plot] loaded  {name}  ({len(rewards)} episodes, eval={mean_r:.2f}±{std_r:.2f})")

    return runs


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def smooth(values: np.ndarray, window: int = 20) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def _run_label(method: str, training_mode: str, temporal_encoding: str) -> str:
    base = METHOD_LABELS.get(method, method)
    extras = []
    if method == "concept_actor_critic":
        extras.append(temporal_encoding)
    if training_mode == "end_to_end":
        extras.append("e2e")
    if extras:
        return f"{base} ({', '.join(extras)})"
    return base


def _run_color(method: str, training_mode: str, temporal_encoding: str) -> str:
    base_color = METHOD_COLORS.get(method, "#999999")
    # Vary shade for different temporal encodings within same method
    if method == "concept_actor_critic":
        if temporal_encoding == "gru":
            return "#2ca02c"
        elif temporal_encoding == "stacked":
            return "#17becf"
        else:  # none
            return "#bcbd22"
    if training_mode == "end_to_end":
        return "#d62728"  # red variant for end_to_end
    return base_color


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_learning_curves(runs: Dict, out_dir: str, window: int = 30) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    plotted = False
    for method in METHODS:
        if method not in runs:
            continue
        for training_mode, te_dict in runs[method].items():
            for temporal_encoding, seed_dict in te_dict.items():
                seed_rewards = [v["rewards"] for v in seed_dict.values() if len(v["rewards"]) > 0]
                if not seed_rewards:
                    continue

                min_len = min(len(r) for r in seed_rewards)
                arr  = np.stack([r[:min_len] for r in seed_rewards])
                mean = smooth(arr.mean(axis=0), window)
                std  = arr.std(axis=0)

                label = _run_label(method, training_mode, temporal_encoding)
                color = _run_color(method, training_mode, temporal_encoding)
                x = np.arange(min_len)

                ax.plot(x, mean, label=label, color=color, linewidth=1.8)
                ax.fill_between(
                    x,
                    smooth(mean - std, window),
                    smooth(mean + std, window),
                    alpha=0.15,
                    color=color,
                )
                plotted = True

    if not plotted:
        print("[plot] no data to plot for learning curves")
        plt.close(fig)
        return

    ax.set_xlabel("Episode", fontsize=13)
    ax.set_ylabel("Reward", fontsize=13)
    ax.set_title("Learning Curves (mean ± std across seeds)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, "learning_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved → {path}")


def write_summary_table(runs: Dict, out_dir: str) -> None:
    lines = ["=" * 70]
    lines.append(f"{'Run':<45}  {'Mean Reward':>12}  {'Std Reward':>10}  {'Seeds':>6}")
    lines.append("-" * 70)

    for method in METHODS:
        if method not in runs:
            continue
        for training_mode, te_dict in runs[method].items():
            for temporal_encoding, seed_dict in te_dict.items():
                label = _run_label(method, training_mode, temporal_encoding)
                seed_means = [v["mean_reward"] for v in seed_dict.values()]
                m = np.mean(seed_means)
                s = np.std(seed_means)
                n = len(seed_means)
                lines.append(f"{label:<45}  {m:>12.2f}  {s:>10.2f}  {n:>6}")

    lines.append("=" * 70)
    table = "\n".join(lines)
    print("\n" + table)

    path = os.path.join(out_dir, "summary_table.txt")
    with open(path, "w") as f:
        f.write(table + "\n")
    print(f"[plot] saved → {path}")


def write_run_index(runs: Dict, out_dir: str) -> None:
    """Save a JSON index of all loaded runs for reference."""
    index = {}
    for method, tm_dict in runs.items():
        for training_mode, te_dict in tm_dict.items():
            for temporal_encoding, seed_dict in te_dict.items():
                key = f"{method}_{training_mode}_{temporal_encoding}"
                index[key] = {
                    "seeds": list(seed_dict.keys()),
                    "n_episodes": {
                        seed: len(v["rewards"])
                        for seed, v in seed_dict.items()
                    },
                    "eval": {
                        seed: {"mean": v["mean_reward"], "std": v["std_reward"]}
                        for seed, v in seed_dict.items()
                    },
                }
    path = os.path.join(out_dir, "run_index.json")
    with open(path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"[plot] saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load completed train.py runs and produce comparison plots."
    )
    parser.add_argument("--env", required=True,
                        choices=["cartpole", "dynamic_obstacles", "lunar_lander"])
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--results_dir",  type=str, default="results",
                        help="Directory containing completed run subdirectories")
    parser.add_argument("--output_dir",   type=str, default="plots",
                        help="Where to save plots and summary")
    parser.add_argument("--smooth_window", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    runs = discover_runs(args.results_dir, args.env, args.methods)

    if not runs:
        print(f"[plot] no completed runs found for env={args.env} in {args.results_dir}")
        sys.exit(1)

    total = sum(
        len(seed_dict)
        for tm_dict in runs.values()
        for te_dict in tm_dict.values()
        for seed_dict in te_dict.values()
    )
    print(f"\n[plot] found {total} completed runs. generating plots → {args.output_dir}\n")

    plot_learning_curves(runs, args.output_dir, window=args.smooth_window)
    write_summary_table(runs, args.output_dir)
    write_run_index(runs, args.output_dir)

    print(f"\n[plot] done.")


if __name__ == "__main__":
    main()
