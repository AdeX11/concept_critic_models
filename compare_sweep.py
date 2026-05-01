"""
compare_sweep.py — Discover and compare all sweep runs, produce plots and ranked tables.

Usage:
  python compare_sweep.py --env highway --results_dir results --output_plots final_plots --output_results final_results

Produces:
  final_plots/learning_curves_concept_methods.png   — vanilla_freeze + concept_actor_critic (6 runs)
  final_plots/learning_curves_all.png               — all 9 runs
  final_results/sweep_table.txt                     — ranked table (plain text)
  final_results/sweep_table.md                      — ranked table (markdown)
"""
import sys
import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIR_PATTERN = re.compile(
    r"^(?P<method>[^_]+(?:_[^_]+)*)_"
    r"(?P<training_mode>two_phase|end_to_end|joint)_"
    r"(?P<temporal_encoding>gru|stacked|none)_"
    r"(?P<env>[^_]+(?:_[^_]+)*)_"
    r"seed(?P<seed>\d+)$"
)

METHOD_COLORS = {
    "no_concept":           "#1f77b4",
    "vanilla_freeze":       "#ff7f0e",
    "concept_actor_critic": "#2ca02c",
}

TEMPORAL_LINESTYLES = {
    "none":    "-",
    "stacked": "--",
    "gru":     ":",
}

TEMPORAL_MARKERS = {
    "none":    "o",
    "stacked": "s",
    "gru":     "^",
}

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_runs(results_dir: str, env: str) -> Dict:
    """Scan results_dir for completed runs matching env."""
    runs: Dict = {}
    if not os.path.isdir(results_dir):
        return runs

    for name in sorted(os.listdir(results_dir)):
        run_dir = os.path.join(results_dir, name)
        if not os.path.isdir(run_dir):
            continue
        m = DIR_PATTERN.match(name)
        if m is None:
            continue
        method            = m.group("method")
        training_mode     = m.group("training_mode")
        temporal_encoding = m.group("temporal_encoding")
        run_env           = m.group("env")
        seed              = int(m.group("seed"))
        if run_env != env:
            continue

        rewards_path = os.path.join(run_dir, "rewards.npy")
        if not os.path.exists(rewards_path):
            continue

        rewards = np.load(rewards_path)

        # Read eval.txt
        mean_r, std_r = 0.0, 0.0
        eval_path = os.path.join(run_dir, "eval.txt")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                for line in f:
                    if line.startswith("mean_reward="):
                        mean_r = float(line.strip().split("=")[1])
                    elif line.startswith("std_reward="):
                        std_r = float(line.strip().split("=")[1])

        # Load concept accuracy
        concept_acc = None
        concept_acc_path = os.path.join(run_dir, "concept_acc.npz")
        if os.path.exists(concept_acc_path):
            d = np.load(concept_acc_path, allow_pickle=True)
            concept_acc = {
                "timesteps": d["timesteps"],
                "names":     [str(n) for n in d["names"]],
                "values":    d["values"],
            }

        runs.setdefault(method, {})
        runs[method].setdefault(training_mode, {})
        runs[method][training_mode].setdefault(temporal_encoding, {})
        runs[method][training_mode][temporal_encoding][seed] = {
            "rewards":      rewards,
            "mean_reward":  mean_r,
            "std_reward":   std_r,
            "run_dir":      run_dir,
            "concept_acc":  concept_acc,
        }
        print(f"[compare_sweep] loaded  {name}  (eval={mean_r:.2f}±{std_r:.2f})")

    return runs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def smooth(values: np.ndarray, window: int = 20) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def _get_temporal_concept_indices(concept_names: List[str]) -> List[int]:
    """Identify temporal concept indices from names using known keywords."""
    temporal_keywords = (
        "velocity", "vel_", "cue", "blindspot", "rel_vx",
        "move_direction", "direction", "temporal",
    )
    return [
        i for i, n in enumerate(concept_names)
        if any(kw in n.lower() for kw in temporal_keywords)
    ]


def _collect_concept_names(runs: Dict) -> List[str]:
    for method in runs:
        for tm in runs[method].values():
            for te in tm.values():
                for v in te.values():
                    ca = v.get("concept_acc")
                    if ca and len(ca["names"]) > 0:
                        return ca["names"]
    return []


def _concept_acc_to_metric(values: np.ndarray, concept_names: List[str],
                           task_types: Optional[List[str]] = None) -> np.ndarray:
    """
    Convert per-concept values to a unified metric.
    Classification concepts are already accuracy [0,1] ↑.
    Regression concepts are MSE — we convert to exp(-MSE) for [0,1] ↑.
    If task_types not provided, treat all non-binary-looking columns as regression.
    """
    result = np.array(values, dtype=np.float32).copy()
    for i in range(len(result)):
        # Heuristic: if value is between 0 and 1, it's likely accuracy (classification)
        # If value > 1, it's likely MSE (regression) — convert to exp(-MSE)
        if result[i] > 1.0:
            result[i] = float(np.exp(-result[i]))
        else:
            # Already in [0,1] — keep as is
            result[i] = float(np.clip(result[i], 0.0, 1.0))
    return result


def run_label(method: str, training_mode: str, temporal_encoding: str) -> str:
    parts = [method]
    if method != "no_concept":
        parts.append(temporal_encoding)
        parts.append(training_mode)
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_learning_curves(
    runs: Dict, methods: List[str], out_path: str, env: str, title: str,
    window: int = 30,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    plotted = False

    for method in methods:
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

                label = run_label(method, training_mode, temporal_encoding)
                color = METHOD_COLORS.get(method, "#999999")
                ls    = TEMPORAL_LINESTYLES.get(temporal_encoding, "-")
                marker = TEMPORAL_MARKERS.get(temporal_encoding, "o")
                x = np.arange(min_len)

                ax.plot(x, mean, label=label, color=color, linestyle=ls, linewidth=1.5,
                        marker=marker, markersize=3, markevery=max(1, min_len // 15))
                ax.fill_between(x, smooth(mean - std, window), smooth(mean + std, window),
                                alpha=0.1, color=color)
                plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[compare_sweep] saved → {out_path}")


# ---------------------------------------------------------------------------
# Ranked Table
# ---------------------------------------------------------------------------

def build_table(runs: Dict, output_dir: str) -> None:
    concept_names = _collect_concept_names(runs)
    temporal_idx = _get_temporal_concept_indices(concept_names)

    rows = []
    for method in ["no_concept", "vanilla_freeze", "concept_actor_critic"]:
        if method not in runs:
            continue
        for training_mode, te_dict in runs[method].items():
            for temporal_encoding, seed_dict in te_dict.items():
                for seed, v in seed_dict.items():
                    label = run_label(method, training_mode, temporal_encoding)
                    overall_acc = 0.0
                    temporal_acc = 0.0
                    ca = v.get("concept_acc")
                    if ca is not None and len(ca["timesteps"]) > 0:
                        final = _concept_acc_to_metric(ca["values"][-1], concept_names)
                        overall_acc = float(np.mean(final))
                        if temporal_idx:
                            temporal_acc = float(np.mean(final[temporal_idx]))
                    rows.append({
                        "label": label,
                        "method": method,
                        "training_mode": training_mode,
                        "temporal_encoding": temporal_encoding,
                        "mean_reward": v["mean_reward"],
                        "std_reward": v["std_reward"],
                        "concept_acc_overall": overall_acc,
                        "concept_acc_temporal": temporal_acc,
                    })

    # Sort by mean reward descending
    rows.sort(key=lambda r: r["mean_reward"], reverse=True)

    os.makedirs(output_dir, exist_ok=True)

    # Text table
    txt_path = os.path.join(output_dir, "sweep_table.txt")
    with open(txt_path, "w") as f:
        header = f"{'Rank':<5} {'Model':<45} {'Mean Reward':>12} {'Std':>8} {'Conc Acc':>9} {'Temp Acc':>9}"
        sep = "-" * len(header)
        f.write(f"{header}\n{sep}\n")
        for i, row in enumerate(rows):
            f.write(
                f"{i+1:<5} {row['label']:<45} {row['mean_reward']:>12.2f} "
                f"{row['std_reward']:>8.2f} {row['concept_acc_overall']:>9.3f} "
                f"{row['concept_acc_temporal']:>9.3f}\n"
            )
        if temporal_idx:
            temporal_names = [concept_names[i] for i in temporal_idx]
            f.write(f"\nTemporal concepts: {', '.join(temporal_names)}\n")
    print(f"[compare_sweep] saved → {txt_path}")

    # Markdown table
    md_path = os.path.join(output_dir, "sweep_table.md")
    with open(md_path, "w") as f:
        f.write("| Rank | Model | Mean Reward | Std | Concept Acc | Temporal Acc |\n")
        f.write("|------|-------|-------------|-----|-------------|-------------|\n")
        for i, row in enumerate(rows):
            f.write(
                f"| {i+1} | {row['label']} | {row['mean_reward']:.2f} | "
                f"{row['std_reward']:.2f} | {row['concept_acc_overall']:.3f} | "
                f"{row['concept_acc_temporal']:.3f} |\n"
            )
        if temporal_idx:
            temporal_names = [concept_names[i] for i in temporal_idx]
            f.write(f"\n*Temporal concepts: {', '.join(temporal_names)}*\n")
    print(f"[compare_sweep] saved → {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare sweep runs and produce ranked tables.")
    parser.add_argument("--env", required=True, help="Environment name")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_plots", type=str, default="final_plots")
    parser.add_argument("--output_results", type=str, default="final_results")
    args = parser.parse_args()

    runs = discover_runs(args.results_dir, args.env)
    if not runs:
        print(f"[compare_sweep] no runs found for env={args.env} in {args.results_dir}")
        sys.exit(1)

    os.makedirs(args.output_plots, exist_ok=True)
    os.makedirs(args.output_results, exist_ok=True)

    # Plot 1: concept methods only (vanilla_freeze + concept_actor_critic)
    concept_methods = ["vanilla_freeze", "concept_actor_critic"]
    plot_learning_curves(
        runs, concept_methods,
        out_path=os.path.join(args.output_plots, "learning_curves_concept_methods.png"),
        env=args.env,
        title=f"Concept Methods Comparison — {args.env}",
    )

    # Plot 2: all 9 models
    plot_learning_curves(
        runs, ["no_concept", "vanilla_freeze", "concept_actor_critic"],
        out_path=os.path.join(args.output_plots, "learning_curves_all.png"),
        env=args.env,
        title=f"Full Sweep Comparison — {args.env}",
    )

    # Table
    build_table(runs, args.output_results)

    print(f"\n[compare_sweep] done. Outputs in {args.output_plots}/ and {args.output_results}/")


if __name__ == "__main__":
    main()