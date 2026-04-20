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
    r"(?P<training_mode>two_phase|end_to_end|joint)_"
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

    # Convert to absolute path immediately
    results_dir = os.path.abspath(results_dir)

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

        # Load concept accuracy log if available
        concept_acc = None
        concept_acc_path = os.path.join(run_dir, "concept_acc.npz")
        if os.path.exists(concept_acc_path):
            d = np.load(concept_acc_path, allow_pickle=True)
            concept_acc = {
                "timesteps": d["timesteps"],
                "names":     [str(n) for n in d["names"]],
                "values":    d["values"],   # [N_checkpoints, concept_dim]
            }

        runs.setdefault(method, {})
        runs[method].setdefault(training_mode, {})
        runs[method][training_mode].setdefault(temporal_encoding, {})
        runs[method][training_mode][temporal_encoding][seed] = {
            "rewards":     rewards,
            "mean_reward": mean_r,
            "std_reward":  std_r,
            "run_dir":     run_dir,
            "concept_acc": concept_acc,
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
    elif training_mode == "joint":
        extras.append("joint")
    if extras:
        return f"{base} ({', '.join(extras)})"
    return base


def _run_color(method: str, training_mode: str, temporal_encoding: str) -> str:
    base_color = METHOD_COLORS.get(method, "#999999")
    if method == "concept_actor_critic":
        if training_mode == "joint":
            return "#9467bd"   # purple for joint
        elif temporal_encoding == "gru":
            return "#2ca02c"
        elif temporal_encoding == "stacked":
            return "#17becf"
        else:  # none
            return "#bcbd22"
    if training_mode == "end_to_end":
        return "#d62728"
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

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Reward", fontsize=13)
    ax.set_title("Learning Curves", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, "learning_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved → {path}")

def plot_concept_accuracy_over_time(runs: Dict, out_dir: str) -> None:
    """
    Plot concept prediction quality vs timestep, split into a 2x2 grid:
      - Rows: Static vs. Temporal
      - Cols: Classification (Accuracy) vs. Regression (MSE)
    """
    concept_names_ref = None
    for method in METHODS:
        if method not in runs or method == "no_concept":
            continue
        for tm in runs[method].values():
            for te in tm.values():
                for v in te.values():
                    ca = v.get("concept_acc")
                    if ca and len(ca["names"]) > 0:
                        concept_names_ref = ca["names"]
                        break
                if concept_names_ref: break
            if concept_names_ref: break
        if concept_names_ref: break

    if concept_names_ref is None:
        print("[plot] no concept accuracy data to plot")
        return

    temporal_keywords = (
        "velocity", "move_direction", "vel_", "direction", 
        "crush", "broken", "_aggress", "accel"
    )
    
    classification_keywords = (
        "contact", "broken", "is_broken", "grasp", "_lane"
    )

    # Categorize indices
    cats = {"stat_class": [], "stat_reg": [], "temp_class": [], "temp_reg": []}
    for i, name in enumerate(concept_names_ref):
        is_temp = any(kw in name.lower() for kw in temporal_keywords)
        is_class = any(kw in name.lower() for kw in classification_keywords)
        
        if is_temp and is_class: cats["temp_class"].append(i)
        elif is_temp and not is_class: cats["temp_reg"].append(i)
        elif not is_temp and is_class: cats["stat_class"].append(i)
        else: cats["stat_reg"].append(i)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_map = {
        "stat_class": axes[0, 0], "stat_reg": axes[0, 1],
        "temp_class": axes[1, 0], "temp_reg": axes[1, 1]
    }

    plotted = False
    for method in METHODS:
        if method not in runs or method == "no_concept":
            continue
        for training_mode, te_dict in runs[method].items():
            for temporal_encoding, seed_dict in te_dict.items():
                all_ts = None
                all_vals = []
                for v in seed_dict.values():
                    ca = v.get("concept_acc")
                    if ca is None or len(ca["timesteps"]) == 0:
                        continue
                    if all_ts is None: all_ts = ca["timesteps"]
                    all_vals.append(ca["values"])

                if not all_vals or all_ts is None:
                    continue

                label = _run_label(method, training_mode, temporal_encoding)
                color = _run_color(method, training_mode, temporal_encoding)
                min_len = min(len(v) for v in all_vals)
                arr = np.stack([v[:min_len] for v in all_vals])
                mean_vals = arr.mean(axis=0)
                ts_plot = all_ts[:min_len]
                markevery = max(1, len(ts_plot) // 10)

                for cat_name, idxs in cats.items():
                    if not idxs: continue
                    cat_mean = mean_vals[:, idxs].mean(axis=1)
                    ax_map[cat_name].plot(ts_plot, cat_mean, label=label, color=color,
                                          linewidth=1.8, marker="o", markersize=4, markevery=markevery)
                plotted = True

    if not plotted:
        print("[plot] no concept accuracy data to plot")
        plt.close(fig)
        return

    # Formatting axes
    titles = {
        "stat_class": ("Static Classification", "Accuracy (↑ better)"),
        "stat_reg":   ("Static Regression", "MSE (↓ better)"),
        "temp_class": ("Temporal Classification", "Accuracy (↑ better)"),
        "temp_reg":   ("Temporal Regression", "MSE (↓ better)")
    }

    for cat_name, ax in ax_map.items():
        idxs = cats[cat_name]
        if not idxs:
            ax.set_visible(False)
            continue
        names = [concept_names_ref[i] for i in idxs]
        title, ylabel = titles[cat_name]
        
        ax.set_title(f"{title}\n({', '.join(names)})", fontsize=11)
        ax.set_xlabel("Timestep", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Concept Accuracy Over Training", fontsize=14, y=1.02)
    plt.tight_layout()

    path = os.path.join(out_dir, "concept_accuracy_over_time.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {path}")


def plot_concept_accuracy_per_concept(runs: Dict, out_dir: str) -> None:
    """
    Final concept accuracy bar chart, one bar per concept, grouped by method.
    Uses the last checkpoint in concept_acc.npz.
    Sorts the bars so Classification concepts and Regression concepts are grouped together.
    """
    # Collect final values for each method/variant
    method_labels = []
    concept_names = None
    method_final = {}  # label → [concept_dim] array

    for method in METHODS:
        if method not in runs or method == "no_concept":
            continue
        for training_mode, te_dict in runs[method].items():
            for temporal_encoding, seed_dict in te_dict.items():
                label = _run_label(method, training_mode, temporal_encoding)
                finals = []
                for v in seed_dict.values():
                    ca = v.get("concept_acc")
                    if ca is None or len(ca["timesteps"]) == 0:
                        continue
                    finals.append(ca["values"][-1])  # last checkpoint, [concept_dim]
                    if concept_names is None:
                        concept_names = ca["names"]
                if finals:
                    method_final[label] = np.mean(finals, axis=0)
                    method_labels.append(label)

    if not method_final or concept_names is None:
        print("[plot] no concept accuracy data for per-concept bar chart")
        return

    # --- NEW SORTING LOGIC ---
    # Use the fixed keyword list to identify classification tasks
    classification_keywords = ("contact", "broken", "is_broken", "grasp", "_lane")
    
    class_idxs = []
    reg_idxs = []
    
    for i, name in enumerate(concept_names):
        if any(kw in name.lower() for kw in classification_keywords):
            class_idxs.append(i)
        else:
            reg_idxs.append(i)
            
    # Create the new sorted order (Classification first, then Regression)
    sorted_idxs = class_idxs + reg_idxs
    sorted_concept_names = [concept_names[i] for i in sorted_idxs]
    # -------------------------

    n_concepts = len(sorted_concept_names)
    n_methods  = len(method_labels)
    x = np.arange(n_concepts)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(max(12, n_concepts * 1.5), 6))
    
    for i, label in enumerate(method_labels):
        # Grab the values and sort them to match the new axis order
        vals = method_final[label][sorted_idxs] 
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=label, alpha=0.8)

    # Add a vertical divider if both types of concepts are present
    if len(class_idxs) > 0 and len(reg_idxs) > 0:
        divider_x = len(class_idxs) - 0.5
        ax.axvline(x=divider_x, color='black', linestyle='--', alpha=0.6, linewidth=2)
        
        # Add text labels at the top to make the grouping clear
        ylim = ax.get_ylim()[1]
        ax.text(divider_x / 2 - 0.5, ylim * 0.95, 'Classification (Acc ↑)', 
                ha='center', fontsize=12, fontweight='bold', alpha=0.7)
        ax.text(divider_x + len(reg_idxs) / 2 + 0.5, ylim * 0.95, 'Regression (MSE ↓)', 
                ha='center', fontsize=12, fontweight='bold', alpha=0.7)

    ax.set_xlabel("Concept", fontsize=12)
    ax.set_ylabel("MSE (↓ better)  /  classification acc (↑ better)", fontsize=12)
    ax.set_title("Final Per-Concept Prediction Accuracy", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_concept_names, rotation=30, ha="right", fontsize=9)
    
    # Place legend outside the plot so it doesn't overlap the bars
    ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    path = os.path.join(out_dir, "concept_accuracy_per_concept.png")
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
                        choices=["cartpole", "dynamic_obstacles", "lunar_lander",
                                 "lunar_lander_state", "lunar_lander_pos_only", "mountain_car",
                                 "pick_place", "pick_place_state", "panda_pickplace", "panda_pickplace_state", "highway", "highway_state"])
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--results_dir",  type=str, default="./results",
                        help="Directory containing completed run subdirectories")
    parser.add_argument("--output_dir",   type=str, default="./plots",
                        help="Where to save plots and summary (kept local)")
    parser.add_argument("--smooth_window", type=int, default=30)
    # Options borrowed from envs/pick_place smoke test (accepted but optional for filtering)
    parser.add_argument("--render", action="store_true", help="(pick_place) include rendering variants")
    parser.add_argument("--state", action="store_true", help="(pick_place) include state-only variants")
    parser.add_argument("--max-force", type=float, default=None, help="(pick_place) max force threshold used during training")
    parser.add_argument("--seed", type=int, default=None, help="(pick_place) seed used during training")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Match train.py/compare.py naming: when --state is passed for pick_place look for pick_place_state
    env_search = args.env + ("_state" if getattr(args, "state", False) and args.env == "pick_place" else "")
    env_search = args.env + ("_state" if getattr(args, "state", False) and args.env == "highway" else "") if env_search == "highway" else env_search
    runs = discover_runs(args.results_dir, env_search, args.methods)

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
    plot_concept_accuracy_over_time(runs, args.output_dir)
    plot_concept_accuracy_per_concept(runs, args.output_dir)
    write_summary_table(runs, args.output_dir)
    write_run_index(runs, args.output_dir)

    print(f"\n[plot] done.")


if __name__ == "__main__":
    main()
