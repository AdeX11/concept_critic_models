"""
plot_results.py — Load completed training runs and produce comparison plots.

Reads results saved by train.py from:
  results/none_{env}_seed{seed}/
  results/{concept_net}_{temporal}_{supervision}_{freeze}_{env}_seed{seed}/

Produces focused plots that each answer one scientific question, plus a
heatmap and scatter summary.

Usage:
  python plot_results.py --env tmaze --results_dir /path/to/results
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHODS = ["none", "cbm", "concept_ac"]

METHOD_LABELS = {
    "none":       "PPO (no concepts)",
    "cbm":        "CBM",
    "concept_ac": "Concept AC",
}

TEMPORAL_LABELS = {
    "gru":  "GRU",
    "none": "No memory",
}

SUPERVISION_LABELS = {
    "online":  "online (labels every iter)",
    "none":    "none (pure AC reward)",
    "queried": "queried (sparse labels)",
}

ENV_REWARD_REF = {
    "tmaze":              {"max": 0.89,   "label": "theoretical max (0.89)"},
    "cartpole":           {"max": 500.0,  "label": "max (500)"},
    "lunar_lander":       {"max": 200.0,  "label": "solved (200)"},
    "mountain_car":       {"max": -110.0, "label": "solved (-110)"},
    "hidden_velocity":    {"max": None,   "label": None},
    "dynamic_obstacles":  {"max": None,   "label": None},
}

ENV_ALL_CLASSIFICATION = {"tmaze", "dynamic_obstacles"}

DIR_PATTERN_NONE = re.compile(
    r"^none_(?P<env>.+)_seed(?P<seed>\d+)$"
)
DIR_PATTERN_CONCEPT = re.compile(
    r"^(?P<concept_net>cbm|concept_ac)_"
    r"(?P<temporal>none|stacked|gru)_"
    r"(?P<supervision>queried|online|none)_"
    r"(?P<freeze>frozen|coupled)_"
    r"(?P<env>.+)_"
    r"seed(?P<seed>\d+)$"
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def discover_runs(results_dir: str, env: str, methods: List[str]) -> Dict:
    """
    Scan results_dir for completed runs matching env and methods.

    Returns nested dict:
      runs[concept_net][temporal][supervision][freeze][seed] = {
          'rewards': np.ndarray,
          'mean_reward': float,
          'std_reward': float,
          'run_dir': str,
          'concept_acc': dict or None,
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

        m_none    = DIR_PATTERN_NONE.match(name)
        m_concept = DIR_PATTERN_CONCEPT.match(name)

        if m_none is not None:
            concept_net = "none"
            temporal    = "none"
            supervision = "online"
            freeze      = "frozen"
            run_env     = m_none.group("env")
            seed        = int(m_none.group("seed"))
        elif m_concept is not None:
            concept_net = m_concept.group("concept_net")
            temporal    = m_concept.group("temporal")
            supervision = m_concept.group("supervision")
            freeze      = m_concept.group("freeze")
            run_env     = m_concept.group("env")
            seed        = int(m_concept.group("seed"))
        else:
            continue

        if run_env != env or concept_net not in methods:
            continue

        rewards_path = os.path.join(run_dir, "rewards.npy")
        if not os.path.exists(rewards_path):
            print(f"[plot] skipping {name} — no rewards.npy")
            continue

        rewards = np.load(rewards_path)
        mean_r, std_r = 0.0, 0.0
        eval_path = os.path.join(run_dir, "eval.txt")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                for line in f:
                    if line.startswith("mean_reward="):
                        mean_r = float(line.strip().split("=")[1])
                    elif line.startswith("std_reward="):
                        std_r = float(line.strip().split("=")[1])

        concept_acc = None
        concept_acc_path = os.path.join(run_dir, "concept_acc.npz")
        if os.path.exists(concept_acc_path):
            d = np.load(concept_acc_path, allow_pickle=True)
            concept_acc = {
                "timesteps": d["timesteps"],
                "names":     [str(n) for n in d["names"]],
                "values":    d["values"],
            }

        runs.setdefault(concept_net, {})
        runs[concept_net].setdefault(temporal, {})
        runs[concept_net][temporal].setdefault(supervision, {})
        runs[concept_net][temporal][supervision].setdefault(freeze, {})
        runs[concept_net][temporal][supervision][freeze][seed] = {
            "rewards":     rewards,
            "mean_reward": mean_r,
            "std_reward":  std_r,
            "run_dir":     run_dir,
            "concept_acc": concept_acc,
        }
        print(f"[plot] loaded  {name}  ({len(rewards)} eps, eval={mean_r:.2f}±{std_r:.2f})")

    return runs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def smooth(values: np.ndarray, window: int = 20) -> np.ndarray:
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="same")


def _seed_mean_reward(seed_dict: dict) -> float:
    vals = [v["mean_reward"] for v in seed_dict.values()]
    return float(np.mean(vals)) if vals else float("nan")


def _seed_rewards_array(seed_dict: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (mean_curve, std_curve) across seeds, or (None, None)."""
    arrays = [v["rewards"] for v in seed_dict.values() if len(v["rewards"]) > 0]
    if not arrays:
        return None, None
    min_len = min(len(a) for a in arrays)
    arr = np.stack([a[:min_len] for a in arrays])
    return arr.mean(axis=0), arr.std(axis=0)


def _seed_final_concept_acc(seed_dict: dict) -> Optional[float]:
    vals = []
    for v in seed_dict.values():
        ca = v.get("concept_acc")
        if ca is not None and len(ca["timesteps"]) > 0:
            vals.append(float(ca["values"][-1].mean()))
    return float(np.mean(vals)) if vals else None


def _add_ref_line(ax, env: str) -> None:
    ref = ENV_REWARD_REF.get(env)
    if ref and ref["max"] is not None:
        ax.axhline(ref["max"], color="black", linestyle="--",
                   linewidth=1.0, alpha=0.5, label=ref["label"])


def _save(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {path}")


# ---------------------------------------------------------------------------
# Plot 1: Final reward bar chart — all runs ranked
# ---------------------------------------------------------------------------

def plot_final_reward_bar(runs: Dict, out_dir: str, env: str) -> None:
    """
    Horizontal bar chart of final eval reward for every run, sorted by reward.
    Gives an at-a-glance ranking of all 19 variants.
    """
    entries = []

    # PPO baseline
    if "none" in runs:
        for temp_d in runs["none"].values():
            for sup_d in temp_d.values():
                for freeze_d in sup_d.values():
                    r = _seed_mean_reward(freeze_d)
                    entries.append(("PPO baseline", r, "#888888"))

    colors = {
        "cbm":        "#ff7f0e",
        "concept_ac": "#2ca02c",
    }
    for concept_net in ["cbm", "concept_ac"]:
        if concept_net not in runs:
            continue
        for temporal, sup_d in runs[concept_net].items():
            for supervision, freeze_d in sup_d.items():
                for freeze, seed_dict in freeze_d.items():
                    r = _seed_mean_reward(seed_dict)
                    sup_str = {"online": "online", "none": "AC-only", "queried": "queried"}.get(supervision, supervision)
                    label = f"{METHOD_LABELS[concept_net]} | {TEMPORAL_LABELS[temporal]} | {sup_str} | {freeze}"
                    entries.append((label, r, colors[concept_net]))

    if not entries:
        return

    entries.sort(key=lambda x: x[1])
    labels, values, bar_colors = zip(*entries)

    fig, ax = plt.subplots(figsize=(10, max(6, len(entries) * 0.38)))
    y = np.arange(len(entries))
    bars = ax.barh(y, values, color=bar_colors, alpha=0.8, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Final eval reward", fontsize=12)
    ax.set_title(f"Final Reward — All Variants ({env})", fontsize=13)
    ax.grid(True, alpha=0.3, axis="x")

    ref = ENV_REWARD_REF.get(env)
    if ref and ref["max"] is not None:
        ax.axvline(ref["max"], color="black", linestyle="--", linewidth=1.0,
                   alpha=0.5, label=ref["label"])
        ax.legend(fontsize=9)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.005 * abs(bar.get_width() or 1),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "final_reward_bar.png"))


# ---------------------------------------------------------------------------
# Plot 2: Heatmap — temporal × freeze, per concept_net × supervision
# ---------------------------------------------------------------------------

def plot_heatmap(runs: Dict, out_dir: str, env: str) -> None:
    """
    For each (concept_net, supervision) combination, produce a
    temporal × freeze heatmap of final eval reward.
    Shows 2-factor interactions cleanly.
    """
    temporals = ["gru", "none"]
    freezes   = ["frozen", "coupled"]

    concept_nets = [cn for cn in ["cbm", "concept_ac"] if cn in runs]
    if not concept_nets:
        return

    # Collect all (concept_net, supervision) panels that have data
    panels = []
    for cn in concept_nets:
        supervisions = sorted(runs[cn].get("gru", {}).keys() or
                              runs[cn].get("none", {}).keys())
        # gather from all temporals
        all_sups = set()
        for temp_d in runs[cn].values():
            all_sups.update(temp_d.keys())
        for sup in sorted(all_sups):
            panels.append((cn, sup))

    if not panels:
        return

    n_panels = len(panels)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    global_min, global_max = float("inf"), float("-inf")
    data_cache = {}

    for cn, sup in panels:
        grid = np.full((len(temporals), len(freezes)), np.nan)
        for i, temp in enumerate(temporals):
            for j, frz in enumerate(freezes):
                try:
                    seed_dict = runs[cn][temp][sup][frz]
                    grid[i, j] = _seed_mean_reward(seed_dict)
                except KeyError:
                    pass
        data_cache[(cn, sup)] = grid
        valid = grid[~np.isnan(grid)]
        if len(valid):
            global_min = min(global_min, valid.min())
            global_max = max(global_max, valid.max())

    for idx, (cn, sup) in enumerate(panels):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        grid = data_cache[(cn, sup)]

        sup_str = SUPERVISION_LABELS.get(sup, sup)
        im = ax.imshow(grid, aspect="auto", vmin=global_min, vmax=global_max,
                       cmap="RdYlGn")
        ax.set_xticks(range(len(freezes)))
        ax.set_xticklabels(freezes, fontsize=10)
        ax.set_yticks(range(len(temporals)))
        ax.set_yticklabels([TEMPORAL_LABELS[t] for t in temporals], fontsize=10)
        ax.set_xlabel("Freeze", fontsize=10)
        ax.set_ylabel("Temporal encoding", fontsize=10)
        ax.set_title(f"{METHOD_LABELS[cn]}\n{sup_str}", fontsize=10)

        for i in range(len(temporals)):
            for j in range(len(freezes)):
                val = grid[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=11, fontweight="bold",
                            color="black" if 0.3 < (val - global_min) / max(global_max - global_min, 1e-6) < 0.7 else "white")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Final reward")

    # Hide unused axes
    for idx in range(len(panels), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(f"Final Reward: temporal × freeze ({env})", fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "heatmap_reward.png"))


# ---------------------------------------------------------------------------
# Plot 3: Focused learning curves — one question per subplot
# ---------------------------------------------------------------------------

def plot_focused_curves(runs: Dict, out_dir: str, env: str, window: int = 30) -> None:
    """
    4-panel figure, each subplot answering one ablation question using 2-4
    carefully chosen runs rather than all 19.

    Q1: Does temporal encoding matter?   → concept_ac, online, frozen: gru vs stacked vs none
    Q2: Does AC signal help over CBM?    → gru, online, frozen: cbm vs concept_ac
    Q3: Does e2e training help?          → concept_ac, gru, online: frozen vs coupled
    Q4: Does label supervision help AC?  → concept_ac, gru, frozen: online vs none
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    def _plot_curve(ax, seed_dict, label, color, lw=2.0):
        mean_c, std_c = _seed_rewards_array(seed_dict)
        if mean_c is None:
            return False
        mean_s = smooth(mean_c, window)
        std_s  = smooth(std_c,  window)
        x = np.arange(len(mean_s))
        ax.plot(x, mean_s, label=label, color=color, linewidth=lw)
        ax.fill_between(x, mean_s - std_s, mean_s + std_s, alpha=0.15, color=color)
        return True

    def _setup(ax, title, xlabel="Episode", ylabel="Reward"):
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        _add_ref_line(ax, env)

    # ---- Q1: temporal encoding ----
    ax = axes[0]
    plotted = False
    colors = {"gru": "#2ca02c", "none": "#d62728"}
    for temp in ["gru", "none"]:
        try:
            sd = runs["concept_ac"][temp]["online"]["frozen"]
            plotted |= _plot_curve(ax, sd, TEMPORAL_LABELS[temp], colors[temp])
        except KeyError:
            pass
    if "none" in runs:
        try:
            sd = list(runs["none"].values())[0]
            sd = list(sd.values())[0]
            sd = list(sd.values())[0]
            _plot_curve(ax, sd, "PPO baseline", "#888888", lw=1.4)
            plotted = True
        except (KeyError, StopIteration, IndexError):
            pass
    if plotted:
        _setup(ax, "Q1: Does temporal encoding matter?\n(Concept AC, online, frozen)")

    # ---- Q2: AC signal vs CBM ----
    ax = axes[1]
    plotted = False
    for cn, color in [("cbm", "#ff7f0e"), ("concept_ac", "#2ca02c")]:
        try:
            sd = runs[cn]["gru"]["online"]["frozen"]
            plotted |= _plot_curve(ax, sd, METHOD_LABELS[cn], color)
        except KeyError:
            pass
    if plotted:
        _setup(ax, "Q2: Does AC signal help over supervised CBM?\n(GRU, online, frozen)")

    # ---- Q3: frozen vs coupled ----
    ax = axes[2]
    plotted = False
    colors3 = {"frozen": "#2ca02c", "coupled": "#d62728"}
    for frz in ["frozen", "coupled"]:
        try:
            sd = runs["concept_ac"]["gru"]["online"][frz]
            plotted |= _plot_curve(ax, sd, frz, colors3[frz])
        except KeyError:
            pass
    if plotted:
        _setup(ax, "Q3: Does end-to-end training help?\n(Concept AC, GRU, online)")

    # ---- Q4: supervision=online vs none (pure AC) ----
    ax = axes[3]
    plotted = False
    colors4 = {"online": "#2ca02c", "none": "#9467bd"}
    labels4 = {"online": "online (labels + AC)", "none": "none (pure AC reward)"}
    for sup in ["online", "none"]:
        try:
            sd = runs["concept_ac"]["gru"][sup]["frozen"]
            plotted |= _plot_curve(ax, sd, labels4[sup], colors4[sup])
        except KeyError:
            pass
    if plotted:
        _setup(ax, "Q4: Does label supervision help Concept AC?\n(GRU, frozen)")

    fig.suptitle(f"Ablation Study — {env}", fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "learning_curves_ablation.png"))


# ---------------------------------------------------------------------------
# Plot 4: Concept accuracy vs final reward scatter
# ---------------------------------------------------------------------------

def plot_acc_vs_reward_scatter(runs: Dict, out_dir: str, env: str) -> None:
    """
    Scatter plot: x = final concept accuracy (mean over concepts),
                  y = final eval reward.
    Each point = one run. Shows whether concept quality correlates with reward.
    """
    colors = {"cbm": "#ff7f0e", "concept_ac": "#2ca02c"}
    markers = {"gru": "o", "none": "^"}

    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = False

    for concept_net in ["cbm", "concept_ac"]:
        if concept_net not in runs:
            continue
        for temporal, sup_d in runs[concept_net].items():
            for supervision, freeze_d in sup_d.items():
                for freeze, seed_dict in freeze_d.items():
                    acc = _seed_final_concept_acc(seed_dict)
                    rew = _seed_mean_reward(seed_dict)
                    if acc is None or np.isnan(rew):
                        continue
                    sup_str = {"online": "online", "none": "AC-only", "queried": "queried"}.get(supervision, supervision)
                    label = f"{METHOD_LABELS[concept_net]} | {TEMPORAL_LABELS[temporal]} | {sup_str} | {freeze}"
                    ax.scatter(acc, rew,
                               color=colors.get(concept_net, "#999"),
                               marker=markers.get(temporal, "o"),
                               s=80, alpha=0.85, label=label,
                               edgecolors="white", linewidths=0.5)
                    plotted = True

    if not plotted:
        plt.close(fig)
        return

    all_cls = env in ENV_ALL_CLASSIFICATION
    acc_label = "Final concept accuracy (↑)" if all_cls else "Final concept metric"
    ax.set_xlabel(acc_label, fontsize=12)
    ax.set_ylabel("Final eval reward (↑)", fontsize=12)
    ax.set_title(f"Concept Accuracy vs Task Reward — {env}", fontsize=13)
    ax.grid(True, alpha=0.3)

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8,
              loc="best", framealpha=0.9)

    # Legend for marker shapes
    import matplotlib.lines as mlines
    shape_legend = [
        mlines.Line2D([], [], color="gray", marker=m, linestyle="None",
                      markersize=8, label=TEMPORAL_LABELS[t])
        for t, m in markers.items()
    ]
    ax.add_artist(ax.legend(handles=shape_legend, fontsize=9,
                            loc="lower right", title="Temporal"))

    _add_ref_line(ax, env)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "scatter_acc_vs_reward.png"))


# ---------------------------------------------------------------------------
# Plot 5: Concept accuracy over time — GRU variants only
# ---------------------------------------------------------------------------

def plot_concept_acc_gru(runs: Dict, out_dir: str, env: str, window: int = 5) -> None:
    """
    Concept accuracy over training timesteps, restricted to GRU variants.
    GRU is the only architecture that can improve concept quality over time
    via BPTT; non-GRU variants are flat after first supervision, so they
    add clutter without insight.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False

    colors = {
        ("cbm",        "online"): "#ff7f0e",
        ("concept_ac", "online"): "#2ca02c",
        ("concept_ac", "none"):   "#9467bd",
        ("concept_ac", "queried"):"#17becf",
    }
    lss = {"frozen": "-", "coupled": "--"}

    for concept_net in ["cbm", "concept_ac"]:
        if concept_net not in runs:
            continue
        gru_d = runs[concept_net].get("gru", {})
        for supervision, freeze_d in gru_d.items():
            for freeze, seed_dict in freeze_d.items():
                all_ts, all_vals = None, []
                for v in seed_dict.values():
                    ca = v.get("concept_acc")
                    if ca is None or len(ca["timesteps"]) == 0:
                        continue
                    if all_ts is None:
                        all_ts = ca["timesteps"]
                    all_vals.append(ca["values"].mean(axis=1))

                if not all_vals or all_ts is None:
                    continue

                sup_str = {"online": "online", "none": "AC-only", "queried": "queried"}.get(supervision, supervision)
                label = f"{METHOD_LABELS[concept_net]} GRU | {sup_str} | {freeze}"
                color = colors.get((concept_net, supervision), "#999999")
                ls    = lss.get(freeze, "-")

                min_len = min(len(v) for v in all_vals)
                arr = np.stack([v[:min_len] for v in all_vals])
                mean_v = smooth(arr.mean(axis=0), window)
                ts_plot = all_ts[:min_len]

                ax.plot(ts_plot, mean_v, label=label, color=color,
                        linestyle=ls, linewidth=2.0)
                plotted = True

    if not plotted:
        plt.close(fig)
        return

    all_cls = env in ENV_ALL_CLASSIFICATION
    ylabel = "Concept accuracy (↑)" if all_cls else "Concept metric"
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Concept Accuracy Over Training — GRU variants ({env})", fontsize=13)
    if all_cls:
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5, label="perfect (1.0)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "concept_acc_gru.png"))


# ---------------------------------------------------------------------------
# Plot 6: Concept accuracy ablation — same 4 questions, y-axis = concept acc
# ---------------------------------------------------------------------------

def plot_concept_acc_ablation(runs: Dict, out_dir: str, env: str, window: int = 5) -> None:
    """
    4-panel ablation for concept accuracy, mirroring plot_focused_curves.
    Each panel answers one question using concept accuracy over timesteps.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    all_cls = env in ENV_ALL_CLASSIFICATION
    ylabel = "Concept accuracy (↑)" if all_cls else "Concept metric"

    def _plot_acc_curve(ax, seed_dict, label, color, ls="-", lw=2.0):
        all_ts, all_vals = None, []
        for v in seed_dict.values():
            ca = v.get("concept_acc")
            if ca is None or len(ca["timesteps"]) == 0:
                continue
            if all_ts is None:
                all_ts = ca["timesteps"]
            all_vals.append(ca["values"].mean(axis=1))
        if not all_vals or all_ts is None:
            return False
        min_len = min(len(v) for v in all_vals)
        arr = np.stack([v[:min_len] for v in all_vals])
        mean_v = smooth(arr.mean(axis=0), window)
        std_v  = smooth(arr.std(axis=0),  window)
        ts = all_ts[:min_len]
        ax.plot(ts, mean_v, label=label, color=color, linestyle=ls, linewidth=lw)
        ax.fill_between(ts, mean_v - std_v, mean_v + std_v, alpha=0.15, color=color)
        return True

    def _setup(ax, title):
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Timestep", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        if all_cls:
            ax.set_ylim(-0.02, 1.05)
            ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ---- Q1: temporal encoding ----
    ax = axes[0]
    colors = {"gru": "#2ca02c", "none": "#d62728"}
    plotted = False
    for temp in ["gru", "none"]:
        try:
            sd = runs["concept_ac"][temp]["online"]["frozen"]
            plotted |= _plot_acc_curve(ax, sd, TEMPORAL_LABELS[temp], colors[temp])
        except KeyError:
            pass
    if plotted:
        _setup(ax, "Q1: Does temporal encoding improve concept accuracy?\n(Concept AC, online, frozen)")

    # ---- Q2: AC signal vs CBM ----
    ax = axes[1]
    plotted = False
    for cn, color in [("cbm", "#ff7f0e"), ("concept_ac", "#2ca02c")]:
        try:
            sd = runs[cn]["gru"]["online"]["frozen"]
            plotted |= _plot_acc_curve(ax, sd, METHOD_LABELS[cn], color)
        except KeyError:
            pass
    if plotted:
        _setup(ax, "Q2: Does AC signal improve concept accuracy over CBM?\n(GRU, online, frozen)")

    # ---- Q3: frozen vs coupled ----
    ax = axes[2]
    plotted = False
    for frz, color in [("frozen", "#2ca02c"), ("coupled", "#d62728")]:
        try:
            sd = runs["concept_ac"]["gru"]["online"][frz]
            plotted |= _plot_acc_curve(ax, sd, frz, color)
        except KeyError:
            pass
    if plotted:
        _setup(ax, "Q3: Does end-to-end training affect concept accuracy?\n(Concept AC, GRU, online)")

    # ---- Q4: supervision=online vs none ----
    ax = axes[3]
    plotted = False
    labels4 = {"online": "online (labels + AC)", "none": "none (pure AC reward)"}
    colors4 = {"online": "#2ca02c", "none": "#9467bd"}
    for sup in ["online", "none"]:
        try:
            sd = runs["concept_ac"]["gru"][sup]["frozen"]
            plotted |= _plot_acc_curve(ax, sd, labels4[sup], colors4[sup])
        except KeyError:
            pass
    if plotted:
        _setup(ax, "Q4: Does label supervision help concept accuracy?\n(Concept AC, GRU, frozen)")

    fig.suptitle(f"Concept Accuracy Ablation — {env}", fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "concept_acc_ablation.png"))


# ---------------------------------------------------------------------------
# Plot 7: Concept accuracy heatmap — temporal × freeze, per concept_net × supervision
# ---------------------------------------------------------------------------

def plot_heatmap_concept_acc(runs: Dict, out_dir: str, env: str) -> None:
    """
    Mirrors plot_heatmap but colored by final concept accuracy instead of reward.
    """
    temporals = ["gru", "none"]
    freezes   = ["frozen", "coupled"]

    concept_nets = [cn for cn in ["cbm", "concept_ac"] if cn in runs]
    if not concept_nets:
        return

    panels = []
    for cn in concept_nets:
        all_sups = set()
        for temp_d in runs[cn].values():
            all_sups.update(temp_d.keys())
        for sup in sorted(all_sups):
            panels.append((cn, sup))

    if not panels:
        return

    n_panels = len(panels)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    global_min, global_max = float("inf"), float("-inf")
    data_cache = {}

    for cn, sup in panels:
        grid = np.full((len(temporals), len(freezes)), np.nan)
        for i, temp in enumerate(temporals):
            for j, frz in enumerate(freezes):
                try:
                    seed_dict = runs[cn][temp][sup][frz]
                    acc = _seed_final_concept_acc(seed_dict)
                    if acc is not None:
                        grid[i, j] = acc
                except KeyError:
                    pass
        data_cache[(cn, sup)] = grid
        valid = grid[~np.isnan(grid)]
        if len(valid):
            global_min = min(global_min, valid.min())
            global_max = max(global_max, valid.max())

    all_cls = env in ENV_ALL_CLASSIFICATION
    cbar_label = "Final concept accuracy" if all_cls else "Final concept metric"

    for idx, (cn, sup) in enumerate(panels):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        grid = data_cache[(cn, sup)]
        sup_str = SUPERVISION_LABELS.get(sup, sup)

        im = ax.imshow(grid, aspect="auto", vmin=global_min, vmax=global_max,
                       cmap="RdYlGn")
        ax.set_xticks(range(len(freezes)))
        ax.set_xticklabels(freezes, fontsize=10)
        ax.set_yticks(range(len(temporals)))
        ax.set_yticklabels([TEMPORAL_LABELS[t] for t in temporals], fontsize=10)
        ax.set_xlabel("Freeze", fontsize=10)
        ax.set_ylabel("Temporal encoding", fontsize=10)
        ax.set_title(f"{METHOD_LABELS[cn]}\n{sup_str}", fontsize=10)

        for i in range(len(temporals)):
            for j in range(len(freezes)):
                val = grid[i, j]
                if not np.isnan(val):
                    rel = (val - global_min) / max(global_max - global_min, 1e-6)
                    txt_color = "black" if 0.3 < rel < 0.7 else "white"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=11, fontweight="bold", color=txt_color)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

    for idx in range(len(panels), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(f"Final Concept Accuracy: temporal × freeze ({env})", fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "heatmap_concept_acc.png"))


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def write_summary_table(runs: Dict, out_dir: str) -> None:
    rows = []
    col_w = [12, 10, 12, 8, 12, 10, 10, 6]

    header = (
        f"{'concept_net':<{col_w[0]}}  {'temporal':<{col_w[1]}}  "
        f"{'supervision':<{col_w[2]}}  {'freeze':<{col_w[3]}}  "
        f"{'mean_reward':>{col_w[4]}}  {'std_reward':>{col_w[5]}}  "
        f"{'concept_acc':>{col_w[6]}}  {'seeds':>{col_w[7]}}"
    )
    sep = "-" * len(header)

    for concept_net in METHODS:
        if concept_net not in runs:
            continue
        for temporal, sup_d in runs[concept_net].items():
            for supervision, freeze_d in sup_d.items():
                for freeze, seed_dict in freeze_d.items():
                    rew  = _seed_mean_reward(seed_dict)
                    std  = float(np.std([v["mean_reward"] for v in seed_dict.values()]))
                    acc  = _seed_final_concept_acc(seed_dict)
                    n    = len(seed_dict)
                    acc_str = f"{acc:.3f}" if acc is not None else "  —"
                    rows.append((rew, (
                        f"{concept_net:<{col_w[0]}}  {temporal:<{col_w[1]}}  "
                        f"{supervision:<{col_w[2]}}  {freeze:<{col_w[3]}}  "
                        f"{rew:>{col_w[4]}.2f}  {std:>{col_w[5]}.2f}  "
                        f"{acc_str:>{col_w[6]}}  {n:>{col_w[7]}}"
                    )))

    rows.sort(key=lambda x: -x[0])
    lines = ["=" * len(header), header, sep] + [r[1] for r in rows] + ["=" * len(header)]
    table = "\n".join(lines)
    print("\n" + table)

    path = os.path.join(out_dir, "summary_table.txt")
    with open(path, "w") as f:
        f.write(table + "\n")
    print(f"[plot] saved → {path}")


def write_run_index(runs: Dict, out_dir: str) -> None:
    index = {}
    for concept_net, temp_d in runs.items():
        for temporal, sup_d in temp_d.items():
            for supervision, freeze_d in sup_d.items():
                for freeze, seed_dict in freeze_d.items():
                    key = f"{concept_net}_{temporal}_{supervision}_{freeze}"
                    index[key] = {
                        "seeds": list(seed_dict.keys()),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True,
                        choices=["cartpole", "dynamic_obstacles", "highway",
                                 "highway_state", "lunar_lander",
                                 "lunar_lander_state", "lunar_lander_pos_only",
                                 "mountain_car", "hidden_velocity", "tmaze"])
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--results_dir",  type=str, default="/results",
                        help="Directory containing completed run subdirectories")
    parser.add_argument("--output_dir",   type=str, default="plots",
                        help="Where to save plots and summary (kept local)")
    parser.add_argument("--smooth_window", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    runs = discover_runs(args.results_dir, args.env, args.methods)

    if not runs:
        print(f"[plot] no completed runs found for env={args.env} in {args.results_dir}")
        sys.exit(1)

    n_runs = sum(
        len(seed_d)
        for temp_d in runs.values()
        for sup_d in temp_d.values()
        for freeze_d in sup_d.values()
        for seed_d in freeze_d.values()
    )
    print(f"\n[plot] {n_runs} runs loaded → {args.output_dir}\n")

    plot_final_reward_bar(runs, args.output_dir, env=args.env)
    plot_heatmap(runs, args.output_dir, env=args.env)
    plot_heatmap_concept_acc(runs, args.output_dir, env=args.env)
    plot_focused_curves(runs, args.output_dir, env=args.env, window=args.smooth_window)
    plot_concept_acc_ablation(runs, args.output_dir, env=args.env)
    plot_acc_vs_reward_scatter(runs, args.output_dir, env=args.env)
    plot_concept_acc_gru(runs, args.output_dir, env=args.env)
    write_summary_table(runs, args.output_dir)
    write_run_index(runs, args.output_dir)

    print(f"\n[plot] done.")


if __name__ == "__main__":
    main()
