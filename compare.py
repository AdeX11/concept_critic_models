"""
compare.py — Comparison script: runs all 3 methods × N seeds × chosen environment.

Produces:
  1. Learning curves (mean reward vs timesteps)  → learning_curves.png
  2. Concept accuracy per concept (static vs temporal split)  → concept_accuracy.png
  3. Summary table (mean ± std reward)  → summary_table.txt
  4. Concept critic value correlation (concept_actor_critic only)  → value_correlation.png

Usage:
  python compare.py --env lunar_lander --seeds 42 123 456 --total_timesteps 500000

All results are saved to compare_results/<env>_<timestamp>/
"""

import argparse
import json
import os
import sys
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.cartpole          import make_cartpole_env,           make_single_cartpole_env
from envs.dynamic_obstacles import make_dynamic_obstacles_env,  make_single_dynamic_obstacles_env
from envs.lunar_lander      import make_lunar_lander_env,       make_single_lunar_lander_env
from ppo.ppo                import PPO


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_obs_shape(env):
    obs_space = env.observation_space
    if hasattr(obs_space, "spaces"):
        return {k: v.shape for k, v in obs_space.spaces.items()}
    return obs_space.shape


def make_env_and_policy_kwargs(env_name: str, n_envs: int, seed: int, n_stack: int = 1):
    if env_name == "cartpole":
        vec_env    = make_cartpole_env(n_envs, seed, n_stack=n_stack)
        single_env = make_single_cartpole_env(seed, n_stack=n_stack)
    elif env_name == "dynamic_obstacles":
        vec_env    = make_dynamic_obstacles_env(n_envs, seed, n_stack=n_stack)
        single_env = make_single_dynamic_obstacles_env(seed, n_stack=n_stack)
    elif env_name == "lunar_lander":
        vec_env    = make_lunar_lander_env(n_envs, seed, n_stack=n_stack)
        single_env = make_single_lunar_lander_env(seed, n_stack=n_stack)
    else:
        raise ValueError(f"Unknown env: {env_name}")

    policy_kwargs = dict(
        obs_shape    = get_obs_shape(vec_env),
        n_actions    = vec_env.single_action_space.n,
        task_types   = single_env.task_types,
        num_classes  = single_env.num_classes,
        concept_dim  = len(single_env.task_types),
        features_dim = 512,
        net_arch     = [64, 64],
    )
    return vec_env, single_env, policy_kwargs


def smooth(values: np.ndarray, window: int = 20) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def run_single(
    method: str,
    env_name: str,
    seed: int,
    total_timesteps: int,
    num_labels: int,
    query_num_times: int,
    n_envs: int,
    n_steps: int,
    batch_size: int,
    device: str,
    temporal_encoding: str = "none",
    training_mode: str = "two_phase",
) -> dict:
    """Train one method/seed and return metrics dict."""
    set_seed(seed)
    n_stack = 4 if temporal_encoding == "stacked" else 1
    vec_env, single_env, policy_kwargs = make_env_and_policy_kwargs(
        env_name, n_envs, seed, n_stack=n_stack
    )
    dev = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    policy_kwargs["device"] = dev
    policy_kwargs["temporal_encoding"] = temporal_encoding

    model = PPO(
        method        = method,
        env           = vec_env,
        policy_kwargs = policy_kwargs,
        n_steps       = n_steps,
        n_epochs      = 10,
        batch_size    = batch_size,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        ent_coef      = 0.01,
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
        learning_rate = 3e-4,
        lambda_v      = 0.5,
        lambda_s      = 0.5,
        training_mode = training_mode,
        normalize_advantage = True,
        seed          = seed,
        device        = device,
        verbose       = 1,
    )

    labels_per_query = max(1, num_labels // max(query_num_times, 1))
    model.learn(
        total_timesteps       = total_timesteps,
        query_num_times       = query_num_times if method != "no_concept" else 0,
        query_labels_per_time = labels_per_query,
    )

    mean_r, std_r = model.evaluate(n_episodes=20, deterministic=True)

    # ---- Concept accuracy evaluation ----
    concept_metrics = _evaluate_concepts(model, single_env)

    vec_env.close()
    single_env.close()

    return {
        "rewards":        np.array(model.episode_reward_history, dtype=np.float32),
        "mean_reward":    mean_r,
        "std_reward":     std_r,
        "concept_metrics": concept_metrics,          # dict: name → accuracy/mse
        "task_types":     single_env.task_types,
        "concept_names":  single_env.concept_names,
        "temporal_concepts": getattr(single_env, "temporal_concepts", []),
    }


def _evaluate_concepts(model: PPO, single_env) -> Dict[str, float]:
    """
    Roll out model for ~200 steps in single env and compute per-concept metrics.
    Returns {concept_name: metric_value}.
    """
    if model.method == "no_concept":
        return {}

    concept_names = single_env.concept_names
    task_types    = single_env.task_types
    n_concepts    = len(concept_names)

    all_preds  = [[] for _ in range(n_concepts)]
    all_truths = [[] for _ in range(n_concepts)]

    model.policy.set_training_mode(False)
    obs, _ = single_env.reset()

    # Hidden state: only needed for GRU temporal encoding
    from ppo.networks import ConceptActorCritic
    use_gru = (
        getattr(model, "temporal_encoding", "none") == "gru"
        and model.method == "concept_actor_critic"
    )
    h_t = torch.zeros(1, ConceptActorCritic.HIDDEN_DIM, device=model.device) if use_gru else None

    for _ in range(200):
        obs_t = torch.as_tensor(
            obs if not isinstance(obs, dict) else {k: np.expand_dims(v, 0) for k, v in obs.items()},
            dtype=torch.float32,
        )
        if isinstance(obs_t, dict):
            obs_t = {k: v.to(model.device) for k, v in obs_t.items()}
        else:
            obs_t = obs_t.unsqueeze(0).to(model.device)

        with torch.no_grad():
            c_t, h_t = model.policy.predict_concepts(obs_t, h_t)

        truth = single_env.get_concept()

        if c_t is not None:
            c_np = c_t.cpu().numpy().flatten()
            for i in range(n_concepts):
                all_preds[i].append(c_np[i])
                all_truths[i].append(truth[i])

        action, h_t = model.policy.predict(obs_t, h_t, deterministic=True)
        obs, _, done, trunc, _ = single_env.step(action.cpu().numpy().item())
        if done or trunc:
            obs, _ = single_env.reset()
            h_t = (
                torch.zeros(1, ConceptActorCritic.HIDDEN_DIM, device=model.device)
                if use_gru else None
            )

    metrics = {}
    for i, (name, tt) in enumerate(zip(concept_names, task_types)):
        preds  = np.array(all_preds[i])
        truths = np.array(all_truths[i])
        if tt == "classification":
            metrics[name] = float(np.mean(np.round(preds).astype(int) == truths.astype(int)))
        else:
            metrics[name] = float(np.mean((preds - truths) ** 2))
    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_learning_curves(
    results: Dict[str, Dict[int, dict]],
    out_dir: str,
    window: int = 30,
) -> None:
    """
    results[method][seed] → {rewards: np.ndarray, ...}
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in METHODS:
        if method not in results:
            continue
        all_rewards = [v["rewards"] for v in results[method].values() if len(v["rewards"]) > 0]
        if not all_rewards:
            continue

        # Align to common length
        min_len = min(len(r) for r in all_rewards)
        arr = np.stack([r[:min_len] for r in all_rewards])  # [n_seeds, episodes]
        mean = smooth(arr.mean(axis=0), window)
        std  = arr.std(axis=0)

        x = np.arange(min_len)
        ax.plot(x, mean, label=METHOD_LABELS[method], color=METHOD_COLORS[method])
        ax.fill_between(
            x,
            smooth(mean - std, window),
            smooth(mean + std, window),
            alpha=0.2,
            color=METHOD_COLORS[method],
        )

    ax.set_xlabel("Episode", fontsize=13)
    ax.set_ylabel("Reward", fontsize=13)
    ax.set_title("Learning Curves (mean ± std across seeds)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "learning_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[compare] saved → {path}")


def plot_concept_accuracy(
    results: Dict[str, Dict[int, dict]],
    out_dir: str,
) -> None:
    """Bar chart of per-concept accuracy, split by static vs temporal."""
    # Collect concept names from first available result
    concept_names = None
    task_types    = None
    temporal_idx  = []
    for method in METHODS:
        if method in results:
            for seed_res in results[method].values():
                if seed_res.get("concept_metrics"):
                    concept_names = seed_res["concept_names"]
                    task_types    = seed_res["task_types"]
                    temporal_idx  = seed_res["temporal_concepts"]
                    break
        if concept_names:
            break

    if not concept_names:
        return  # no_concept only run

    n = len(concept_names)
    concept_methods = [m for m in METHODS if m != "no_concept"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    static_idx   = [i for i in range(n) if i not in temporal_idx]
    temporal_idx_ = list(temporal_idx)

    for ax, indices, title in [
        (axes[0], static_idx,   "Static Concepts"),
        (axes[1], temporal_idx_, "Temporal Concepts"),
    ]:
        x    = np.arange(len(indices))
        wid  = 0.35 if len(concept_methods) == 2 else 0.25
        offsets = np.linspace(-(len(concept_methods)-1)*wid/2,
                               (len(concept_methods)-1)*wid/2,
                               len(concept_methods))

        for j, method in enumerate(concept_methods):
            if method not in results:
                continue
            mean_metrics = []
            for ci in indices:
                name = concept_names[ci]
                vals = [
                    v["concept_metrics"].get(name, np.nan)
                    for v in results[method].values()
                    if v.get("concept_metrics")
                ]
                mean_metrics.append(np.nanmean(vals) if vals else 0.0)

            ax.bar(x + offsets[j], mean_metrics, wid,
                   label=METHOD_LABELS[method], color=METHOD_COLORS[method], alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([concept_names[i] for i in indices], rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Accuracy (cls) / -MSE (reg)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Concept Accuracy by Type", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, "concept_accuracy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[compare] saved → {path}")


def write_summary_table(
    results: Dict[str, Dict[int, dict]],
    out_dir: str,
) -> None:
    lines = ["=" * 60]
    lines.append(f"{'Method':<30}  {'Mean Reward':>12}  {'Std Reward':>10}")
    lines.append("-" * 60)

    for method in METHODS:
        if method not in results:
            continue
        seed_means = [v["mean_reward"] for v in results[method].values()]
        m = np.mean(seed_means)
        s = np.std(seed_means)
        lines.append(f"{METHOD_LABELS[method]:<30}  {m:>12.2f}  {s:>10.2f}")

    lines.append("=" * 60)
    table = "\n".join(lines)
    print("\n" + table)

    path = os.path.join(out_dir, "summary_table.txt")
    with open(path, "w") as f:
        f.write(table + "\n")
    print(f"[compare] saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare three RL methods on a single environment.")
    parser.add_argument("--env",    required=True,
                        choices=["cartpole", "dynamic_obstacles", "lunar_lander"])
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        choices=METHODS, help="Subset of methods to compare")
    parser.add_argument("--temporal_encoding", type=str, default="none",
                        choices=["gru", "stacked", "none"],
                        help="Temporal encoding for concept_actor_critic")
    parser.add_argument("--training_mode", type=str, default="two_phase",
                        choices=["two_phase", "end_to_end"],
                        help="'two_phase': concept net frozen during PPO; 'end_to_end': joint training")
    parser.add_argument("--seeds",   nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--num_labels",      type=int, default=500)
    parser.add_argument("--query_num_times", type=int, default=2)
    parser.add_argument("--n_envs",          type=int, default=4)
    parser.add_argument("--n_steps",         type=int, default=512)
    parser.add_argument("--batch_size",      type=int, default=256)
    parser.add_argument("--device",          type=str, default="auto")
    parser.add_argument("--output_dir",      type=str, default="compare_results")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        args.output_dir,
        f"{args.env}_{args.training_mode}_{args.temporal_encoding}_{timestamp}"
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[compare] results → {out_dir}")

    # Save config
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---- Run all (method, seed) combinations ----
    results: Dict[str, Dict[int, dict]] = {}

    for method in args.methods:
        results[method] = {}
        for seed in args.seeds:
            print(f"\n{'='*60}")
            print(f"[compare] method={method}  seed={seed}")
            print(f"{'='*60}")
            t0 = time.time()
            try:
                res = run_single(
                    method             = method,
                    env_name           = args.env,
                    seed               = seed,
                    total_timesteps    = args.total_timesteps,
                    num_labels         = args.num_labels,
                    query_num_times    = args.query_num_times,
                    n_envs             = args.n_envs,
                    n_steps            = args.n_steps,
                    batch_size         = args.batch_size,
                    device             = args.device,
                    temporal_encoding  = args.temporal_encoding,
                    training_mode      = args.training_mode,
                )
                results[method][seed] = res
                elapsed = time.time() - t0
                print(
                    f"[compare] done in {elapsed:.0f}s  "
                    f"mean_reward={res['mean_reward']:.2f} ± {res['std_reward']:.2f}"
                )
                # Save per-run rewards
                np.save(
                    os.path.join(out_dir, f"{method}_seed{seed}_rewards.npy"),
                    res["rewards"],
                )
            except Exception as e:
                print(f"[compare] ERROR in {method} seed={seed}: {e}")
                import traceback; traceback.print_exc()
                results[method][seed] = {"rewards": np.array([]), "mean_reward": 0.0,
                                          "std_reward": 0.0, "concept_metrics": {},
                                          "concept_names": [], "task_types": [],
                                          "temporal_concepts": []}

    # ---- Plots ----
    print("\n[compare] generating plots...")
    plot_learning_curves(results, out_dir)
    plot_concept_accuracy(results, out_dir)
    write_summary_table(results, out_dir)

    print(f"\n[compare] all done. Results in {out_dir}")


if __name__ == "__main__":
    main()
