"""
run_interp.py — Run the full interpretability suite on a trained concept model.

Usage:
  python experiments/run_interp.py \
      --env cartpole \
      --method concept_actor_critic \
      --model_path results/concept_actor_critic_joint_gru_cartpole_seed42/model.pt \
      --temporal_encoding gru \
      --seed 42 \
      --output_dir interp_results/cartpole_seed42 \
      --plots_dir interp_plots/cartpole_seed42

All three analyses (ablation, attribution, causal graph) are run and saved.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import torch

from benchmark_registry import get_benchmark_spec, list_benchmark_ids
from envs.registry import make_single_env
from interpretability import (
    ablation_sweep,
    attribute_action_logits,
    concept_sensitivity_matrix,
    counterfactual_replay,
    discover_concept_dependencies,
    plot_ablation_heatmap,
    plot_attribution_bars,
    plot_causal_graph,
    run_with_intervention,
)
from interpretability.intervention import _obs_to_tensor
from ppo.policy import ActorCriticPolicy
from runtime_utils import get_obs_shape, write_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_policy(
    env,
    method: str,
    model_path: str,
    temporal_encoding: str,
    device: torch.device,
) -> ActorCriticPolicy:
    policy = ActorCriticPolicy(
        obs_shape=get_obs_shape(env),
        n_actions=env.action_space.n,
        method=method,
        task_types=env.task_types,
        num_classes=env.num_classes,
        concept_dim=len(env.task_types),
        temporal_encoding=temporal_encoding,
        features_dim=512,
        net_arch=[64, 64],
        device=device,
    ).to(device)
    state = torch.load(model_path, map_location=device)
    policy.load_state_dict(state)
    policy.set_training_mode(False)
    return policy


def env_factory(env_name: str, seed: int, temporal_encoding: str):
    def _make():
        return make_single_env(env_name, seed, temporal_encoding=temporal_encoding)
    return _make


# ---------------------------------------------------------------------------
# Analysis runners
# ---------------------------------------------------------------------------

def run_ablation(
    policy: ActorCriticPolicy,
    env_name: str,
    seed: int,
    temporal_encoding: str,
    output_dir: str,
    plots_dir: str,
) -> dict:
    print("  [ablation] Running zero-ablation sweep...")
    env = make_single_env(env_name, seed, temporal_encoding=temporal_encoding)
    concept_names = getattr(env, "concept_names", None)
    env.close()

    result = ablation_sweep(
        policy,
        env_factory(env_name, seed, temporal_encoding),
        n_episodes=5,
        seed=seed,
        max_steps=200,
        deterministic=True,
        device=torch.device("cpu"),
        temporal_encoding=temporal_encoding,
        concept_names=concept_names,
    )

    write_json(os.path.join(output_dir, "ablation.json"), result)

    fig = plot_ablation_heatmap(result, save_path=os.path.join(plots_dir, "ablation_importance.png"))
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("  [ablation] Done.")
    return result


def run_attribution(
    policy: ActorCriticPolicy,
    env_name: str,
    seed: int,
    temporal_encoding: str,
    output_dir: str,
    plots_dir: str,
) -> dict:
    print("  [attribution] Running concept → action sensitivity...")
    env = make_single_env(env_name, seed, temporal_encoding=temporal_encoding)

    result = concept_sensitivity_matrix(
        policy,
        env,
        n_steps=100,
        seed=seed,
        deterministic=True,
        device=torch.device("cpu"),
        temporal_encoding=temporal_encoding,
    )

    # Serialise only the key matrices for JSON
    serialisable = {
        "mean_sensitivity": result["mean_sensitivity"].tolist(),
        "std_sensitivity": result["std_sensitivity"].tolist(),
        "concept_names": result["concept_names"],
        "action_names": result["action_names"],
        "n_steps": result["n_steps"],
    }
    write_json(os.path.join(output_dir, "attribution.json"), serialisable)

    fig = plot_attribution_bars(result, save_path=os.path.join(plots_dir, "attribution_sensitivity.png"))
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("  [attribution] Done.")
    return result


def run_causal_graph(
    policy: ActorCriticPolicy,
    env_name: str,
    seed: int,
    temporal_encoding: str,
    output_dir: str,
    plots_dir: str,
) -> dict:
    print("  [causal_graph] Discovering inter-concept dependencies...")
    env = make_single_env(env_name, seed, temporal_encoding=temporal_encoding)
    concept_names = getattr(env, "concept_names", None)
    env.close()

    result = discover_concept_dependencies(
        policy,
        env_factory(env_name, seed, temporal_encoding),
        n_steps=100,
        seed=seed,
        device=torch.device("cpu"),
        temporal_encoding=temporal_encoding,
        concept_names=concept_names,
    )

    serialisable = {
        "adjacency_matrix": result["adjacency_matrix"].tolist(),
        "concept_names": result["concept_names"],
    }
    write_json(os.path.join(output_dir, "causal_graph.json"), serialisable)

    fig = plot_causal_graph(result, save_path=os.path.join(plots_dir, "causal_graph_heatmap.png"))
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("  [causal_graph] Done.")
    return result


def run_counterfactual_demo(
    policy: ActorCriticPolicy,
    env_name: str,
    seed: int,
    temporal_encoding: str,
    output_dir: str,
    plots_dir: str,
    concept_names: list,
) -> dict:
    """
    Run counterfactual replay for each concept that is relevant.
    Overrides each concept to 0.0 and records divergence steps.
    """
    print("  [counterfactual] Running counterfactual replay per concept...")
    env = make_single_env(env_name, seed, temporal_encoding=temporal_encoding)
    n_concepts = len(env.task_types)
    env.close()

    results = []
    for c_idx in range(n_concepts):
        c_name = concept_names[c_idx] if concept_names else f"concept_{c_idx}"
        result = counterfactual_replay(
            policy,
            make_single_env(env_name, seed, temporal_encoding=temporal_encoding),
            overrides={c_idx: 0.0},
            seed=seed,
            max_steps=200,
            deterministic=True,
            device=torch.device("cpu"),
            temporal_encoding=temporal_encoding,
            output_gif=os.path.join(plots_dir, f"counterfactual_{c_name}.gif"),
            fps=4,
        )
        summary = {
            "concept_idx": c_idx,
            "concept_name": c_name,
            "steps": result["steps"],
            "n_divergences": len(result["divergence_steps"]),
            "divergence_steps": result["divergence_steps"],
            "divergence_ratio": len(result["divergence_steps"]) / max(result["steps"], 1),
        }
        results.append(summary)
        print(f"    [{c_name}] {len(result['divergence_steps'])} divergences in {result['steps']} steps")

    write_json(os.path.join(output_dir, "counterfactual_summary.json"), results)
    print("  [counterfactual] Done.")
    return {"concepts": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run interpretability suite on a trained model")
    parser.add_argument("--benchmark", default=None, choices=list_benchmark_ids())
    parser.add_argument("--env", default=None)
    parser.add_argument("--method", required=True, choices=["vanilla_freeze", "concept_actor_critic"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--temporal_encoding", default="none", choices=["gru", "stacked", "none"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="interp_results")
    parser.add_argument("--plots_dir", default="interp_plots")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip_counterfactual", action="store_true", help="Skip slow counterfactual GIF generation")
    args = parser.parse_args()

    if args.benchmark:
        env_name = get_benchmark_spec(args.benchmark).env_name
    elif args.env:
        env_name = args.env
    else:
        raise ValueError("One of --benchmark or --env is required")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    device = torch.device(args.device)

    # Load policy once
    print(f"[run_interp] Loading model from {args.model_path}")
    env = make_single_env(env_name, args.seed, temporal_encoding=args.temporal_encoding)
    concept_names = list(getattr(env, "concept_names", [f"c{i}" for i in range(len(env.task_types))]))
    policy = load_policy(env, args.method, args.model_path, args.temporal_encoding, device)
    env.close()

    print(f"[run_interp] Running full interpretability suite on {env_name} (method={args.method}, seed={args.seed})")

    # 1. Ablation
    run_ablation(policy, env_name, args.seed, args.temporal_encoding, args.output_dir, args.plots_dir)

    # 2. Attribution
    run_attribution(policy, env_name, args.seed, args.temporal_encoding, args.output_dir, args.plots_dir)

    # 3. Causal graph
    run_causal_graph(policy, env_name, args.seed, args.temporal_encoding, args.output_dir, args.plots_dir)

    # 4. Counterfactual (optional — slow)
    if not args.skip_counterfactual and args.method == "concept_actor_critic":
        run_counterfactual_demo(
            policy, env_name, args.seed, args.temporal_encoding,
            args.output_dir, args.plots_dir, concept_names,
        )

    print(f"\n[run_interp] All analyses complete.")
    print(f"  JSON results → {args.output_dir}/")
    print(f"  Plots       → {args.plots_dir}/")


if __name__ == "__main__":
    main()