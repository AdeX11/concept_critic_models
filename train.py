"""
train.py — Unified training entry point.

Usage:
  python train.py --method no_concept           --env cartpole           --seed 42
  python train.py --method vanilla_freeze        --env dynamic_obstacles  --seed 42
  python train.py --method concept_actor_critic  --env lunar_lander       --seed 42

All results are saved to results/<method>_<env>_seed<seed>/ as:
  - rewards.npy       — episode reward history
  - concept_acc.npy   — concept accuracy log (if applicable)
  - model.pt          — saved policy state_dict
"""

import argparse
import os
import sys
import random
import numpy as np
import torch

# ---------------------------------------------------------------------------
# allow running from parent dir or research/ dir
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_registry import get_benchmark_spec, list_benchmark_ids
from envs.registry import list_env_names, make_env_pair
from ppo.ppo              import PPO
from runtime_utils import get_obs_shape, make_summary_writer, write_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env_and_policy_kwargs(env_name: str, n_envs: int, seed: int, temporal_encoding: str = "none"):
    vec_env, single_env, _ = make_env_pair(env_name, n_envs, seed, temporal_encoding=temporal_encoding)
    policy_kwargs = dict(
        obs_shape     = get_obs_shape(single_env),
        n_actions     = vec_env.single_action_space.n,
        task_types    = single_env.task_types,
        num_classes   = single_env.num_classes,
        concept_dim   = len(single_env.task_types),
        concept_names = single_env.concept_names,
        features_dim  = 512,
        net_arch      = [64, 64],
    )
    return vec_env, single_env, policy_kwargs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL agent with optional concept bottleneck.")
    parser.add_argument("--method", required=True,
                        choices=["no_concept", "vanilla_freeze", "concept_actor_critic", "gvf"])
    parser.add_argument("--benchmark", default=None, choices=list_benchmark_ids())
    parser.add_argument("--env", default=None, choices=list_env_names())
    parser.add_argument("--temporal_encoding", type=str, default="none",
                        choices=["gru", "stacked", "none"],
                        help="Temporal encoding for concept_actor_critic: "
                             "'gru' (GRUCell in network), 'stacked' (env-level frame stack), "
                             "'none' (no temporal info, ablation)")
    parser.add_argument("--training_mode", type=str, default="two_phase",
                        choices=["two_phase", "end_to_end", "joint"],
                        help="'two_phase': concept net frozen during PPO update (LICORICE-style); "
                             "'end_to_end': policy gradient flows through concept net; "
                             "'joint': end_to_end plus per-iteration supervised rollout concept updates")
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--total_timesteps",   type=int,   default=None)
    parser.add_argument("--num_labels",        type=int,   default=None,
                        help="Total labeled samples across all queries")
    parser.add_argument("--query_num_times",   type=int,   default=None,
                        help="How many times to query labels during training")
    parser.add_argument("--gvf_pairing", type=str, default=None,
                        help="Comma-separated 0-based concept indices for GVF heads; defaults to all concepts")
    parser.add_argument("--n_envs",            type=int,   default=4)
    parser.add_argument("--n_steps",           type=int,   default=512)
    parser.add_argument("--n_epochs",          type=int,   default=10)
    parser.add_argument("--batch_size",        type=int,   default=256)
    parser.add_argument("--learning_rate",     type=float, default=3e-4)
    parser.add_argument("--ent_coef",          type=float, default=0.01)
    parser.add_argument("--vf_coef",           type=float, default=0.5)
    parser.add_argument("--lambda_v",          type=float, default=0.5,
                        help="Concept critic loss weight (concept_actor_critic only)")
    parser.add_argument("--lambda_s",          type=float, default=0.5,
                        help="Supervised anchor loss weight (concept_actor_critic only)")
    parser.add_argument("--device",            type=str,   default="auto")
    parser.add_argument("--output_dir",        type=str,   default="results")
    parser.add_argument("--resume_from",       type=str,   default=None)
    parser.add_argument("--checkpoint_every_timesteps", type=int, default=50_000)
    parser.add_argument("--eval_every_timesteps", type=int, default=50_000)
    parser.add_argument("--eval_episodes",     type=int,   default=20)
    args = parser.parse_args()

    benchmark_id = args.benchmark or args.env
    if benchmark_id is None:
        raise ValueError("One of --benchmark or --env is required")
    benchmark_spec = get_benchmark_spec(benchmark_id) if args.benchmark else None
    env_name = benchmark_spec.env_name if benchmark_spec is not None else benchmark_id

    if args.total_timesteps is None:
        args.total_timesteps = benchmark_spec.canonical_total_timesteps if benchmark_spec is not None else 1_000_000
    if args.num_labels is None:
        args.num_labels = benchmark_spec.canonical_num_labels if benchmark_spec is not None else 500
    if args.query_num_times is None:
        args.query_num_times = benchmark_spec.canonical_query_num_times if benchmark_spec is not None else 1
    gvf_pairing = None
    if args.gvf_pairing:
        gvf_pairing = [
            int(part.strip())
            for part in args.gvf_pairing.split(",")
            if part.strip()
        ]

    set_seed(args.seed)

    out_dir = os.path.join(
        args.output_dir,
        f"{args.method}_{args.training_mode}_{args.temporal_encoding}_{benchmark_id}_seed{args.seed}"
    )
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_dir = os.path.join(out_dir, "checkpoints")
    writer = make_summary_writer(os.path.join(out_dir, "tb"))
    run_metadata = {
        "benchmark_id": benchmark_id,
        "env_name": env_name,
        "seed": args.seed,
        "method": args.method,
        "temporal_encoding": args.temporal_encoding,
        "training_mode": args.training_mode,
        "total_timesteps": args.total_timesteps,
        "num_labels": args.num_labels,
        "query_num_times": args.query_num_times,
        "gvf_pairing": gvf_pairing,
        "n_envs": args.n_envs,
        "n_steps": args.n_steps,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "lambda_v": args.lambda_v,
        "lambda_s": args.lambda_s,
    }
    write_json(os.path.join(out_dir, "metadata.json"), run_metadata)
    print(f"[train] method={args.method}  training_mode={args.training_mode}  "
          f"temporal_encoding={args.temporal_encoding}  benchmark={benchmark_id}  seed={args.seed}")
    print(f"[train] output → {out_dir}")

    # ---- Environment ----
    vec_env, single_env, policy_kwargs = make_env_and_policy_kwargs(
        env_name, args.n_envs, args.seed, temporal_encoding=args.temporal_encoding
    )
    policy_kwargs["device"] = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    policy_kwargs["temporal_encoding"] = args.temporal_encoding
    if args.method == "gvf" and gvf_pairing is not None:
        policy_kwargs["gvf_pairing"] = gvf_pairing

    # ---- PPO ----
    model = PPO(
        method         = args.method,
        env            = vec_env,
        policy_kwargs  = policy_kwargs,
        n_steps        = args.n_steps,
        n_epochs       = args.n_epochs,
        batch_size     = args.batch_size,
        gamma          = 0.99,
        gae_lambda     = 0.95,
        clip_range     = 0.2,
        ent_coef       = args.ent_coef,
        vf_coef        = args.vf_coef,
        max_grad_norm  = 0.5,
        learning_rate  = args.learning_rate,
        lambda_v       = args.lambda_v,
        lambda_s       = args.lambda_s,
        training_mode  = args.training_mode,
        normalize_advantage = True,
        seed           = args.seed,
        device         = args.device,
        verbose        = 1,
        eval_env       = single_env,
        writer         = writer,
        benchmark_spec = benchmark_spec,
    )

    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=model.device)
        model.load_checkpoint_state(checkpoint)

    labels_per_query = max(1, args.num_labels // max(args.query_num_times, 1))

    model.learn(
        total_timesteps   = args.total_timesteps,
        query_num_times   = args.query_num_times if args.method != "no_concept" else 0,
        query_labels_per_time = labels_per_query,
        eval_every_timesteps = args.eval_every_timesteps,
        eval_n_episodes      = args.eval_episodes,
        checkpoint_dir       = checkpoint_dir,
        checkpoint_every_timesteps = args.checkpoint_every_timesteps,
        run_metadata         = run_metadata,
    )

    # ---- Save ----
    rewards_path = os.path.join(out_dir, "rewards.npy")
    np.save(rewards_path, np.array(model.episode_reward_history, dtype=np.float32))
    print(f"[train] saved episode rewards → {rewards_path}")

    model_path = os.path.join(out_dir, "model.pt")
    torch.save(model.policy.state_dict(), model_path)
    print(f"[train] saved model → {model_path}")

    if model.concept_acc_log:
        timesteps = np.array([t for t, _ in model.concept_acc_log], dtype=np.int64)
        names     = list(model.concept_acc_log[0][1].keys())
        values    = np.array([[d[n] for n in names]
                               for _, d in model.concept_acc_log], dtype=np.float32)
        np.savez(os.path.join(out_dir, "concept_acc.npz"),
                 timesteps=timesteps, names=np.array(names), values=values)
        print(f"[train] saved concept accuracy log → {out_dir}/concept_acc.npz")

    # ---- Quick evaluation ----
    final_eval = model.evaluate_detailed(n_episodes=args.eval_episodes, deterministic=True)
    mean_r, std_r = final_eval["mean_reward"], final_eval["std_reward"]
    print(f"[train] eval: mean_reward={mean_r:.2f} ± {std_r:.2f}")

    # Save eval result
    with open(os.path.join(out_dir, "eval.txt"), "w") as f:
        f.write(f"mean_reward={mean_r:.4f}\nstd_reward={std_r:.4f}\n")
        if final_eval.get("success_rate") is not None:
            f.write(f"success_rate={final_eval['success_rate']:.4f}\n")
        if final_eval.get("normalized_return") is not None:
            f.write(f"normalized_return={final_eval['normalized_return']:.4f}\n")
    write_json(os.path.join(out_dir, "eval.json"), final_eval)
    if model.eval_log:
        write_json(
            os.path.join(out_dir, "eval_checkpoints.json"),
            {"eval_log": [{"timesteps": t, "metrics": m} for t, m in model.eval_log]},
        )

    vec_env.close()
    single_env.close()
    writer.close()
    print("[train] done.")


if __name__ == "__main__":
    main()
