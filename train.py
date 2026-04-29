"""
train.py — Unified training entry point.

Usage:
  python train.py --concept_net none       --env cartpole          --seed 42
  python train.py --concept_net cbm        --env dynamic_obstacles --seed 42 --freeze_concept
  python train.py --concept_net concept_ac --env lunar_lander      --seed 42 --temporal gru

Parameters:
  --concept_net   : none | cbm | concept_ac   (what concept network architecture)
  --temporal      : none | stacked | gru       (temporal encoding)
  --supervision   : queried | online | none    (when supervised anchor fires)
  --freeze_concept: flag — freeze concept net during policy gradient update

Output: results/<concept_net>_<temporal>_<supervision>_<frozen|coupled>_<env>_seed<seed>/
  - rewards.npy      — episode reward history
  - concept_acc.npz  — concept accuracy log (if applicable)
  - model.pt         — saved policy state_dict
  - eval.txt         — final eval reward
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

from envs.cartpole        import make_cartpole_env,          make_single_cartpole_env
from envs.dynamic_obstacles import make_dynamic_obstacles_env, make_single_dynamic_obstacles_env
from envs.lunar_lander    import (make_lunar_lander_env,           make_single_lunar_lander_env,
                                   make_lunar_lander_state_env,      make_single_lunar_lander_state_env,
                                   make_lunar_lander_pos_only_env,   make_single_lunar_lander_pos_only_env)
from envs.mountain_car    import  make_mountain_car_env,             make_single_mountain_car_env
from envs.hidden_velocity import  make_hidden_velocity_env,          make_single_hidden_velocity_env
from envs.tmaze          import  make_tmaze_env,                     make_single_tmaze_env
from ppo.ppo              import PPO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_obs_shape(env):
    obs_space = env.observation_space
    if hasattr(obs_space, "spaces"):
        return {k: v.shape for k, v in obs_space.spaces.items()}
    return obs_space.shape


def make_env_and_policy_kwargs(env_name: str, n_envs: int, seed: int, n_stack: int = 4,
                               reward_mode: str = "dense"):
    """
    Returns (vec_env, single_env, policy_kwargs_base).
    policy_kwargs_base contains obs_shape, n_actions, task_types, num_classes, concept_dim.
    n_stack=4 for 'stacked' temporal encoding, n_stack=1 for 'gru' or 'none'.
    """
    if env_name == "cartpole":
        vec_env    = make_cartpole_env(n_envs, seed, n_stack=n_stack)
        single_env = make_single_cartpole_env(seed, n_stack=n_stack)
    elif env_name == "dynamic_obstacles":
        vec_env    = make_dynamic_obstacles_env(n_envs, seed, n_stack=n_stack)
        single_env = make_single_dynamic_obstacles_env(seed, n_stack=n_stack)
    elif env_name == "lunar_lander":
        vec_env    = make_lunar_lander_env(n_envs, seed, n_stack=n_stack)
        single_env = make_single_lunar_lander_env(seed, n_stack=n_stack)
    elif env_name == "lunar_lander_state":
        vec_env    = make_lunar_lander_state_env(n_envs, seed)
        single_env = make_single_lunar_lander_state_env(seed)
    elif env_name == "lunar_lander_pos_only":
        vec_env    = make_lunar_lander_pos_only_env(n_envs, seed)
        single_env = make_single_lunar_lander_pos_only_env(seed)
    elif env_name == "mountain_car":
        vec_env    = make_mountain_car_env(n_envs, seed)
        single_env = make_single_mountain_car_env(seed)
    elif env_name == "hidden_velocity":
        vec_env    = make_hidden_velocity_env(n_envs, seed)
        single_env = make_single_hidden_velocity_env(seed)
    elif env_name == "tmaze":
        vec_env    = make_tmaze_env(n_envs, seed, n_stack=n_stack, reward_mode=reward_mode)
        single_env = make_single_tmaze_env(seed, n_stack=n_stack, reward_mode=reward_mode)
    else:
        raise ValueError(f"Unknown env: {env_name}")

    # low-dim obs envs use a smaller feature extractor
    features_dim = 128 if env_name in ("hidden_velocity", "tmaze") else 512

    policy_kwargs = dict(
        obs_shape     = get_obs_shape(single_env),
        n_actions     = vec_env.single_action_space.n,
        task_types    = single_env.task_types,
        num_classes   = single_env.num_classes,
        concept_dim   = len(single_env.task_types),
        concept_names = single_env.concept_names,
        features_dim  = features_dim,
        net_arch      = [64, 64],
    )
    return vec_env, single_env, policy_kwargs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL agent with optional concept bottleneck.")
    parser.add_argument("--concept_net", required=True,
                        choices=["none", "cbm", "concept_ac"],
                        help="Concept network architecture: "
                             "'none' (plain PPO), "
                             "'cbm' (supervised concept bottleneck), "
                             "'concept_ac' (concept actor-critic)")
    parser.add_argument("--env",    required=True,
                        choices=["cartpole", "dynamic_obstacles", "lunar_lander",
                                 "lunar_lander_state", "lunar_lander_pos_only",
                                 "mountain_car", "hidden_velocity", "tmaze"])
    parser.add_argument("--temporal", type=str, default="none",
                        choices=["gru", "stacked", "none"],
                        help="Temporal encoding: "
                             "'gru' (GRUCell hidden state), "
                             "'stacked' (env-level frame stack), "
                             "'none' (single-frame, no memory)")
    parser.add_argument("--supervision", type=str, default="queried",
                        choices=["queried", "online", "none"],
                        help="When supervised anchor fires: "
                             "'queried' (only at explicit label query times), "
                             "'online' (every PPO iteration from rollout buffer)")
    parser.add_argument("--freeze_concept", action="store_true",
                        help="Freeze concept net during policy gradient update "
                             "(concept net updated only via supervision, not policy loss)")
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--total_timesteps",   type=int,   default=1_000_000)
    parser.add_argument("--num_labels",        type=int,   default=500,
                        help="Total labeled samples across all queries")
    parser.add_argument("--query_num_times",   type=int,   default=1,
                        help="How many times to query labels during training")
    parser.add_argument("--n_envs",            type=int,   default=4)
    parser.add_argument("--n_steps",           type=int,   default=512)
    parser.add_argument("--n_epochs",          type=int,   default=10)
    parser.add_argument("--batch_size",        type=int,   default=256)
    parser.add_argument("--learning_rate",     type=float, default=3e-4)
    parser.add_argument("--ent_coef",          type=float, default=0.01)
    parser.add_argument("--vf_coef",           type=float, default=0.5)
    parser.add_argument("--lambda_v",          type=float, default=0.5,
                        help="Concept critic loss weight (concept_ac only)")
    parser.add_argument("--lambda_s",          type=float, default=0.5,
                        help="Supervised anchor loss weight (concept_ac only)")
    parser.add_argument("--device",            type=str,   default="auto")
    parser.add_argument("--output_dir",        type=str,   default="/glade/derecho/scratch/adadelek/results")
    parser.add_argument("--reward_mode",       type=str,   default="dense",
                        choices=["dense", "sparse"],
                        help="Reward shaping: 'dense' (step penalties + ±1 at junction) "
                             "or 'sparse' (+1 correct at junction only)")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.concept_net == "none":
        run_tag = f"none_{args.env}_seed{args.seed}"
    else:
        freeze_str = "frozen" if args.freeze_concept else "coupled"
        run_tag = f"{args.concept_net}_{args.temporal}_{args.supervision}_{freeze_str}_{args.env}_seed{args.seed}"
    out_dir = os.path.join(args.output_dir, run_tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[train] concept_net={args.concept_net}  temporal={args.temporal}  "
          f"supervision={args.supervision}  freeze_concept={args.freeze_concept}  "
          f"env={args.env}  seed={args.seed}")
    print(f"[train] output → {out_dir}")

    # n_stack=4 for frame-stacking temporal encoding, else 1
    n_stack = 4 if args.temporal == "stacked" else 1

    # ---- Environment ----
    vec_env, single_env, policy_kwargs = make_env_and_policy_kwargs(
        args.env, args.n_envs, args.seed, n_stack=n_stack, reward_mode=args.reward_mode
    )
    policy_kwargs["device"] = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    policy_kwargs["temporal_encoding"] = args.temporal

    # ---- PPO ----
    model = PPO(
        concept_net    = args.concept_net,
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
        freeze_concept = args.freeze_concept,
        supervision    = args.supervision,
        normalize_advantage = True,
        seed           = args.seed,
        device         = args.device,
        verbose        = 1,
    )

    labels_per_query = max(1, args.num_labels // max(args.query_num_times, 1))

    model.learn(
        total_timesteps       = args.total_timesteps,
        query_num_times       = args.query_num_times if args.concept_net != "none" else 0,
        query_labels_per_time = labels_per_query,
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
    mean_r, std_r = model.evaluate(n_episodes=20, deterministic=True)
    print(f"[train] eval: mean_reward={mean_r:.2f} ± {std_r:.2f}")

    # Save eval result
    with open(os.path.join(out_dir, "eval.txt"), "w") as f:
        f.write(f"mean_reward={mean_r:.4f}\nstd_reward={std_r:.4f}\n")

    vec_env.close()
    single_env.close()
    print("[train] done.")


if __name__ == "__main__":
    main()
