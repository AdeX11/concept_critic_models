#!/usr/bin/env python3
"""
Concept steerability evaluation for TMaze runs.

This evaluator uses the current registry/checkpoint format. For each episode, it
walks the corridor normally until the junction, then queries the same policy
state under three concept-bottleneck conditions:

    baseline: normal inferred concepts
    correct:  cue concept forced to the ground-truth cue
    flipped:  cue concept forced to the opposite cue

The trajectory is closed with the baseline action, so the three action queries
share the same hidden state and observation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from envs.registry import make_single_env
from ppo.policy import ActorCriticPolicy
from runtime_utils import get_obs_shape, write_json


def _load_metadata(run_dir: Path) -> Dict[str, object]:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {run_dir}")
    return json.loads(metadata_path.read_text())


def _load_policy(run_dir: Path, device: torch.device) -> tuple[ActorCriticPolicy, object, Dict[str, object]]:
    metadata = _load_metadata(run_dir)
    benchmark_id = str(metadata.get("benchmark_id") or "tmaze")
    env_name = str(metadata.get("env_name") or benchmark_id)
    temporal_encoding = str(metadata.get("temporal_encoding") or "none")
    method = str(metadata.get("method") or "concept_actor_critic")
    seed = int(metadata.get("seed") or 0)

    env = make_single_env(env_name, seed=seed, temporal_encoding=temporal_encoding)
    policy_kwargs = dict(
        obs_shape=get_obs_shape(env),
        n_actions=env.action_space.n,
        method=method,
        task_types=env.task_types,
        num_classes=env.num_classes,
        concept_dim=len(env.task_types),
        temporal_encoding=temporal_encoding,
        features_dim=int(metadata.get("features_dim") or 512),
        net_arch=[64, 64],
        device=str(device),
    )
    gvf_pairing = metadata.get("gvf_pairing")
    if method == "gvf" and gvf_pairing is not None:
        policy_kwargs["gvf_pairing"] = [int(idx) for idx in gvf_pairing]

    policy = ActorCriticPolicy(**policy_kwargs).to(device)
    model_path = run_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.pt in {run_dir}")
    state = torch.load(model_path, map_location=device)
    policy.load_state_dict(state)
    policy.eval()
    return policy, env, metadata


def _last_tmaze_frame(obs: np.ndarray) -> np.ndarray:
    flat = np.asarray(obs, dtype=np.float32).reshape(-1)
    if flat.size < 4:
        raise ValueError(f"TMaze observation must have at least 4 values, got shape {np.shape(obs)}")
    return flat[-4:]


def _obs_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.expand_dims(obs, 0), dtype=torch.float32, device=device)


def _junction_reward(action: int, cue: int, n_steps: int) -> float:
    time_penalty = -0.01 * n_steps
    if action == 1:
        outcome = 1.0 if cue == 0 else -1.0
    elif action == 2:
        outcome = 1.0 if cue == 1 else -1.0
    else:
        outcome = -1.0
    return float(outcome + time_penalty)


def evaluate_steerability(
    policy: ActorCriticPolicy,
    env,
    *,
    n_episodes: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Optional[float]]:
    concept_dim = len(env.task_types)
    has_concepts = policy.method != "no_concept"

    rewards_base = []
    rewards_correct = []
    rewards_flipped = []
    correct_changed = []
    flip_changed = []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)
        cue = int(np.asarray(info["concept"])[0])
        h_t = None
        done = False
        steps = 0

        while not done:
            frame = _last_tmaze_frame(obs)
            at_junction = bool(frame[3] > 0.5)
            obs_t = _obs_tensor(obs, device)
            steps += 1

            if at_junction:
                with torch.no_grad():
                    action_base, h_new = policy.predict(obs_t, h_t, deterministic=True)
                    a_base = int(action_base.item())

                    if has_concepts:
                        correct = torch.zeros(1, concept_dim, device=device)
                        correct[0, 0] = float(cue)
                        correct[0, 1] = 1.0
                        flipped = correct.clone()
                        flipped[0, 0] = float(1 - cue)

                        action_correct, _ = policy.predict(
                            obs_t,
                            h_t,
                            deterministic=True,
                            concept_override=correct,
                        )
                        action_flipped, _ = policy.predict(
                            obs_t,
                            h_t,
                            deterministic=True,
                            concept_override=flipped,
                        )
                        a_correct = int(action_correct.item())
                        a_flipped = int(action_flipped.item())

                        rewards_correct.append(_junction_reward(a_correct, cue, steps))
                        rewards_flipped.append(_junction_reward(a_flipped, cue, steps))
                        correct_changed.append(float(a_correct != a_base))
                        flip_changed.append(float(a_flipped != a_base))

                rewards_base.append(_junction_reward(a_base, cue, steps))
                obs, _, terminated, truncated, _ = env.step(a_base)
                h_t = h_new
                done = bool(terminated or truncated)
            else:
                with torch.no_grad():
                    action, h_t = policy.predict(obs_t, h_t, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(int(action.item()))
                done = bool(terminated or truncated)

    reward_baseline = float(np.mean(rewards_base)) if rewards_base else None
    if reward_baseline is None:
        return {
            "reward_baseline": None,
            "reward_correct": None,
            "reward_flipped": None,
            "correct_change_rate": None,
            "flip_change_rate": None,
            "steerability_score": None,
            "causal_sensitivity": None,
            "n_episodes": n_episodes,
        }
    if not has_concepts:
        return {
            "reward_baseline": reward_baseline,
            "reward_correct": None,
            "reward_flipped": None,
            "correct_change_rate": None,
            "flip_change_rate": None,
            "steerability_score": None,
            "causal_sensitivity": None,
            "n_episodes": n_episodes,
        }

    reward_correct = float(np.mean(rewards_correct))
    reward_flipped = float(np.mean(rewards_flipped))
    max_reward = 0.89
    denom = max_reward - float(reward_baseline)
    steerability_score = 0.0 if abs(denom) < 1e-8 else (reward_correct - float(reward_baseline)) / denom
    return {
        "reward_baseline": reward_baseline,
        "reward_correct": reward_correct,
        "reward_flipped": reward_flipped,
        "correct_change_rate": float(np.mean(correct_changed)),
        "flip_change_rate": float(np.mean(flip_changed)),
        "steerability_score": float(np.clip(steerability_score, -1.0, 1.0)),
        "causal_sensitivity": float(float(reward_baseline) - reward_flipped),
        "n_episodes": n_episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TMaze concept steerability evaluation.")
    parser.add_argument("--run_dir", required=True, help="Directory containing metadata.json and model.pt")
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device(args.device)
    policy, env, metadata = _load_policy(run_dir, device)
    try:
        if str(metadata.get("env_name") or metadata.get("benchmark_id")) != "tmaze":
            raise ValueError("steerability_eval.py currently supports only the tmaze benchmark")
        metrics = evaluate_steerability(
            policy,
            env,
            n_episodes=args.n_episodes,
            seed=args.seed,
            device=device,
        )
    finally:
        env.close()

    payload = {"run_dir": str(run_dir), "metadata": metadata, "steerability": metrics}
    out_path = Path(args.out) if args.out else run_dir / "steerability.json"
    write_json(out_path, payload)
    print(json.dumps(payload["steerability"], indent=2))
    print(f"[steerability] wrote {out_path}")


if __name__ == "__main__":
    main()
