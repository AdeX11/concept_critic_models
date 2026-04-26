"""
replay.py — Render saved models acting in an environment and save annotated GIFs.

Example:
  python replay.py \
      --env pick_place \
      --method concept_actor_critic \
      --model_path results/.../model.pt \
      --temporal_encoding none \
      --output_gif ./replays/concept_actor_critic.gif
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.cartpole import make_single_cartpole_env
from envs.dynamic_obstacles import make_single_dynamic_obstacles_env
from envs.lunar_lander import (
    make_single_lunar_lander_env,
    make_single_lunar_lander_state_env,
    make_single_lunar_lander_pos_only_env,
    make_single_lunar_lander_pos_only_concept_env,
)
from envs.mountain_car import make_single_mountain_car_env
from envs.pick_place import (
    make_single_panda_env,
    make_single_panda_pickplace_state_env,
)
from envs.highway import make_single_highway_env, make_single_highway_state_env
from ppo.networks import ConceptActorCritic
from ppo.policy import ActorCriticPolicy


def get_obs_shape(env):
    obs_space = env.observation_space
    if hasattr(obs_space, "spaces"):
        return {k: v.shape for k, v in obs_space.spaces.items()}
    return obs_space.shape


def _make_single_env(env_name: str, seed: int = 0):
    """Factory that mirrors train.py make_env_and_policy_kwargs."""
    if env_name in ("pick_place", "panda_pickplace"):
        return make_single_panda_env(seed=seed)
    elif env_name == "panda_pickplace_state":
        return make_single_panda_pickplace_state_env(seed=seed)
    elif env_name == "cartpole":
        return make_single_cartpole_env(seed=seed)
    elif env_name == "dynamic_obstacles":
        return make_single_dynamic_obstacles_env(seed=seed)
    elif env_name == "lunar_lander":
        return make_single_lunar_lander_env(seed=seed)
    elif env_name == "lunar_lander_state":
        return make_single_lunar_lander_state_env(seed=seed)
    elif env_name == "lunar_lander_pos_only":
        return make_single_lunar_lander_pos_only_env(seed=seed)
    elif env_name == "lunar_lander_pos_only_concept":
        return make_single_lunar_lander_pos_only_concept_env(seed=seed)
    elif env_name == "mountain_car":
        return make_single_mountain_car_env(seed=seed)
    elif env_name == "highway":
        return make_single_highway_env(seed=seed)
    elif env_name == "highway_state":
        return make_single_highway_state_env(seed=seed)
    else:
        raise ValueError(f"Unknown env: {env_name}")


def obs_to_tensor(obs, device: torch.device):
    if isinstance(obs, dict):
        return {
            k: torch.as_tensor(np.expand_dims(v, 0), dtype=torch.float32, device=device)
            for k, v in obs.items()
        }
    return torch.as_tensor(np.expand_dims(obs, 0), dtype=torch.float32, device=device)


def render_frame(env) -> np.ndarray:
    """Render current frame from env.render()."""
    rendered = env.render()
    if rendered is not None:
        arr = np.asarray(rendered, dtype=np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 3:
            return arr
    return np.zeros((84, 84, 3), dtype=np.uint8)


def predict_step(
    policy: ActorCriticPolicy,
    obs_t,
    h_t: Optional[torch.Tensor],
    deterministic: bool,
) -> Tuple[int, Optional[np.ndarray], Optional[torch.Tensor]]:
    with torch.no_grad():
        features = policy.extract_features(obs_t)
        latent, h_new, c_t, _ = policy._get_latent(features, h_t)
        action_logits = policy.action_net(latent)
        if deterministic:
            action_t = action_logits.argmax(dim=1)
        else:
            action_t = torch.distributions.Categorical(logits=action_logits).sample()

    pred_concepts = None
    if c_t is not None:
        if policy.method == "concept_actor_critic" and hasattr(
            policy.concept_net, "decode_concept_vector"
        ):
            c_t = policy.concept_net.decode_concept_vector(c_t)
        elif policy.method == "gvf":
            n_concepts = len(policy.task_types)
            c_t = c_t[:, :n_concepts]
        pred_concepts = c_t.squeeze(0).cpu().numpy()

    return int(action_t.item()), pred_concepts, h_new


def build_policy(args, single_env, device: torch.device) -> ActorCriticPolicy:
    gvf_pairing = args.gvf_pairing if args.method == "gvf" else []
    policy = ActorCriticPolicy(
        obs_shape=get_obs_shape(single_env),
        n_actions=single_env.action_space.n,
        method=args.method,
        task_types=single_env.task_types,
        num_classes=single_env.num_classes,
        gvf_pairing=gvf_pairing,
        concept_dim=len(single_env.task_types),
        temporal_encoding=args.temporal_encoding,
        features_dim=512,
        net_arch=[64, 64],
        device=device,
    ).to(device)
    state = torch.load(args.model_path, map_location=device)
    policy.load_state_dict(state)
    policy.set_training_mode(False)
    return policy


def action_label(env, action: int) -> str:
    enum_obj = getattr(getattr(env, "unwrapped", env), "actions", None)
    if enum_obj is not None:
        try:
            return f"{action}:{enum_obj(action).name}"
        except Exception:
            pass
    return str(action)


def format_concepts(
    names: List[str], truth: Optional[np.ndarray], pred: Optional[np.ndarray]
) -> List[str]:
    lines: List[str] = []
    if truth is None and pred is None:
        return lines
    count = 0
    for i, name in enumerate(names):
        if pred is not None and truth is not None:
            lines.append(f"{name}: p={pred[i]:.0f} t={truth[i]:.0f}")
        elif pred is not None:
            lines.append(f"{name}: p={pred[i]:.0f}")
        else:
            lines.append(f"{name}: t={truth[i]:.0f}")
        count += 1
        if count >= 6:
            break
    return lines


def annotate_frame(
    frame: np.ndarray,
    header_lines: List[str],
    concept_lines: List[str],
) -> Image.Image:
    image = Image.fromarray(frame)
    font = ImageFont.load_default()
    padding = 8
    line_h = 14
    overlay_h = padding * 2 + line_h * (len(header_lines) + len(concept_lines))
    canvas = Image.new("RGB", (image.width, image.height + overlay_h), color=(18, 18, 18))
    canvas.paste(image, (0, 0))

    draw = ImageDraw.Draw(canvas)
    y = image.height + padding
    for line in header_lines + concept_lines:
        draw.text((padding, y), line, fill=(235, 235, 235), font=font)
        y += line_h
    return canvas


def save_gif(frames: List[Image.Image], path: str, fps: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    duration_ms = int(1000 / max(fps, 1))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def replay(args) -> None:
    device = torch.device(args.device)
    env = _make_single_env(args.env, args.seed)
    policy = build_policy(args, env, device)

    frames: List[Image.Image] = []
    _uses_gru = (
        args.temporal_encoding == "gru"
        and args.method in ("concept_actor_critic", "gvf")
    )
    h_t = (
        torch.zeros(1, ConceptActorCritic.HIDDEN_DIM, device=device)
        if _uses_gru
        else None
    )

    obs, _ = env.reset(seed=args.seed)
    total_reward = 0.0
    step_idx = 0
    episode_idx = 0

    while episode_idx < args.episodes and step_idx < args.max_steps:
        obs_t = obs_to_tensor(obs, device)
        action, pred_concepts, h_new = predict_step(
            policy, obs_t, h_t, deterministic=args.deterministic
        )

        if hasattr(env, "get_concept") and callable(getattr(env, "get_concept", None)):
            try:
                truth = env.get_concept()
            except Exception:
                truth = None
        elif hasattr(env, "current_concept"):
            truth = env.current_concept.copy()
        else:
            truth = None
        frame = render_frame(env)
        header_lines = [
            f"method={args.method} env={args.env}",
            f"episode={episode_idx + 1}/{args.episodes} step={step_idx} action={action_label(env, action)}",
            f"reward_so_far={total_reward:.2f}",
        ]
        concept_lines = (
            format_concepts(env.concept_names, truth, pred_concepts)
            if args.show_concepts
            else []
        )
        frames.append(annotate_frame(frame, header_lines, concept_lines))

        obs, reward, done, truncated, _ = env.step(action)
        total_reward += float(reward)
        step_idx += 1

        if _uses_gru:
            h_t = h_new
            if done or truncated:
                h_t = torch.zeros_like(h_t)

        if done or truncated:
            episode_idx += 1
            if episode_idx < args.episodes:
                obs, _ = env.reset()
                total_reward = 0.0

    if not frames:
        raise RuntimeError("No frames captured during replay")

    save_gif(frames, args.output_gif, args.fps)
    env.close()
    print(f"[replay] saved {len(frames)} frames -> {args.output_gif}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved model and save an annotated GIF.")
    parser.add_argument("--env", required=True)
    parser.add_argument("--method", required=True, choices=[
        "no_concept", "vanilla_freeze", "concept_actor_critic", "gvf",
    ])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_gif", required=True)
    parser.add_argument("--temporal_encoding", default="none", choices=["gru", "stacked", "none"])
    parser.add_argument(
        "--gvf_pairing",
        type=int,
        nargs="*",
        default=[],
        help="Concept indices paired with GVF heads (required when method=gvf).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--show_concepts", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    replay(args)


if __name__ == "__main__":
    main()
