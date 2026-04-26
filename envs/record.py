# ---------------------------------------------------------------------------
# Rollout Video Recording Utility
# ---------------------------------------------------------------------------

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def _obs_to_tensor(obs, device: torch.device):
    """Convert numpy obs (or dict of numpy) to batched tensor on device."""
    if isinstance(obs, dict):
        return {
            k: torch.as_tensor(np.expand_dims(v, 0), dtype=torch.float32, device=device)
            for k, v in obs.items()
        }
    return torch.as_tensor(np.expand_dims(obs, 0), dtype=torch.float32, device=device)


def _predict_step(
    policy,
    obs_t,
    h_t: Optional[torch.Tensor],
    deterministic: bool,
) -> Tuple[int, Optional[np.ndarray], Optional[torch.Tensor]]:
    """
    Single-step action selection matching the replay.py pattern.

    Returns
    -------
    action : int
    pred_concepts : np.ndarray or None
    h_new : torch.Tensor or None
    """
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
        pred_concepts = c_t.squeeze(0).cpu().numpy()

    return int(action_t.item()), pred_concepts, h_new


def _annotate_frame(
    frame: np.ndarray,
    header_lines: List[str],
    concept_lines: List[str],
) -> Image.Image:
    """Overlay text on frame using PIL (matches replay.py style)."""
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


def record_rollout_from_env(
    model,
    env,
    video_path: str = "videos/rollout.gif",
    max_steps: int = 300,
    fps: int = 30,
    deterministic: bool = True,
    overlay: bool = True,
    seed: Optional[int] = None,
):
    """
    Record a rollout GIF from a trained model acting in a single env.

    Parameters
    ----------
    model : PPO
        The trained PPO wrapper.  We call model.policy directly so this
        matches the replay.py pattern and does not require PPO.predict.
    env : gym.Env
        Single (non-vectorised) environment with render_mode='rgb_array'.
    video_path : str
        Output path (should end in .gif unless you have ffmpeg).
    max_steps : int
        Maximum frames to record per rollout.
    fps : int
        Playback frames per second.
    deterministic : bool
        Use argmax action if True, sample if False.
    overlay : bool
        Annotate frames with step info and concepts.
    seed : int, optional
        Seed passed to env.reset() for reproducibility.
    """
    policy = model.policy
    device = policy._device
    policy.set_training_mode(False)

    # Hidden state for GRU / temporal methods
    h_t = None
    if (
        policy.method == "concept_actor_critic"
        and getattr(policy, "temporal_encoding", None) == "gru"
    ):
        from ppo.networks import ConceptActorCritic

        h_t = torch.zeros(1, ConceptActorCritic.HIDDEN_DIM, device=device)

    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()

    frames: List[Image.Image] = []
    total_reward = 0.0

    for step in range(max_steps):
        obs_t = _obs_to_tensor(obs, device)
        action, pred_concepts, h_new = _predict_step(
            policy, obs_t, h_t, deterministic=deterministic
        )

        obs, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)

        # Render frame
        frame = env.render()
        if frame is None:
            raise RuntimeError(
                "render() returned None — make sure env uses render_mode='rgb_array'"
            )
        frame = np.asarray(frame, dtype=np.uint8)

        # Overlay
        if overlay:
            concept = info.get("concept", None)
            header_lines = [
                f"Step: {step}  Reward: {reward:.2f}  Total: {total_reward:.2f}",
            ]
            concept_lines = []
            if concept is not None:
                concept_lines = [
                    f"Risk: {concept[5]:.2f}  Broken: {int(concept[6])}  Speed: {concept[7]:.2f}  Contact: {int(concept[4])}",
                ]
            if pred_concepts is not None:
                concept_lines.append(
                    f"Pred Risk: {pred_concepts[5]:.2f}  Pred Broken: {int(pred_concepts[6])}  Pred Speed: {pred_concepts[7]:.2f}  Pred Contact: {int(pred_concepts[4])}",
                )
            frames.append(_annotate_frame(frame, header_lines, concept_lines))
        else:
            frames.append(Image.fromarray(frame))

        # GRU state management
        if h_new is not None:
            h_t = h_new
            if done or truncated:
                h_t = torch.zeros_like(h_t)

        if done or truncated:
            break

    if not frames:
        raise RuntimeError("No frames captured during recording")

    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
    duration_ms = int(1000 / max(fps, 1))
    frames[0].save(
        video_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"[record] saved {len(frames)} frames → {video_path}")
