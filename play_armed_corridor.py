"""
play_armed_corridor.py - Manual keyboard play for the Armed Corridor environment.

Controls:
  Arrow keys or WASD : move
  Space              : stay
  R                  : reset
  0                  : reset with random fuse
  1                  : reset with short fuse
  2                  : reset with long fuse
  Esc / Q            : quit

Example:
  ./.venv/bin/python play_armed_corridor.py --scale 6
"""

from __future__ import annotations

import argparse
from typing import Iterable, Optional

import pygame

from envs.armed_corridor import make_single_armed_corridor_env


WINDOW_BG = (20, 22, 26)
TEXT_COLOR = (240, 242, 245)
SUBTLE_TEXT = (170, 176, 184)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Armed Corridor manually with the keyboard.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=int, default=6, help="Pixel scale factor for the env render.")
    parser.add_argument(
        "--forced_fuse_type",
        choices=["short", "long"],
        default=None,
        help="Force fuse type on the initial reset.",
    )
    parser.add_argument(
        "--scripted_actions",
        type=str,
        default="",
        help="Optional comma-separated actions for non-interactive smoke runs, e.g. right,right,space.",
    )
    parser.add_argument(
        "--quit_after_script",
        action="store_true",
        help="Quit automatically after scripted actions are exhausted.",
    )
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def action_from_token(token: str) -> Optional[int]:
    normalized = token.strip().lower()
    mapping = {
        "up": 0,
        "w": 0,
        "right": 1,
        "d": 1,
        "down": 2,
        "s": 2,
        "left": 3,
        "a": 3,
        "stay": 4,
        "space": 4,
        "wait": 4,
        "x": 4,
    }
    if not normalized:
        return None
    if normalized not in mapping:
        raise ValueError(f"Unknown scripted action token: {token!r}")
    return mapping[normalized]


def parse_scripted_actions(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [action for token in raw.split(",") if (action := action_from_token(token)) is not None]


def key_to_action(key: int) -> Optional[int]:
    if key in (pygame.K_UP, pygame.K_w):
        return 0
    if key in (pygame.K_RIGHT, pygame.K_d):
        return 1
    if key in (pygame.K_DOWN, pygame.K_s):
        return 2
    if key in (pygame.K_LEFT, pygame.K_a):
        return 3
    if key == pygame.K_SPACE:
        return 4
    return None


def reset_env(env, seed: int, forced_fuse_type: Optional[str]):
    obs, info = env.reset(seed=seed, options={"forced_fuse_type": forced_fuse_type})
    return obs, info


def build_info_lines(env, last_reward: float, terminal_info: dict, forced_fuse_type: Optional[str]) -> Iterable[str]:
    state = env.sim.state
    remaining_budget = env.sim.compute_remaining_budget()
    lines = [
        f"pos={tuple(state.agent_pos)}  step={state.step_count}/{env.sim.MAX_STEPS if hasattr(env.sim, 'MAX_STEPS') else 40}",
        f"triggered={state.triggered}  fuse={state.fuse_type}  cue_visible={state.cue_visible}",
        f"remaining_budget={remaining_budget}  last_reward={last_reward:+.2f}",
        f"forced_fuse={forced_fuse_type or 'random'}  route_taken={terminal_info.get('route_taken')}",
        "Controls: arrows/WASD move, space stay, r reset, 0 random, 1 short, 2 long, q quit",
    ]
    failure_reason = terminal_info.get("failure_reason")
    if failure_reason:
        lines.append(f"terminal={failure_reason}")
    elif terminal_info.get("success"):
        lines.append("terminal=success")
    return lines


def render_screen(
    screen: pygame.Surface,
    font: pygame.font.Font,
    env,
    scale: int,
    last_reward: float,
    terminal_info: dict,
    forced_fuse_type: Optional[str],
) -> None:
    frame = env.render()
    frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    scaled = pygame.transform.scale(
        frame_surface,
        (frame.shape[1] * scale, frame.shape[0] * scale),
    )

    screen.fill(WINDOW_BG)
    screen.blit(scaled, (16, 16))

    y = 16
    text_x = scaled.get_width() + 32
    for idx, line in enumerate(build_info_lines(env, last_reward, terminal_info, forced_fuse_type)):
        color = TEXT_COLOR if idx < 4 else SUBTLE_TEXT
        surf = font.render(line, True, color)
        screen.blit(surf, (text_x, y))
        y += surf.get_height() + 8

    pygame.display.flip()


def main() -> int:
    args = parse_args()
    scripted_actions = parse_scripted_actions(args.scripted_actions)

    env = make_single_armed_corridor_env(seed=args.seed, n_stack=1)
    reset_env(env, seed=args.seed, forced_fuse_type=args.forced_fuse_type)

    pygame.init()
    pygame.display.set_caption("Armed Corridor Manual Play")
    pygame.font.init()
    font = pygame.font.SysFont("Menlo", 18)

    frame = env.render()
    width = frame.shape[1] * args.scale + 520
    height = max(frame.shape[0] * args.scale + 32, 220)
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    last_reward = 0.0
    terminal_info: dict = {}
    forced_fuse_type = args.forced_fuse_type
    script_index = 0
    running = True

    while running:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    reset_env(env, seed=args.seed, forced_fuse_type=forced_fuse_type)
                    last_reward = 0.0
                    terminal_info = {}
                elif event.key == pygame.K_0:
                    forced_fuse_type = None
                    reset_env(env, seed=args.seed, forced_fuse_type=forced_fuse_type)
                    last_reward = 0.0
                    terminal_info = {}
                elif event.key == pygame.K_1:
                    forced_fuse_type = "short"
                    reset_env(env, seed=args.seed, forced_fuse_type=forced_fuse_type)
                    last_reward = 0.0
                    terminal_info = {}
                elif event.key == pygame.K_2:
                    forced_fuse_type = "long"
                    reset_env(env, seed=args.seed, forced_fuse_type=forced_fuse_type)
                    last_reward = 0.0
                    terminal_info = {}
                else:
                    action = key_to_action(event.key)

        if action is None and script_index < len(scripted_actions):
            action = scripted_actions[script_index]
            script_index += 1

        if action is not None:
            _, reward, terminated, truncated, info = env.step(action)
            last_reward = float(reward)
            terminal_info = dict(info)
            if terminated or truncated:
                terminal_info.setdefault("route_taken", "none")
            if (terminated or truncated) and script_index < len(scripted_actions):
                reset_env(env, seed=args.seed, forced_fuse_type=forced_fuse_type)
                last_reward = 0.0
                terminal_info = {}

        render_screen(
            screen,
            font,
            env,
            args.scale,
            last_reward,
            terminal_info,
            forced_fuse_type,
        )

        if args.quit_after_script and script_index >= len(scripted_actions):
            running = False

        clock.tick(args.fps)

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
