#!/usr/bin/env python3
"""Print a compact TensorBoard scalar summary for one training run."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable


def _load_event_accumulator(log_dir: Path):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:
        raise SystemExit(
            "TensorBoard is not installed in this runtime. Run colab/setup.sh first."
        ) from exc

    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    return accumulator


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    if not math.isfinite(value):
        return str(value)
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.5f}"


def _trend(values: list[float], n: int = 5) -> float | None:
    if len(values) < 2:
        return None
    head = values[: min(n, len(values))]
    tail = values[-min(n, len(values)) :]
    return float(sum(tail) / len(tail) - sum(head) / len(head))


def summarize_scalars(log_dir: Path, preferred_tags: Iterable[str]) -> int:
    accumulator = _load_event_accumulator(log_dir)
    tags = accumulator.Tags().get("scalars", [])
    if not tags:
        print(f"[tb] no scalar tags found under {log_dir}")
        return 1

    preferred = [tag for tag in preferred_tags if tag in tags]
    remaining = sorted(tag for tag in tags if tag not in preferred)
    ordered_tags = preferred + remaining

    print(f"[tb] log_dir: {log_dir}")
    print(f"[tb] scalar tags: {len(tags)}")
    print()
    print(f"{'tag':<36} {'n':>4} {'first':>10} {'last':>10} {'min':>10} {'max':>10} {'trend':>10} {'last_step':>10}")
    print("-" * 98)

    suspicious = []
    for tag in ordered_tags:
        events = accumulator.Scalars(tag)
        values = [float(event.value) for event in events]
        if not values:
            continue
        last_step = int(events[-1].step)
        delta = _trend(values)
        print(
            f"{tag:<36} {len(values):>4} "
            f"{_fmt(values[0]):>10} {_fmt(values[-1]):>10} "
            f"{_fmt(min(values)):>10} {_fmt(max(values)):>10} "
            f"{_fmt(delta):>10} {last_step:>10}"
        )
        if any(not math.isfinite(value) for value in values):
            suspicious.append(f"{tag}: contains NaN/inf")

    print()
    if suspicious:
        print("[tb] warnings:")
        for warning in suspicious:
            print(f"  - {warning}")
    else:
        print("[tb] no NaN/inf scalar values detected")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Run directory containing tb/ or a TensorBoard log directory.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    log_dir = run_dir / "tb" if (run_dir / "tb").exists() else run_dir
    if not log_dir.exists():
        raise SystemExit(f"TensorBoard log directory not found: {log_dir}")

    preferred_tags = (
        "train/mean_episode_reward",
        "eval/mean_reward",
        "eval/success_rate",
        "eval/normalized_return",
        "eval/mean_length",
        "train/pg_loss",
        "train/vf_loss",
        "concept/concept_actor_loss",
        "concept/concept_critic_loss",
        "concept/concept_ent_loss",
    )
    raise SystemExit(summarize_scalars(log_dir, preferred_tags))


if __name__ == "__main__":
    main()
