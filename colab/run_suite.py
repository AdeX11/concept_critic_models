#!/usr/bin/env python3
"""colab/run_suite.py — Resumable Stage 0 pilot runner for Google Colab.

Reuses cluster.run_pilot.build_configs to define what to run, then executes
each RunConfig as a `python train.py ...` subprocess. Skips runs whose
eval.json already exists, so the suite is resumable across Colab session
disconnects: re-run the same command and it picks up where it left off.

Usage (round 1, no prior results needed):
  python colab/run_suite.py --round round1 \
      --benchmarks armed_corridor phase_crossing \
      --output_dir /content/drive/MyDrive/concept_critic/stage0 \
      --max_minutes 660

Usage (round 2+, depends on prior round results):
  python colab/run_suite.py --round round2 \
      --benchmarks armed_corridor phase_crossing \
      --results_root /content/drive/MyDrive/concept_critic/stage0 \
      --output_dir /content/drive/MyDrive/concept_critic/stage0 \
      --max_minutes 660
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cluster"))

from cluster.run_pilot import build_configs, RunConfig  # noqa: E402


def run_dir_for(config: RunConfig, output_dir: Path) -> Path:
    name = (
        f"{config.method}_{config.training_mode}_{config.temporal_encoding}_"
        f"{config.benchmark_id}_seed{config.seed}"
    )
    return output_dir / name


def is_complete(run_dir: Path) -> bool:
    return (run_dir / "eval.json").exists()


def train_command(config: RunConfig, output_dir: Path) -> List[str]:
    cmd = [
        sys.executable, "-u", str(REPO_ROOT / "train.py"),
        "--method", config.method,
        "--benchmark", config.benchmark_id,
        "--seed", str(config.seed),
        "--training_mode", config.training_mode,
        "--temporal_encoding", config.temporal_encoding,
        "--total_timesteps", str(config.total_timesteps),
        "--learning_rate", f"{config.learning_rate:g}",
        "--ent_coef", f"{config.ent_coef:g}",
        "--lambda_v", f"{config.lambda_v:g}",
        "--lambda_s", f"{config.lambda_s:g}",
        "--num_labels", str(config.num_labels),
        "--query_num_times", str(config.query_num_times),
        "--device", "auto",
        "--output_dir", str(output_dir),
    ]
    if config.gvf_pairing:
        cmd.extend(["--gvf_pairing", config.gvf_pairing])
    return cmd


def append_progress(progress_path: Path, entry: dict) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def run_train_process(cmd: List[str], log_path: Path, stream_train_logs: bool) -> int:
    """Run train.py, always saving train.log, optionally teeing live output."""
    with log_path.open("w") as logf:
        if not stream_train_logs:
            proc = subprocess.run(
                cmd, stdout=logf, stderr=subprocess.STDOUT, check=False,
            )
            return proc.returncode

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                print(line, end="", flush=True)
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
        return proc.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 0 pilot on Colab.")
    parser.add_argument(
        "--round", dest="round_name", required=True,
        choices=(
            "round1", "round2", "round3", "round4",
            "revalidate", "confirm", "ablation",
            "lean_compare", "lean_confirm",
        ),
    )
    parser.add_argument(
        "--benchmarks", nargs="+",
        default=["armed_corridor", "phase_crossing"],
    )
    parser.add_argument(
        "--results_root", default="results",
        help="Where prior round results live (used for round2+ winner selection).",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Where new run outputs go (e.g. a Drive path).",
    )
    parser.add_argument(
        "--max_minutes", type=float, default=660.0,
        help="Stop launching new runs after this many wall-clock minutes "
             "(default 660 = 11h, safely under Colab Pro session cap).",
    )
    parser.add_argument("--include_gvf", action="store_true")
    parser.add_argument("--gvf_pairing", default=None)
    parser.add_argument(
        "--stream_train_logs",
        action="store_true",
        help="Tee each train.py subprocess to notebook/stdout while still writing train.log.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    configs = build_configs(args)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "_suite_progress.jsonl"

    pending: List[tuple[RunConfig, Path]] = []
    skipped = 0
    for config in configs:
        rd = run_dir_for(config, output_dir)
        if is_complete(rd):
            skipped += 1
            continue
        pending.append((config, rd))

    print(
        f"[suite] round={args.round_name} "
        f"total={len(configs)} pending={len(pending)} "
        f"skipped(complete)={skipped} "
        f"stream_train_logs={'on' if args.stream_train_logs else 'off'}"
    )
    if args.dry_run:
        for config, rd in pending:
            print("DRY:", " ".join(train_command(config, output_dir)))
        return

    started_at = time.time()
    elapsed_limit_s = args.max_minutes * 60
    completed = 0
    failed = 0
    for idx, (config, rd) in enumerate(pending, 1):
        if time.time() - started_at >= elapsed_limit_s:
            print(
                f"[suite] hit max_minutes={args.max_minutes}, "
                f"exiting after {idx - 1}/{len(pending)} runs"
            )
            break

        cmd = train_command(config, output_dir)
        rd.mkdir(parents=True, exist_ok=True)
        log_path = rd / "train.log"
        print(f"[suite] [{idx}/{len(pending)}] launching: {rd.name}", flush=True)
        run_started = time.time()
        returncode = run_train_process(cmd, log_path, args.stream_train_logs)
        run_duration_s = time.time() - run_started
        ok = returncode == 0 and is_complete(rd)
        completed += int(ok)
        failed += int(not ok)
        append_progress(progress_path, {
            "ts": time.time(),
            "run_dir": rd.name,
            "ok": ok,
            "returncode": returncode,
            "duration_s": run_duration_s,
            "method": config.method,
            "benchmark_id": config.benchmark_id,
            "seed": config.seed,
            "training_mode": config.training_mode,
            "temporal_encoding": config.temporal_encoding,
            "total_timesteps": config.total_timesteps,
        })
        status = "ok" if ok else f"FAIL(rc={returncode})"
        print(f"[suite] [{idx}/{len(pending)}] {status} in {run_duration_s:.0f}s")

    elapsed = time.time() - started_at
    print(
        f"[suite] done. ok={completed} fail={failed} "
        f"elapsed={elapsed/60:.1f}min"
    )


if __name__ == "__main__":
    main()
