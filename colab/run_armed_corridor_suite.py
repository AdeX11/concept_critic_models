#!/usr/bin/env python3
"""Armed-corridor-only Colab runner with live diagnostics.

This runner is intentionally separate from colab/run_suite.py.  It focuses on
the state -> visible pixel -> hidden pixel armed-corridor progression so reward,
representation, and concept-bottleneck failures can be isolated without mixing
in phase-crossing results.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class RunConfig:
    benchmark_id: str
    method: str
    seed: int
    training_mode: str
    temporal_encoding: str
    total_timesteps: int
    learning_rate: float
    ent_coef: float
    lambda_v: float
    lambda_s: float
    num_labels: int
    query_num_times: int


def run_dir_for(config: RunConfig, output_dir: Path) -> Path:
    name = (
        f"{config.method}_{config.training_mode}_{config.temporal_encoding}_"
        f"{config.benchmark_id}_seed{config.seed}"
    )
    return output_dir / name


def is_complete(run_dir: Path) -> bool:
    return (run_dir / "eval.json").exists()


def train_command(config: RunConfig, output_dir: Path) -> List[str]:
    return [
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


def append_progress(progress_path: Path, entry: dict) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def run_train_process(cmd: List[str], log_path: Path, stream_train_logs: bool) -> int:
    with log_path.open("w") as logf:
        if not stream_train_logs:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=False)
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


def build_configs(args: argparse.Namespace) -> List[RunConfig]:
    configs: List[RunConfig] = []
    for benchmark_id in args.benchmarks:
        total_timesteps = (
            args.state_timesteps
            if benchmark_id.endswith("_state")
            else args.pixel_timesteps
        )
        for method in args.methods:
            temporal_options = args.temporal_encodings
            if method == "concept_actor_critic" and args.include_cac_none:
                temporal_options = sorted(set([*temporal_options, "none"]))
            for temporal_encoding in temporal_options:
                configs.append(
                    RunConfig(
                        benchmark_id=benchmark_id,
                        method=method,
                        seed=args.seed,
                        training_mode="two_phase",
                        temporal_encoding=temporal_encoding,
                        total_timesteps=total_timesteps,
                        learning_rate=args.learning_rate,
                        ent_coef=args.ent_coef,
                        lambda_v=args.lambda_v,
                        lambda_s=args.lambda_s,
                        num_labels=args.num_labels,
                        query_num_times=args.query_num_times,
                    )
                )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run armed-corridor diagnostic suite on Colab.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["armed_corridor_state", "armed_corridor_visible", "armed_corridor"],
        choices=["armed_corridor_state", "armed_corridor_visible", "armed_corridor"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["no_concept", "vanilla_freeze", "concept_actor_critic"],
        choices=["no_concept", "vanilla_freeze", "concept_actor_critic"],
    )
    parser.add_argument("--temporal_encodings", nargs="+", default=["gru"], choices=["gru", "none", "stacked"])
    parser.add_argument("--include_cac_none", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state_timesteps", type=int, default=100_000)
    parser.add_argument("--pixel_timesteps", type=int, default=300_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--lambda_v", type=float, default=0.5)
    parser.add_argument("--lambda_s", type=float, default=0.5)
    parser.add_argument("--num_labels", type=int, default=500)
    parser.add_argument("--query_num_times", type=int, default=1)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_minutes", type=float, default=660.0)
    parser.add_argument("--stream_train_logs", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    configs = build_configs(args)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "_armed_corridor_progress.jsonl"

    pending: List[tuple[RunConfig, Path]] = []
    skipped = 0
    for config in configs:
        rd = run_dir_for(config, output_dir)
        if is_complete(rd):
            skipped += 1
        else:
            pending.append((config, rd))

    print(
        f"[armed-suite] total={len(configs)} pending={len(pending)} "
        f"skipped(complete)={skipped} "
        f"stream_train_logs={'on' if args.stream_train_logs else 'off'}"
    )
    if args.dry_run:
        for config, _ in pending:
            print("DRY:", " ".join(train_command(config, output_dir)))
        return

    started_at = time.time()
    elapsed_limit_s = args.max_minutes * 60
    completed = 0
    failed = 0
    for idx, (config, rd) in enumerate(pending, 1):
        if time.time() - started_at >= elapsed_limit_s:
            print(
                f"[armed-suite] hit max_minutes={args.max_minutes}, "
                f"exiting after {idx - 1}/{len(pending)} runs"
            )
            break

        cmd = train_command(config, output_dir)
        rd.mkdir(parents=True, exist_ok=True)
        log_path = rd / "train.log"
        print(f"[armed-suite] [{idx}/{len(pending)}] launching: {rd.name}", flush=True)
        run_started = time.time()
        returncode = run_train_process(cmd, log_path, args.stream_train_logs)
        duration_s = time.time() - run_started
        ok = returncode == 0 and is_complete(rd)
        completed += int(ok)
        failed += int(not ok)
        append_progress(progress_path, {
            "ts": time.time(),
            "run_dir": rd.name,
            "ok": ok,
            "returncode": returncode,
            "duration_s": duration_s,
            "benchmark_id": config.benchmark_id,
            "method": config.method,
            "seed": config.seed,
            "training_mode": config.training_mode,
            "temporal_encoding": config.temporal_encoding,
            "total_timesteps": config.total_timesteps,
        })
        status = "ok" if ok else f"FAIL(rc={returncode})"
        print(f"[armed-suite] [{idx}/{len(pending)}] {status} in {duration_s:.0f}s")

    elapsed = time.time() - started_at
    print(
        f"[armed-suite] done. ok={completed} fail={failed} "
        f"elapsed={elapsed/60:.1f}min"
    )


if __name__ == "__main__":
    main()
