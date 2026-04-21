#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_registry import emit_manifest, list_benchmark_ids, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand a benchmark manifest into sbatch jobs.")
    parser.add_argument("--manifest", default="cluster/benchmarks_manifest.json")
    parser.add_argument("--benchmarks", nargs="*", default=list(list_benchmark_ids()))
    parser.add_argument("--methods", nargs="+", default=["no_concept", "vanilla_freeze", "concept_actor_critic"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    manifest_path = emit_manifest(args.manifest, benchmark_ids=args.benchmarks)
    run_single = str(Path("cluster/run_single.sbatch"))

    for benchmark in load_manifest(manifest_path):
        benchmark_id = benchmark.benchmark_id
        for method in args.methods:
            for seed in args.seeds:
                cmd = [
                    "sbatch",
                    f"--export=ALL,BENCHMARK={benchmark_id},METHOD={method},SEED={seed},OUTPUT_DIR={args.output_dir}",
                    run_single,
                ]
                if args.dry_run:
                    print(" ".join(cmd))
                else:
                    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
