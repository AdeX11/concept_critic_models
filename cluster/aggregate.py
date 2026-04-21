#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate benchmark eval summaries.")
    parser.add_argument("results_root")
    parser.add_argument("--out", default="cluster_summary.json")
    args = parser.parse_args()

    rows = []
    for eval_path in Path(args.results_root).rglob("eval.json"):
        try:
            payload = json.loads(eval_path.read_text())
        except Exception:
            continue
        rows.append({
            "run_dir": str(eval_path.parent),
            "mean_reward": payload.get("mean_reward"),
            "std_reward": payload.get("std_reward"),
            "success_rate": payload.get("success_rate"),
            "normalized_return": payload.get("normalized_return"),
        })

    Path(args.out).write_text(json.dumps({"runs": rows}, indent=2) + "\n")
    print(f"[aggregate] wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
