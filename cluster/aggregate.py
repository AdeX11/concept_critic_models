#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


METADATA_FIELDS = [
    "benchmark_id",
    "method",
    "seed",
    "training_mode",
    "temporal_encoding",
    "learning_rate",
    "ent_coef",
    "lambda_v",
    "lambda_s",
    "num_labels",
    "query_num_times",
    "gvf_pairing",
]

EVAL_FIELDS = [
    "mean_reward",
    "std_reward",
    "success_rate",
    "normalized_return",
    "mean_episode_length",
    "dominant_action_fraction",
]

CSV_FIELDS = [
    "run_dir",
    *METADATA_FIELDS,
    *EVAL_FIELDS,
    "terminal_cause_breakdown",
    "terminal_info_counts",
    "concept_metrics",
]

TERMINAL_CAUSE_KEYS = (
    "failure_reason",
    "success",
    "route_taken",
    "terminated",
    "truncated",
)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _load_concept_metrics(run_dir: Path) -> Dict[str, float]:
    path = run_dir / "concept_acc.npz"
    if not path.exists():
        return {}
    try:
        data = np.load(path, allow_pickle=True)
        names = [str(name) for name in data["names"].tolist()]
        values = data["values"]
        if values.size == 0:
            return {}
        final = values[-1]
        return {name: float(value) for name, value in zip(names, final)}
    except Exception:
        return {}


def _terminal_cause_breakdown(terminal_info_counts: Dict[str, Any]) -> Dict[str, Any]:
    for key in TERMINAL_CAUSE_KEYS:
        counts = terminal_info_counts.get(key)
        if isinstance(counts, dict) and counts:
            return {key: counts}
    return {}


def collect_rows(results_root: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for eval_path in sorted(Path(results_root).rglob("eval.json")):
        eval_payload = _load_json(eval_path)
        if not eval_payload:
            continue
        run_dir = eval_path.parent
        metadata = _load_json(run_dir / "metadata.json")
        terminal_info_counts = eval_payload.get("terminal_info_counts") or {}

        row: Dict[str, Any] = {"run_dir": str(run_dir)}
        for field in METADATA_FIELDS:
            row[field] = metadata.get(field)
        for field in EVAL_FIELDS:
            row[field] = eval_payload.get(field)
        row["terminal_info_counts"] = terminal_info_counts
        row["terminal_cause_breakdown"] = _terminal_cause_breakdown(terminal_info_counts)
        row["concept_metrics"] = _load_concept_metrics(run_dir)
        rows.append(_jsonable(row))
    return rows


def write_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            for field in ("terminal_cause_breakdown", "terminal_info_counts", "concept_metrics"):
                csv_row[field] = json.dumps(csv_row.get(field, {}), sort_keys=True)
            writer.writerow(csv_row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate benchmark eval summaries.")
    parser.add_argument("results_root")
    parser.add_argument("--out", default="cluster_summary.json")
    parser.add_argument("--csv-out", default=None)
    args = parser.parse_args()

    rows = collect_rows(args.results_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"runs": rows}, indent=2) + "\n")
    if args.csv_out:
        write_csv(rows, args.csv_out)
    print(f"[aggregate] wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
