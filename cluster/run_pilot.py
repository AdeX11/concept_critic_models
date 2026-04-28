#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import subprocess
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import collect_rows


PILOT_BENCHMARKS = ("armed_corridor", "phase_crossing")
METHODS = ("no_concept", "vanilla_freeze", "concept_actor_critic")
REVALIDATION_SEEDS = (42, 123, 456)
BASE_SEED = 42
PILOT_TIMESTEPS = 300_000
CONFIRM_TIMESTEPS = 1_000_000


@dataclass(frozen=True)
class RunConfig:
    benchmark_id: str
    method: str
    seed: int = BASE_SEED
    training_mode: str = "two_phase"
    temporal_encoding: str = "none"
    learning_rate: float = 3e-4
    ent_coef: float = 0.01
    lambda_v: float = 0.5
    lambda_s: float = 0.5
    num_labels: int = 500
    query_num_times: int = 1
    total_timesteps: int = PILOT_TIMESTEPS
    architecture_variant: str = "current"
    gvf_pairing: Optional[str] = None

    def export_items(self, output_dir: str) -> Dict[str, str]:
        values = {
            "BENCHMARK": self.benchmark_id,
            "METHOD": self.method,
            "SEED": str(self.seed),
            "TEMPORAL_ENCODING": self.temporal_encoding,
            "TRAINING_MODE": self.training_mode,
            "LEARNING_RATE": f"{self.learning_rate:g}",
            "ENT_COEF": f"{self.ent_coef:g}",
            "LAMBDA_V": f"{self.lambda_v:g}",
            "LAMBDA_S": f"{self.lambda_s:g}",
            "NUM_LABELS": str(self.num_labels),
            "QUERY_NUM_TIMES": str(self.query_num_times),
            "TOTAL_TIMESTEPS": str(self.total_timesteps),
            "OUTPUT_DIR": output_dir,
        }
        if self.gvf_pairing:
            values["GVF_PAIRING"] = self.gvf_pairing.replace(",", ":")
        if self.architecture_variant != "current":
            # Placeholder export for the Stage 0 architecture-fix ablation. The
            # current train.py has no pre-fix architecture switch; this keeps the
            # job identity explicit until that ablation backend exists.
            values["ARCHITECTURE_VARIANT"] = self.architecture_variant
        return values

    def stage0_key(self) -> Tuple[Any, ...]:
        return (
            self.method,
            self.training_mode,
            self.temporal_encoding,
            self.learning_rate,
            self.ent_coef,
            self.lambda_v,
            self.lambda_s,
            self.num_labels,
            self.query_num_times,
        )


def round1_configs(
    benchmarks: Sequence[str] = PILOT_BENCHMARKS,
    include_gvf: bool = False,
    gvf_pairing: Optional[str] = None,
) -> List[RunConfig]:
    configs: List[RunConfig] = []
    for benchmark_id in benchmarks:
        for temporal_encoding in ("gru", "none"):
            configs.append(
                RunConfig(
                    benchmark_id=benchmark_id,
                    method="no_concept",
                    temporal_encoding=temporal_encoding,
                )
            )
            configs.append(
                RunConfig(
                    benchmark_id=benchmark_id,
                    method="vanilla_freeze",
                    training_mode="two_phase",
                    temporal_encoding=temporal_encoding,
                )
            )

        for training_mode in ("two_phase", "end_to_end", "joint"):
            for temporal_encoding in ("gru", "none"):
                configs.append(
                    RunConfig(
                        benchmark_id=benchmark_id,
                        method="concept_actor_critic",
                        training_mode=training_mode,
                        temporal_encoding=temporal_encoding,
                    )
                )
        if include_gvf:
            for temporal_encoding in ("gru", "none"):
                configs.append(
                    RunConfig(
                        benchmark_id=benchmark_id,
                        method="gvf",
                        training_mode="two_phase",
                        temporal_encoding=temporal_encoding,
                        gvf_pairing=gvf_pairing,
                    )
                )
    return configs


def _row_score(row: Dict[str, Any]) -> Optional[float]:
    score = row.get("normalized_return")
    if score is None:
        score = row.get("mean_reward")
    return None if score is None else float(score)


def _dominant_terminal_fraction(row: Dict[str, Any]) -> Optional[float]:
    breakdown = row.get("terminal_cause_breakdown") or {}
    if not isinstance(breakdown, dict) or not breakdown:
        return None
    counts = next(iter(breakdown.values()))
    if not isinstance(counts, dict) or not counts:
        return None
    total = sum(int(v) for v in counts.values())
    if total <= 0:
        return None
    return max(int(v) for v in counts.values()) / total


def candidate_passes_filters(row: Dict[str, Any]) -> bool:
    success_rate = row.get("success_rate")
    if success_rate is not None and float(success_rate) < 0.05:
        return False
    dominant_action_fraction = row.get("dominant_action_fraction")
    if dominant_action_fraction is not None and float(dominant_action_fraction) > 0.85:
        return False
    terminal_fraction = _dominant_terminal_fraction(row)
    if terminal_fraction is not None and terminal_fraction > 0.85:
        return False
    return True


def _candidate_key(row: Dict[str, Any], fields: Sequence[str]) -> Tuple[Any, ...]:
    return tuple(row.get(field) for field in fields)


def _best_candidates(
    rows: Sequence[Dict[str, Any]],
    method: str,
    key_fields: Sequence[str],
    top_n: int = 1,
) -> List[Tuple[Tuple[Any, ...], float]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("method") != method:
            continue
        if not candidate_passes_filters(row):
            continue
        score = _row_score(row)
        if score is None:
            continue
        grouped[_candidate_key(row, key_fields)].append(row)

    scored: List[Tuple[Tuple[Any, ...], float]] = []
    for key, key_rows in grouped.items():
        scores = [_row_score(row) for row in key_rows]
        usable_scores = [score for score in scores if score is not None]
        if usable_scores:
            scored.append((key, sum(usable_scores) / len(usable_scores)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_n]


def _rows_from_results(results_root: str) -> List[Dict[str, Any]]:
    rows = collect_rows(results_root)
    if not rows:
        raise SystemExit(f"No eval.json rows found under {results_root!r}")
    return rows


def _round1_winners(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    winners: Dict[str, Dict[str, Any]] = {}
    no_concept = _best_candidates(rows, "no_concept", ("temporal_encoding",))
    vanilla = _best_candidates(rows, "vanilla_freeze", ("temporal_encoding",))
    cac = _best_candidates(
        rows,
        "concept_actor_critic",
        ("training_mode", "temporal_encoding"),
        top_n=2,
    )
    if len(cac) >= 2:
        print(f"[pilot] Round 1 top 2 CAC candidates: {cac}")
    elif cac:
        print(f"[pilot] Round 1 top CAC candidate: {cac[0]}")

    if not no_concept or not vanilla or not cac:
        raise SystemExit("Could not select Round 1 winners for all methods.")

    winners["no_concept"] = {
        "training_mode": "two_phase",
        "temporal_encoding": no_concept[0][0][0],
    }
    winners["vanilla_freeze"] = {
        "training_mode": "two_phase",
        "temporal_encoding": vanilla[0][0][0],
    }
    winners["concept_actor_critic"] = {
        "training_mode": cac[0][0][0],
        "temporal_encoding": cac[0][0][1],
    }
    return winners


def round2_configs(rows: Sequence[Dict[str, Any]], benchmarks: Sequence[str]) -> List[RunConfig]:
    winners = _round1_winners(rows)
    configs: List[RunConfig] = []
    for benchmark_id in benchmarks:
        for method in METHODS:
            base = winners[method]
            for learning_rate in (3e-4, 1e-4):
                for ent_coef in (0.01, 0.001):
                    configs.append(
                        RunConfig(
                            benchmark_id=benchmark_id,
                            method=method,
                            training_mode=base["training_mode"],
                            temporal_encoding=base["temporal_encoding"],
                            learning_rate=learning_rate,
                            ent_coef=ent_coef,
                        )
                    )
    return configs


def _best_full_configs(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    key_fields = (
        "training_mode",
        "temporal_encoding",
        "learning_rate",
        "ent_coef",
        "lambda_v",
        "lambda_s",
        "num_labels",
        "query_num_times",
    )
    winners: Dict[str, Dict[str, Any]] = {}
    for method in METHODS:
        best = _best_candidates(rows, method, key_fields)
        if not best:
            raise SystemExit(f"Could not select best config for method {method!r}.")
        key = best[0][0]
        winners[method] = dict(zip(key_fields, key))
        winners[method]["method"] = method
    return winners


def _config_from_winner(
    benchmark_id: str,
    winner: Dict[str, Any],
    seed: int = BASE_SEED,
    total_timesteps: int = PILOT_TIMESTEPS,
    architecture_variant: str = "current",
) -> RunConfig:
    return RunConfig(
        benchmark_id=benchmark_id,
        method=winner["method"],
        seed=seed,
        training_mode=str(winner.get("training_mode") or "two_phase"),
        temporal_encoding=str(winner.get("temporal_encoding") or "none"),
        learning_rate=float(winner.get("learning_rate") or 3e-4),
        ent_coef=float(winner.get("ent_coef") or 0.01),
        lambda_v=float(winner.get("lambda_v") or 0.5),
        lambda_s=float(winner.get("lambda_s") or 0.5),
        num_labels=int(winner.get("num_labels") or 500),
        query_num_times=int(winner.get("query_num_times") or 1),
        total_timesteps=total_timesteps,
        architecture_variant=architecture_variant,
    )


def round3_configs(rows: Sequence[Dict[str, Any]], benchmarks: Sequence[str]) -> List[RunConfig]:
    winners = _best_full_configs(rows)
    cac = winners["concept_actor_critic"]
    configs: List[RunConfig] = []
    for benchmark_id in benchmarks:
        for lambda_v in (0.25, 0.5):
            for lambda_s in (0.25, 0.5):
                base = dict(cac)
                base["lambda_v"] = lambda_v
                base["lambda_s"] = lambda_s
                configs.append(_config_from_winner(benchmark_id, base))
    return configs


def round4_configs(rows: Sequence[Dict[str, Any]], benchmarks: Sequence[str]) -> List[RunConfig]:
    winners = _best_full_configs(rows)
    vanilla = winners["vanilla_freeze"]
    configs: List[RunConfig] = []
    for benchmark_id in benchmarks:
        for num_labels in (250, 500, 1000):
            for query_num_times in (1, 2):
                base = dict(vanilla)
                base["num_labels"] = num_labels
                base["query_num_times"] = query_num_times
                configs.append(_config_from_winner(benchmark_id, base))
    return configs


def revalidation_configs(rows: Sequence[Dict[str, Any]], benchmarks: Sequence[str]) -> List[RunConfig]:
    winners = _best_full_configs(rows)
    configs: List[RunConfig] = []
    for benchmark_id in benchmarks:
        for winner in winners.values():
            for seed in REVALIDATION_SEEDS:
                configs.append(_config_from_winner(benchmark_id, winner, seed=seed))
    return configs


def confirmation_configs(rows: Sequence[Dict[str, Any]], benchmarks: Sequence[str]) -> List[RunConfig]:
    winners = _best_full_configs(rows)
    return [
        _config_from_winner(
            benchmark_id,
            winner,
            seed=BASE_SEED,
            total_timesteps=CONFIRM_TIMESTEPS,
        )
        for benchmark_id in benchmarks
        for winner in winners.values()
    ]


def ablation_configs(rows: Sequence[Dict[str, Any]]) -> List[RunConfig]:
    winners = _best_full_configs(rows)
    return [
        _config_from_winner(
            "armed_corridor",
            winners["concept_actor_critic"],
            seed=BASE_SEED,
            total_timesteps=CONFIRM_TIMESTEPS,
            architecture_variant="pre_fix",
        )
    ]


def command_for_config(config: RunConfig, output_dir: str, run_single: str) -> List[str]:
    export_value = ",".join(
        ["ALL"] + [f"{key}={value}" for key, value in config.export_items(output_dir).items()]
    )
    return ["sbatch", f"--export={export_value}", run_single]


def print_or_submit(configs: Iterable[RunConfig], output_dir: str, dry_run: bool) -> None:
    run_single = str(Path("cluster/run_single.sbatch"))
    count = 0
    for config in configs:
        cmd = command_for_config(config, output_dir=output_dir, run_single=run_single)
        count += 1
        if dry_run:
            print(" ".join(cmd))
        else:
            subprocess.run(cmd, check=True)
    print(f"[pilot] generated {count} job(s)")


def build_configs(args: argparse.Namespace) -> List[RunConfig]:
    benchmarks = tuple(args.benchmarks)
    if args.round_name == "round1":
        return round1_configs(
            benchmarks,
            include_gvf=args.include_gvf,
            gvf_pairing=args.gvf_pairing,
        )

    rows = _rows_from_results(args.results_root)
    if args.round_name == "round2":
        return round2_configs(rows, benchmarks)
    if args.round_name == "round3":
        return round3_configs(rows, benchmarks)
    if args.round_name == "round4":
        return round4_configs(rows, benchmarks)
    if args.round_name == "revalidate":
        return revalidation_configs(rows, benchmarks)
    if args.round_name == "confirm":
        return confirmation_configs(rows, benchmarks)
    if args.round_name == "ablation":
        return ablation_configs(rows)
    raise ValueError(f"Unknown round: {args.round_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the staged Stage 0 pilot sweep.")
    parser.add_argument(
        "--round",
        dest="round_name",
        required=True,
        choices=("round1", "round2", "round3", "round4", "revalidate", "confirm", "ablation"),
    )
    parser.add_argument("--benchmarks", nargs="+", default=list(PILOT_BENCHMARKS))
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--output_dir", default="stage0_results")
    parser.add_argument(
        "--include_gvf",
        action="store_true",
        help="Add GVF jobs to Round 1. Off by default to preserve the Stage 0 protocol.",
    )
    parser.add_argument(
        "--gvf_pairing",
        default=None,
        help="Optional comma-separated 0-based concept indices for GVF cumulants.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    configs = build_configs(args)
    print_or_submit(configs, output_dir=args.output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
