"""
Benchmark registry and manifest helpers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


ALLOWED_CAPABILITY_TAGS = {
    "calibration",
    "static_visible",
    "legacy_temporal",
    "event_memory",
    "countdown",
    "latent_dynamics",
    "phase_inference",
    "reactive_control",
    "diagnostic",
    "pixel",
    "delayed_cue",
    "steerability",
    "hidden_velocity",
}
ALLOWED_DIFFICULTY_LEVELS = {"calibration", "easy", "medium", "hard"}
ALLOWED_ROLES = {"calibration", "legacy", "state", "visible", "hidden", "pilot"}


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    env_name: str
    family: str
    variant: str
    role: str
    capability_tags: Tuple[str, ...]
    difficulty_level: str
    canonical_total_timesteps: int
    canonical_num_labels: int
    canonical_query_num_times: int
    primary_metrics: Tuple[str, ...]
    acceptance_gate: Dict[str, object]
    paper_central: bool = False
    requires_control_twin: bool = False
    visible_control_id: Optional[str] = None
    hidden_benchmark_id: Optional[str] = None
    reactive_reference: Optional[float] = None
    oracle_reference: Optional[float] = None


BENCHMARK_REGISTRY: Dict[str, BenchmarkSpec] = {
    "cartpole": BenchmarkSpec(
        benchmark_id="cartpole",
        env_name="cartpole",
        family="cartpole",
        variant="default",
        role="calibration",
        capability_tags=("calibration", "static_visible"),
        difficulty_level="calibration",
        canonical_total_timesteps=300_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "success_rate"),
        acceptance_gate={"type": "solvable", "min_mean_return": 450.0},
    ),
    "mountain_car": BenchmarkSpec(
        benchmark_id="mountain_car",
        env_name="mountain_car",
        family="mountain_car",
        variant="default",
        role="calibration",
        capability_tags=("calibration", "latent_dynamics"),
        difficulty_level="calibration",
        canonical_total_timesteps=500_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "success_rate"),
        acceptance_gate={"type": "solvable"},
    ),
    "lunar_lander_state": BenchmarkSpec(
        benchmark_id="lunar_lander_state",
        env_name="lunar_lander_state",
        family="lunar_lander",
        variant="state",
        role="calibration",
        capability_tags=("calibration", "static_visible"),
        difficulty_level="calibration",
        canonical_total_timesteps=500_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "success_rate", "concept_accuracy"),
        acceptance_gate={"type": "solvable"},
    ),
    "lunar_lander_pos_only": BenchmarkSpec(
        benchmark_id="lunar_lander_pos_only",
        env_name="lunar_lander_pos_only",
        family="lunar_lander",
        variant="pos_only",
        role="calibration",
        capability_tags=("calibration", "static_visible"),
        difficulty_level="calibration",
        canonical_total_timesteps=500_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "success_rate", "concept_accuracy"),
        acceptance_gate={"type": "solvable"},
    ),
    "lunar_lander": BenchmarkSpec(
        benchmark_id="lunar_lander",
        env_name="lunar_lander",
        family="lunar_lander",
        variant="pixel",
        role="calibration",
        capability_tags=("calibration", "static_visible", "pixel"),
        difficulty_level="calibration",
        canonical_total_timesteps=750_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "success_rate", "concept_accuracy"),
        acceptance_gate={"type": "solvable"},
    ),
    "dynamic_obstacles": BenchmarkSpec(
        benchmark_id="dynamic_obstacles",
        env_name="dynamic_obstacles",
        family="dynamic_obstacles",
        variant="default",
        role="legacy",
        capability_tags=("legacy_temporal", "pixel"),
        difficulty_level="medium",
        canonical_total_timesteps=750_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "concept_accuracy"),
        acceptance_gate={"type": "legacy_reference"},
    ),
    "armed_corridor_state": BenchmarkSpec(
        benchmark_id="armed_corridor_state",
        env_name="armed_corridor_state",
        family="armed_corridor",
        variant="state",
        role="state",
        capability_tags=("event_memory", "countdown", "diagnostic"),
        difficulty_level="easy",
        canonical_total_timesteps=200_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "success_rate"),
        acceptance_gate={"type": "state_gate", "min_mean_return": 0.60, "min_success_rate": 0.95},
        paper_central=True,
        reactive_reference=0.66,
        oracle_reference=0.795,
    ),
    "armed_corridor_visible": BenchmarkSpec(
        benchmark_id="armed_corridor_visible",
        env_name="armed_corridor_visible",
        family="armed_corridor",
        variant="visible",
        role="visible",
        capability_tags=("event_memory", "countdown", "reactive_control", "pixel"),
        difficulty_level="medium",
        canonical_total_timesteps=500_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "success_rate", "route_taken"),
        acceptance_gate={"type": "reactive_control"},
        paper_central=True,
        hidden_benchmark_id="armed_corridor",
        reactive_reference=0.66,
        oracle_reference=0.795,
    ),
    "armed_corridor": BenchmarkSpec(
        benchmark_id="armed_corridor",
        env_name="armed_corridor",
        family="armed_corridor",
        variant="hidden",
        role="hidden",
        capability_tags=("event_memory", "countdown", "pixel"),
        difficulty_level="medium",
        canonical_total_timesteps=1_000_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "success_rate", "route_taken", "concept_accuracy"),
        acceptance_gate={"type": "interpretable_or_solved"},
        paper_central=True,
        requires_control_twin=True,
        visible_control_id="armed_corridor_visible",
        reactive_reference=0.66,
        oracle_reference=0.795,
    ),
    "phase_crossing_state": BenchmarkSpec(
        benchmark_id="phase_crossing_state",
        env_name="phase_crossing_state",
        family="phase_crossing",
        variant="state",
        role="state",
        capability_tags=("phase_inference", "diagnostic"),
        difficulty_level="easy",
        canonical_total_timesteps=250_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "success_rate"),
        acceptance_gate={"type": "state_gate"},
        paper_central=True,
        reactive_reference=0.5239,
        oracle_reference=0.92728,
    ),
    "phase_crossing_visible": BenchmarkSpec(
        benchmark_id="phase_crossing_visible",
        env_name="phase_crossing_visible",
        family="phase_crossing",
        variant="visible",
        role="visible",
        capability_tags=("phase_inference", "reactive_control", "pixel"),
        difficulty_level="medium",
        canonical_total_timesteps=600_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "success_rate"),
        acceptance_gate={"type": "reactive_control"},
        paper_central=True,
        hidden_benchmark_id="phase_crossing",
        reactive_reference=0.5239,
        oracle_reference=0.92728,
    ),
    "phase_crossing": BenchmarkSpec(
        benchmark_id="phase_crossing",
        env_name="phase_crossing",
        family="phase_crossing",
        variant="hidden",
        role="hidden",
        capability_tags=("phase_inference", "pixel"),
        difficulty_level="medium",
        canonical_total_timesteps=800_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "success_rate", "concept_accuracy"),
        acceptance_gate={"type": "interpretable_or_solved"},
        paper_central=True,
        requires_control_twin=True,
        visible_control_id="phase_crossing_visible",
        reactive_reference=0.5239,
        oracle_reference=0.92728,
    ),
    "phase_crossing_hard_state": BenchmarkSpec(
        benchmark_id="phase_crossing_hard_state",
        env_name="phase_crossing_hard_state",
        family="phase_crossing",
        variant="hard_state",
        role="state",
        capability_tags=("phase_inference", "diagnostic"),
        difficulty_level="hard",
        canonical_total_timesteps=300_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "success_rate"),
        acceptance_gate={"type": "state_gate"},
        paper_central=True,
        reactive_reference=-0.24632,
        oracle_reference=0.916,
    ),
    "phase_crossing_hard_visible": BenchmarkSpec(
        benchmark_id="phase_crossing_hard_visible",
        env_name="phase_crossing_hard_visible",
        family="phase_crossing",
        variant="hard_visible",
        role="visible",
        capability_tags=("phase_inference", "reactive_control", "pixel"),
        difficulty_level="hard",
        canonical_total_timesteps=700_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "success_rate"),
        acceptance_gate={"type": "reactive_control"},
        paper_central=True,
        hidden_benchmark_id="phase_crossing_hard",
        reactive_reference=-0.24632,
        oracle_reference=0.916,
    ),
    "phase_crossing_hard": BenchmarkSpec(
        benchmark_id="phase_crossing_hard",
        env_name="phase_crossing_hard",
        family="phase_crossing",
        variant="hard_hidden",
        role="hidden",
        capability_tags=("phase_inference", "pixel"),
        difficulty_level="hard",
        canonical_total_timesteps=900_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "success_rate", "concept_accuracy"),
        acceptance_gate={"type": "interpretable_or_solved"},
        paper_central=True,
        requires_control_twin=True,
        visible_control_id="phase_crossing_hard_visible",
        reactive_reference=-0.24632,
        oracle_reference=0.916,
    ),
    "momentum_corridor_state": BenchmarkSpec(
        benchmark_id="momentum_corridor_state",
        env_name="momentum_corridor_state",
        family="momentum_corridor",
        variant="state",
        role="state",
        capability_tags=("latent_dynamics", "diagnostic"),
        difficulty_level="easy",
        canonical_total_timesteps=250_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "success_rate"),
        acceptance_gate={"type": "state_gate"},
        paper_central=True,
        reactive_reference=0.36484,
        oracle_reference=0.93082,
    ),
    "momentum_corridor_visible": BenchmarkSpec(
        benchmark_id="momentum_corridor_visible",
        env_name="momentum_corridor_visible",
        family="momentum_corridor",
        variant="visible",
        role="visible",
        capability_tags=("latent_dynamics", "reactive_control", "pixel"),
        difficulty_level="medium",
        canonical_total_timesteps=600_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "success_rate"),
        acceptance_gate={"type": "reactive_control"},
        paper_central=True,
        hidden_benchmark_id="momentum_corridor",
        reactive_reference=0.36484,
        oracle_reference=0.93082,
    ),
    "momentum_corridor": BenchmarkSpec(
        benchmark_id="momentum_corridor",
        env_name="momentum_corridor",
        family="momentum_corridor",
        variant="hidden",
        role="hidden",
        capability_tags=("latent_dynamics", "pixel"),
        difficulty_level="medium",
        canonical_total_timesteps=800_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "success_rate", "concept_accuracy"),
        acceptance_gate={"type": "interpretable_or_solved"},
        paper_central=True,
        requires_control_twin=True,
        visible_control_id="momentum_corridor_visible",
        reactive_reference=0.36484,
        oracle_reference=0.93082,
    ),
    "momentum_corridor_hard_state": BenchmarkSpec(
        benchmark_id="momentum_corridor_hard_state",
        env_name="momentum_corridor_hard_state",
        family="momentum_corridor",
        variant="hard_state",
        role="state",
        capability_tags=("latent_dynamics", "diagnostic"),
        difficulty_level="hard",
        canonical_total_timesteps=300_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "success_rate"),
        acceptance_gate={"type": "state_gate"},
        paper_central=True,
        reactive_reference=0.04296,
        oracle_reference=0.91344,
    ),
    "momentum_corridor_hard_visible": BenchmarkSpec(
        benchmark_id="momentum_corridor_hard_visible",
        env_name="momentum_corridor_hard_visible",
        family="momentum_corridor",
        variant="hard_visible",
        role="visible",
        capability_tags=("latent_dynamics", "reactive_control", "pixel"),
        difficulty_level="hard",
        canonical_total_timesteps=700_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "success_rate"),
        acceptance_gate={"type": "reactive_control"},
        paper_central=True,
        hidden_benchmark_id="momentum_corridor_hard",
        reactive_reference=0.04296,
        oracle_reference=0.91344,
    ),
    "momentum_corridor_hard": BenchmarkSpec(
        benchmark_id="momentum_corridor_hard",
        env_name="momentum_corridor_hard",
        family="momentum_corridor",
        variant="hard_hidden",
        role="hidden",
        capability_tags=("latent_dynamics", "pixel"),
        difficulty_level="hard",
        canonical_total_timesteps=900_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "success_rate", "concept_accuracy"),
        acceptance_gate={"type": "interpretable_or_solved"},
        paper_central=True,
        requires_control_twin=True,
        visible_control_id="momentum_corridor_hard_visible",
        reactive_reference=0.04296,
        oracle_reference=0.91344,
    ),
    "synchrony_window_state": BenchmarkSpec(
        benchmark_id="synchrony_window_state",
        env_name="synchrony_window_state",
        family="synchrony_window",
        variant="state",
        role="state",
        capability_tags=("phase_inference", "diagnostic"),
        difficulty_level="easy",
        canonical_total_timesteps=250_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "success_rate"),
        acceptance_gate={"type": "state_gate"},
        paper_central=True,
        reactive_reference=0.3702,
        oracle_reference=0.9334,
    ),
    "synchrony_window_visible": BenchmarkSpec(
        benchmark_id="synchrony_window_visible",
        env_name="synchrony_window_visible",
        family="synchrony_window",
        variant="visible",
        role="visible",
        capability_tags=("phase_inference", "reactive_control", "pixel"),
        difficulty_level="medium",
        canonical_total_timesteps=600_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "success_rate"),
        acceptance_gate={"type": "reactive_control"},
        paper_central=True,
        hidden_benchmark_id="synchrony_window",
        reactive_reference=0.3702,
        oracle_reference=0.9334,
    ),
    "synchrony_window": BenchmarkSpec(
        benchmark_id="synchrony_window",
        env_name="synchrony_window",
        family="synchrony_window",
        variant="hidden",
        role="hidden",
        capability_tags=("phase_inference", "pixel"),
        difficulty_level="medium",
        canonical_total_timesteps=800_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "success_rate", "concept_accuracy"),
        acceptance_gate={"type": "interpretable_or_solved"},
        paper_central=True,
        requires_control_twin=True,
        visible_control_id="synchrony_window_visible",
        reactive_reference=0.3702,
        oracle_reference=0.9334,
    ),
    "tmaze": BenchmarkSpec(
        benchmark_id="tmaze",
        env_name="tmaze",
        family="tmaze",
        variant="delayed_cue",
        role="pilot",
        capability_tags=("event_memory", "delayed_cue", "steerability", "diagnostic"),
        difficulty_level="medium",
        canonical_total_timesteps=1_000_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "concept_accuracy", "steerability"),
        acceptance_gate={
            "type": "memory_gate",
            "min_mean_return": 0.70,
            "requires_steerability_eval": True,
        },
        paper_central=False,
        reactive_reference=-0.11,
        oracle_reference=0.89,
    ),
    "hidden_velocity": BenchmarkSpec(
        benchmark_id="hidden_velocity",
        env_name="hidden_velocity",
        family="hidden_velocity",
        variant="default",
        role="pilot",
        capability_tags=("latent_dynamics", "hidden_velocity", "diagnostic"),
        difficulty_level="medium",
        canonical_total_timesteps=1_000_000,
        canonical_num_labels=500,
        canonical_query_num_times=1,
        primary_metrics=("mean_return", "concept_accuracy"),
        acceptance_gate={"type": "temporal_concept_diagnostic"},
        paper_central=False,
    ),
}


def list_benchmark_ids() -> Tuple[str, ...]:
    return tuple(sorted(BENCHMARK_REGISTRY.keys()))


def get_benchmark_spec(benchmark_id: str) -> BenchmarkSpec:
    try:
        return BENCHMARK_REGISTRY[benchmark_id]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark: {benchmark_id}") from exc


def compute_normalized_return(
    benchmark_id: str,
    mean_return: float,
) -> Optional[float]:
    spec = get_benchmark_spec(benchmark_id)
    if spec.reactive_reference is None or spec.oracle_reference is None:
        return None
    denom = spec.oracle_reference - spec.reactive_reference
    if abs(denom) < 1e-8:
        return None
    return float(max(0.0, min(1.0, (mean_return - spec.reactive_reference) / denom)))


def validate_benchmark_registry(registry: Optional[Dict[str, BenchmarkSpec]] = None) -> None:
    registry = BENCHMARK_REGISTRY if registry is None else registry
    seen_ids = set()
    for key, spec in registry.items():
        if key != spec.benchmark_id:
            raise ValueError(f"Registry key {key!r} does not match benchmark_id {spec.benchmark_id!r}")
        if spec.benchmark_id in seen_ids:
            raise ValueError(f"Duplicate benchmark_id: {spec.benchmark_id}")
        seen_ids.add(spec.benchmark_id)
        if spec.role not in ALLOWED_ROLES:
            raise ValueError(f"Invalid role {spec.role!r} for {spec.benchmark_id}")
        if spec.difficulty_level not in ALLOWED_DIFFICULTY_LEVELS:
            raise ValueError(f"Invalid difficulty_level {spec.difficulty_level!r} for {spec.benchmark_id}")
        bad_tags = set(spec.capability_tags) - ALLOWED_CAPABILITY_TAGS
        if bad_tags:
            raise ValueError(f"Invalid capability tags for {spec.benchmark_id}: {sorted(bad_tags)}")
        if spec.requires_control_twin and not spec.visible_control_id:
            raise ValueError(f"{spec.benchmark_id} requires a visible control twin")
        if spec.visible_control_id and spec.visible_control_id not in registry:
            raise ValueError(f"{spec.benchmark_id} references missing visible_control_id {spec.visible_control_id}")
        if spec.hidden_benchmark_id and spec.hidden_benchmark_id not in registry:
            raise ValueError(f"{spec.benchmark_id} references missing hidden_benchmark_id {spec.hidden_benchmark_id}")


def emit_manifest(path: str | Path, benchmark_ids: Optional[Sequence[str]] = None) -> Path:
    validate_benchmark_registry()
    path = Path(path)
    selected = [
        asdict(get_benchmark_spec(benchmark_id))
        for benchmark_id in (benchmark_ids or list_benchmark_ids())
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"benchmarks": selected}, indent=2) + "\n")
    return path


def benchmark_spec_from_dict(payload: dict) -> BenchmarkSpec:
    allowed_keys = {field.name for field in fields(BenchmarkSpec)}
    payload_keys = set(payload.keys())
    unknown_keys = sorted(payload_keys - allowed_keys)
    missing_keys = sorted(allowed_keys - payload_keys)
    if unknown_keys:
        raise ValueError(f"Unknown BenchmarkSpec fields: {unknown_keys}")
    if missing_keys:
        raise ValueError(f"Missing BenchmarkSpec fields: {missing_keys}")
    return BenchmarkSpec(**payload)


def validate_manifest_payload(payload: dict) -> List[BenchmarkSpec]:
    if set(payload.keys()) != {"benchmarks"}:
        raise ValueError("Manifest must contain exactly one top-level key: 'benchmarks'")
    benchmarks = payload["benchmarks"]
    if not isinstance(benchmarks, list):
        raise ValueError("Manifest 'benchmarks' must be a list")
    specs = [benchmark_spec_from_dict(entry) for entry in benchmarks]
    validate_benchmark_registry({spec.benchmark_id: spec for spec in specs})
    return specs


def load_manifest(path: str | Path) -> List[BenchmarkSpec]:
    payload = json.loads(Path(path).read_text())
    return validate_manifest_payload(payload)
