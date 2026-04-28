import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cluster"))

from cluster.aggregate import collect_rows, write_csv
from cluster.run_pilot import (
    PILOT_BENCHMARKS,
    command_for_config,
    round1_configs,
    round2_configs,
)


def write_run(
    root,
    name,
    *,
    benchmark_id="armed_corridor",
    method="no_concept",
    training_mode="two_phase",
    temporal_encoding="none",
    normalized_return=0.5,
    success_rate=0.5,
    dominant_action_fraction=0.5,
):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    metadata = {
        "benchmark_id": benchmark_id,
        "method": method,
        "seed": 42,
        "training_mode": training_mode,
        "temporal_encoding": temporal_encoding,
        "learning_rate": 0.0003,
        "ent_coef": 0.01,
        "lambda_v": 0.5,
        "lambda_s": 0.5,
        "num_labels": 500,
        "query_num_times": 1,
    }
    eval_payload = {
        "mean_reward": normalized_return,
        "std_reward": 0.1,
        "success_rate": success_rate,
        "normalized_return": normalized_return,
        "mean_episode_length": 3.0,
        "dominant_action_fraction": dominant_action_fraction,
        "terminal_info_counts": {
            "failure_reason": {"miss": 1, "none": 1},
            "success": {"True": 1, "False": 1},
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata) + "\n")
    (run_dir / "eval.json").write_text(json.dumps(eval_payload) + "\n")
    np.savez(
        run_dir / "concept_acc.npz",
        timesteps=np.array([4, 8]),
        names=np.array(["parity"]),
        values=np.array([[0.25], [0.75]], dtype=np.float32),
    )
    return run_dir


def test_aggregate_collects_metadata_eval_and_concept_metrics(tmp_path):
    write_run(tmp_path, "run_a")
    rows = collect_rows(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["benchmark_id"] == "armed_corridor"
    assert row["method"] == "no_concept"
    assert row["dominant_action_fraction"] == 0.5
    assert row["terminal_cause_breakdown"] == {"failure_reason": {"miss": 1, "none": 1}}
    assert row["concept_metrics"]["parity"] == 0.75

    csv_path = tmp_path / "summary.csv"
    write_csv(rows, csv_path)
    assert "concept_metrics" in csv_path.read_text()


def test_round1_generates_specified_stage0_jobs():
    configs = round1_configs(PILOT_BENCHMARKS)
    assert len(configs) == 20
    vanilla = [config for config in configs if config.method == "vanilla_freeze"]
    assert vanilla
    assert {config.training_mode for config in vanilla} == {"two_phase"}
    cac = [config for config in configs if config.method == "concept_actor_critic"]
    assert len(cac) == 12
    assert {config.training_mode for config in cac} == {"two_phase", "end_to_end", "joint"}
    assert {config.temporal_encoding for config in configs} == {"gru", "none"}

    cmd = command_for_config(configs[0], output_dir="stage0_results", run_single="cluster/run_single.sbatch")
    export_arg = next(part for part in cmd if part.startswith("--export="))
    for required in (
        "TEMPORAL_ENCODING=",
        "TRAINING_MODE=",
        "LEARNING_RATE=",
        "ENT_COEF=",
        "LAMBDA_V=",
        "LAMBDA_S=",
        "NUM_LABELS=",
        "QUERY_NUM_TIMES=",
        "TOTAL_TIMESTEPS=300000",
    ):
        assert required in export_arg


def test_round1_can_optionally_add_gvf_without_changing_default():
    default_configs = round1_configs(PILOT_BENCHMARKS)
    extended_configs = round1_configs(PILOT_BENCHMARKS, include_gvf=True, gvf_pairing="0,1")
    assert len(default_configs) == 20
    assert len(extended_configs) == 24
    gvf_configs = [config for config in extended_configs if config.method == "gvf"]
    assert len(gvf_configs) == 4
    assert {config.training_mode for config in gvf_configs} == {"two_phase"}
    assert {config.temporal_encoding for config in gvf_configs} == {"gru", "none"}

    cmd = command_for_config(gvf_configs[0], output_dir="stage0_results", run_single="cluster/run_single.sbatch")
    export_arg = next(part for part in cmd if part.startswith("--export="))
    assert "METHOD=gvf" in export_arg
    assert "GVF_PAIRING=0:1" in export_arg


def test_round2_carries_top1_cac_architecture(tmp_path):
    for benchmark in PILOT_BENCHMARKS:
        write_run(
            tmp_path,
            f"{benchmark}_no_concept_gru",
            benchmark_id=benchmark,
            method="no_concept",
            temporal_encoding="gru",
            normalized_return=0.7,
        )
        write_run(
            tmp_path,
            f"{benchmark}_vanilla_none",
            benchmark_id=benchmark,
            method="vanilla_freeze",
            temporal_encoding="none",
            normalized_return=0.6,
        )
        write_run(
            tmp_path,
            f"{benchmark}_cac_top",
            benchmark_id=benchmark,
            method="concept_actor_critic",
            training_mode="two_phase",
            temporal_encoding="gru",
            normalized_return=0.9,
        )
        write_run(
            tmp_path,
            f"{benchmark}_cac_runner_up",
            benchmark_id=benchmark,
            method="concept_actor_critic",
            training_mode="joint",
            temporal_encoding="none",
            normalized_return=0.8,
        )

    configs = round2_configs(collect_rows(tmp_path), PILOT_BENCHMARKS)
    assert len(configs) == 24
    cac_configs = [config for config in configs if config.method == "concept_actor_critic"]
    assert len(cac_configs) == 8
    assert {config.training_mode for config in cac_configs} == {"two_phase"}
    assert {config.temporal_encoding for config in cac_configs} == {"gru"}
    assert {(config.learning_rate, config.ent_coef) for config in cac_configs} == {
        (0.0003, 0.01),
        (0.0003, 0.001),
        (0.0001, 0.01),
        (0.0001, 0.001),
    }
