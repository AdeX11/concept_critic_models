import json
import os
import sys
from dataclasses import replace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark_registry import (
    BENCHMARK_REGISTRY,
    emit_manifest,
    get_benchmark_spec,
    load_manifest,
    validate_manifest_payload,
    validate_benchmark_registry,
)


def test_benchmark_registry_validates():
    validate_benchmark_registry()


def test_emit_manifest_writes_selected_benchmarks(tmp_path):
    out = tmp_path / "manifest.json"
    emit_manifest(out, benchmark_ids=["armed_corridor", "armed_corridor_visible"])
    payload = json.loads(out.read_text())
    assert [entry["benchmark_id"] for entry in payload["benchmarks"]] == [
        "armed_corridor",
        "armed_corridor_visible",
    ]


def test_validate_manifest_rejects_missing_control_twin():
    broken = dict(BENCHMARK_REGISTRY)
    spec = get_benchmark_spec("armed_corridor")
    broken["armed_corridor"] = replace(spec, visible_control_id="missing_control")
    with pytest.raises(ValueError):
        validate_benchmark_registry(broken)


def test_validate_manifest_rejects_invalid_tag():
    broken = dict(BENCHMARK_REGISTRY)
    spec = get_benchmark_spec("armed_corridor")
    broken["armed_corridor"] = replace(spec, capability_tags=spec.capability_tags + ("bad_tag",))
    with pytest.raises(ValueError):
        validate_benchmark_registry(broken)


def test_load_manifest_round_trips_emitted_json(tmp_path):
    out = tmp_path / "manifest.json"
    emit_manifest(out, benchmark_ids=["cartpole", "armed_corridor_visible", "armed_corridor"])
    specs = load_manifest(out)
    assert [spec.benchmark_id for spec in specs] == ["cartpole", "armed_corridor_visible", "armed_corridor"]


def test_validate_manifest_rejects_unknown_fields():
    payload = {
        "benchmarks": [
            {
                **json.loads(json.dumps(get_benchmark_spec("armed_corridor").__dict__)),
                "unknown_field": 123,
            }
        ]
    }
    with pytest.raises(ValueError):
        validate_manifest_payload(payload)
