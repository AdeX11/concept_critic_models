import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark_registry import get_benchmark_spec, validate_benchmark_registry
from envs.registry import make_env_pair, make_single_env


def test_tmaze_registered_and_requires_delayed_cue_memory():
    validate_benchmark_registry()
    spec = get_benchmark_spec("tmaze")
    assert spec.env_name == "tmaze"
    assert "delayed_cue" in spec.capability_tags

    env = make_single_env("tmaze", seed=0, temporal_encoding="none")
    obs, info = env.reset(seed=1)
    cue = int(info["concept"][0])
    assert obs.shape == (4,)
    assert obs[1] == 1.0
    assert obs[2] == float(cue)

    for _ in range(3):
        obs, _, terminated, truncated, info = env.step(0)
        assert not terminated
        assert not truncated
    assert obs[1] == 0.0
    assert obs[2] == 0.0
    assert int(info["concept"][0]) == cue

    while obs[3] < 0.5:
        obs, _, terminated, truncated, _ = env.step(0)
        assert not terminated
        assert not truncated
    _, reward, terminated, truncated, _ = env.step(0)
    assert terminated
    assert not truncated
    assert reward < -0.5
    env.close()


def test_tmaze_stacked_registry_shape_and_vector_concepts():
    vec_env, single_env, n_stack = make_env_pair("tmaze", n_envs=2, seed=7, temporal_encoding="stacked")
    assert n_stack == 4
    obs, infos = vec_env.reset()
    assert obs.shape == (2, 16)
    concepts = vec_env.get_attr("current_concept")
    assert len(concepts) == 2
    assert np.asarray(concepts[0]).shape == (2,)
    single_obs, _ = single_env.reset(seed=7)
    assert single_obs.shape == (16,)
    vec_env.close()
    single_env.close()


def test_hidden_velocity_registered_and_hides_velocity_from_observation():
    spec = get_benchmark_spec("hidden_velocity")
    assert spec.env_name == "hidden_velocity"
    assert "hidden_velocity" in spec.capability_tags

    env = make_single_env("hidden_velocity", seed=3)
    obs, info = env.reset(seed=3)
    concept = info["concept"]
    assert obs.shape == (4,)
    assert concept.shape == (8,)
    assert np.allclose(obs[:4], concept[:4])
    assert not np.allclose(concept[5:7], 0.0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (4,)
    assert isinstance(float(reward), float)
    assert not terminated
    assert isinstance(truncated, bool)
    assert info["concept"].shape == (8,)
    env.close()
