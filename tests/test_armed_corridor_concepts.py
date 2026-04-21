import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.armed_corridor import (
    make_single_armed_corridor_env,
    make_single_armed_corridor_state_env,
    make_single_armed_corridor_visible_env,
)
from envs.armed_corridor_core import (
    ArmedCorridorSimulator,
    ArmedCorridorState,
    FUSE_LONG,
    GOAL,
    TRIGGER,
)


def test_pre_trigger_concept_labels_are_exact():
    sim = ArmedCorridorSimulator(seed=0)
    state = sim.reset()
    concepts = sim.extract_concepts(state)
    np.testing.assert_array_equal(
        concepts,
        np.array([1, 5, 0, 0, 0, 0, 0], dtype=np.float32),
    )


def test_at_trigger_is_one_on_reentry_after_activation():
    sim = ArmedCorridorSimulator(seed=0)
    state = ArmedCorridorState(
        agent_pos=TRIGGER,
        step_count=4,
        triggered=True,
        fuse_type=FUSE_LONG,
        steps_since_trigger=2,
        cue_visible=False,
    )
    concepts = sim.extract_concepts(state)
    assert concepts[2] == 1.0
    assert concepts[5] == 2.0


def test_on_short_route_is_zero_at_goal_cell():
    sim = ArmedCorridorSimulator(seed=0)
    state = ArmedCorridorState(
        agent_pos=GOAL,
        step_count=9,
        triggered=True,
        fuse_type=FUSE_LONG,
        steps_since_trigger=6,
        cue_visible=False,
    )
    concepts = sim.extract_concepts(state)
    assert concepts[4] == 0.0


def test_get_concept_returns_a_copy():
    env = make_single_armed_corridor_state_env(seed=0)
    concept = env.get_concept()
    concept[0] = 99.0
    assert env.get_concept()[0] != 99.0


def test_state_wrapper_observation_shape_is_2():
    env = make_single_armed_corridor_state_env(seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (2,)


def test_pixel_wrapper_observation_shapes_match_n_stack():
    env = make_single_armed_corridor_env(seed=0, n_stack=1)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3, 84, 84)

    env = make_single_armed_corridor_env(seed=0, n_stack=4)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (12, 84, 84)


def test_visible_wrapper_exposes_fuse_and_budget_in_pixels():
    env = make_single_armed_corridor_visible_env(seed=0, n_stack=1)
    obs_short, _ = env.reset(seed=0, options={"forced_fuse_type": "short"})
    env.step(1)
    obs_short, _, _, _, _ = env.step(1)  # trigger short

    env = make_single_armed_corridor_visible_env(seed=0, n_stack=1)
    obs_long, _ = env.reset(seed=0, options={"forced_fuse_type": "long"})
    env.step(1)
    obs_long, _, _, _, _ = env.step(1)   # trigger long

    # Fuse block differs between short and long.
    short_rgb = obs_short[:3].transpose(1, 2, 0)
    long_rgb = obs_long[:3].transpose(1, 2, 0)
    assert not np.array_equal(short_rgb[74:80, 6:12], long_rgb[74:80, 6:12])
