import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.synchrony_window import make_single_synchrony_window_state_env
from envs.synchrony_window_core import Actions, SynchronyWindowSimulator


def test_extract_concepts_matches_handcrafted_state():
    sim = SynchronyWindowSimulator(seed=0)
    state = sim.reset(
        forced_top_x=6,
        forced_top_direction="left",
        forced_bottom_x=2,
        forced_bottom_direction="right",
    )
    concept = sim.extract_concepts(state)
    np.testing.assert_array_equal(concept, np.array([5, 1, 0, 5, 1, 0, 1], dtype=np.float32))


def test_temporal_direction_concepts_update():
    sim = SynchronyWindowSimulator(seed=0)
    state = sim.reset(
        forced_top_x=2,
        forced_top_direction="right",
        forced_bottom_x=6,
        forced_bottom_direction="left",
    )
    concept = sim.extract_concepts(state)
    assert concept[5] == 1.0
    assert concept[6] == 0.0

    state, *_ = sim.step(Actions.STAY)
    concept = sim.extract_concepts(state)
    assert concept[3] == 2.0
    assert concept[4] == 4.0
    assert concept[5] == 1.0
    assert concept[6] == 0.0


def test_in_hazard_zone_label_turns_on_after_entry():
    sim = SynchronyWindowSimulator(seed=0)
    sim.reset(
        forced_top_x=1,
        forced_top_direction="left",
        forced_bottom_x=1,
        forced_bottom_direction="left",
    )
    state, *_ = sim.step(Actions.UP)
    concept = sim.extract_concepts(state)
    assert state.agent_pos == (4, 4)
    assert concept[2] == 1.0


def test_get_concept_returns_copy():
    env = make_single_synchrony_window_state_env(seed=0)
    env.reset(seed=0)
    concept = env.get_concept()
    concept[0] = -99
    assert env.get_concept()[0] != -99


def test_reset_options_reject_unknown_fields():
    env = make_single_synchrony_window_state_env(seed=0)
    try:
        env.reset(options={"bad_key": 1})
    except ValueError:
        return
    raise AssertionError("reset should reject unknown options")
