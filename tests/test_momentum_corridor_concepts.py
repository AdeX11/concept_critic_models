import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.momentum_corridor import make_single_momentum_corridor_state_env
from envs.momentum_corridor_core import (
    Actions,
    MomentumCorridorSimulator,
)


def test_extract_concepts_matches_handcrafted_state():
    sim = MomentumCorridorSimulator(seed=0)
    state = sim.reset(forced_mover_x=6, forced_mover_velocity=-1)
    concept = sim.extract_concepts(state)
    np.testing.assert_array_equal(concept, np.array([5, 1, 0, 5, 1], dtype=np.float32))


def test_temporal_velocity_concept_updates():
    sim = MomentumCorridorSimulator(seed=0)
    state = sim.reset(forced_mover_x=2, forced_mover_velocity=2)
    concept = sim.extract_concepts(state)
    assert concept[-1] == 3.0

    state, *_ = sim.step(Actions.STAY)
    concept = sim.extract_concepts(state)
    assert concept[3] == 3.0
    assert concept[-1] == 3.0


def test_in_hazard_zone_label_turns_on_after_entry():
    sim = MomentumCorridorSimulator(seed=0)
    sim.reset(forced_mover_x=1, forced_mover_velocity=-1)
    state, *_ = sim.step(Actions.UP)
    concept = sim.extract_concepts(state)
    assert state.agent_pos == (4, 4)
    assert concept[2] == 1.0


def test_get_concept_returns_copy():
    env = make_single_momentum_corridor_state_env(seed=0)
    env.reset(seed=0)
    concept = env.get_concept()
    concept[0] = -99
    assert env.get_concept()[0] != -99


def test_reset_options_reject_unknown_fields():
    env = make_single_momentum_corridor_state_env(seed=0)
    try:
        env.reset(options={"bad_key": 1})
    except ValueError:
        return
    raise AssertionError("reset should reject unknown options")
