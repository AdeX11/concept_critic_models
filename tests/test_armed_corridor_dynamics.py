import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.armed_corridor import (
    make_single_armed_corridor_env,
    make_single_armed_corridor_state_env,
)
from envs.armed_corridor_core import (
    Actions,
    ArmedCorridorSimulator,
    FUSE_LONG,
    FUSE_SHORT,
    GOAL,
    SHORT_ROUTE_HAZARD_CELLS,
    TRIGGER,
)


def run_actions(sim: ArmedCorridorSimulator, actions):
    history = []
    for action in actions:
        state, reward, terminated, truncated, info = sim.step(action)
        history.append((state, reward, terminated, truncated, info))
        if terminated or truncated:
            break
    return history


def trigger_pixel_sample(obs_chw):
    arr = np.asarray(obs_chw).transpose(1, 2, 0)
    x0 = 1 + TRIGGER[0] * 9
    y0 = TRIGGER[1] * 12
    return tuple(int(v) for v in arr[y0 + 1, x0 + 1])


def run_policy(policy_fn, forced_fuse_type):
    sim = ArmedCorridorSimulator(seed=0)
    state = sim.reset(forced_fuse_type=forced_fuse_type)
    total_reward = 0.0
    memory = {}

    while True:
        action = policy_fn(state, memory)
        state, reward, terminated, truncated, info = sim.step(action)
        total_reward += reward
        if terminated or truncated:
            return total_reward, info


def always_short_policy(state, memory):
    x, y = state.agent_pos
    if x < 7 and y == 5:
        return Actions.RIGHT
    if x == 7 and y > 3:
        return Actions.UP
    return Actions.STAY


def always_detour_policy(state, memory):
    x, y = state.agent_pos
    if y == 5 and x < 5:
        return Actions.RIGHT
    if x == 5 and y > 2:
        return Actions.UP
    if y == 2 and x < 7:
        return Actions.RIGHT
    if x == 7 and y < 3:
        return Actions.DOWN
    return Actions.STAY


def fuse_aware_policy(state, memory):
    if state.cue_visible and state.fuse_type is not None:
        memory["fuse_type"] = state.fuse_type

    x, y = state.agent_pos
    if y == 5 and x < 5:
        return Actions.RIGHT
    if (x, y) == (5, 5):
        if memory.get("fuse_type") == FUSE_LONG:
            return Actions.RIGHT
        return Actions.UP
    if y == 5 and x < 7:
        return Actions.RIGHT
    if x == 7 and y > 3:
        return Actions.UP
    if x == 5 and y > 2:
        return Actions.UP
    if y == 2 and x < 7:
        return Actions.RIGHT
    if x == 7 and y < 3:
        return Actions.DOWN
    return Actions.STAY


def test_trigger_activates_once_with_forced_fuse():
    sim = ArmedCorridorSimulator(seed=0)
    sim.reset(forced_fuse_type=FUSE_SHORT)
    state, *_ = run_actions(sim, [Actions.RIGHT, Actions.RIGHT])[-1]
    assert state.triggered is True
    assert state.fuse_type == FUSE_SHORT
    assert state.cue_visible is True
    assert state.agent_pos == TRIGGER

    state, *_ = run_actions(sim, [Actions.LEFT, Actions.RIGHT])[-1]
    assert state.triggered is True
    assert state.fuse_type == FUSE_SHORT
    assert state.cue_visible is False
    assert state.agent_pos == TRIGGER


@pytest.mark.parametrize("forced", [FUSE_SHORT, FUSE_LONG, None])
def test_forced_fuse_type_reset_variants_work(forced):
    sim = ArmedCorridorSimulator(seed=0)
    state = sim.reset(forced_fuse_type=forced)
    assert state.triggered is False
    assert state.fuse_type is None


def test_invalid_forced_fuse_type_raises_value_error():
    env = make_single_armed_corridor_state_env(seed=0)
    with pytest.raises(ValueError):
        env.reset(options={"forced_fuse_type": "bad"})


def test_cue_visible_for_exactly_one_returned_pixel_observation():
    env = make_single_armed_corridor_env(seed=0, n_stack=1)
    env.reset(seed=0, options={"forced_fuse_type": FUSE_SHORT})
    env.step(Actions.RIGHT)
    obs, _, _, _, _ = env.step(Actions.RIGHT)
    cue_pixel = trigger_pixel_sample(obs)

    obs2, _, _, _, _ = env.step(Actions.STAY)
    neutral_pixel = trigger_pixel_sample(obs2)

    assert cue_pixel != neutral_pixel


def test_steps_since_trigger_and_remaining_budget_progress_as_specified():
    sim = ArmedCorridorSimulator(seed=0)
    sim.reset(forced_fuse_type=FUSE_SHORT)
    history = run_actions(sim, [Actions.RIGHT, Actions.RIGHT, Actions.RIGHT, Actions.RIGHT])
    steps_since = [s.steps_since_trigger for s, *_ in history]
    budgets = [sim.compute_remaining_budget(s) for s, *_ in history]

    assert steps_since == [None, 0, 1, 2]
    assert budgets == [None, 5, 4, 3]


def test_short_route_fails_for_short_fuse_at_expected_hazard_cell():
    sim = ArmedCorridorSimulator(seed=0)
    sim.reset(forced_fuse_type=FUSE_SHORT)
    history = run_actions(
        sim,
        [Actions.RIGHT, Actions.RIGHT, Actions.RIGHT, Actions.RIGHT,
         Actions.RIGHT, Actions.RIGHT, Actions.UP],
    )
    state, reward, terminated, truncated, info = history[-1]
    assert state.agent_pos == (7, 4)
    assert state.agent_pos in SHORT_ROUTE_HAZARD_CELLS
    assert sim.compute_remaining_budget(state) == 0
    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(-1.0)
    assert info["failure_reason"] == "collapse"
    assert info["route_taken"] == "short"


def test_short_route_succeeds_for_long_fuse():
    sim = ArmedCorridorSimulator(seed=0)
    sim.reset(forced_fuse_type=FUSE_LONG)
    history = run_actions(
        sim,
        [Actions.RIGHT, Actions.RIGHT, Actions.RIGHT, Actions.RIGHT,
         Actions.RIGHT, Actions.RIGHT, Actions.UP, Actions.UP],
    )
    state, reward, terminated, truncated, info = history[-1]
    assert state.agent_pos == GOAL
    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(1.0)
    assert info["success"] is True
    assert info["route_taken"] == "short"


@pytest.mark.parametrize("forced", [FUSE_SHORT, FUSE_LONG])
def test_detour_succeeds_for_both_fuse_types(forced):
    sim = ArmedCorridorSimulator(seed=0)
    sim.reset(forced_fuse_type=forced)
    history = run_actions(
        sim,
        [Actions.RIGHT, Actions.RIGHT, Actions.RIGHT, Actions.RIGHT,
         Actions.UP, Actions.UP, Actions.UP, Actions.RIGHT, Actions.RIGHT, Actions.DOWN],
    )
    state, reward, terminated, truncated, info = history[-1]
    assert state.agent_pos == GOAL
    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(1.0)
    assert info["route_taken"] == "detour"


def test_reward_terms_are_exact_for_step_detour_and_goal():
    sim = ArmedCorridorSimulator(seed=0)
    sim.reset(forced_fuse_type=FUSE_LONG)
    history = run_actions(sim, [Actions.RIGHT, Actions.RIGHT, Actions.RIGHT, Actions.RIGHT, Actions.UP])
    _, reward, terminated, truncated, _ = history[-1]
    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(-0.06)

    sim = ArmedCorridorSimulator(seed=0)
    sim.reset(forced_fuse_type=FUSE_LONG)
    history = run_actions(
        sim,
        [Actions.RIGHT, Actions.RIGHT, Actions.RIGHT, Actions.RIGHT,
         Actions.RIGHT, Actions.RIGHT, Actions.UP, Actions.UP],
    )
    _, reward, terminated, truncated, _ = history[-1]
    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(1.0)


def test_wall_movement_is_a_no_op():
    sim = ArmedCorridorSimulator(seed=0)
    sim.reset()
    state, reward, terminated, truncated, _ = sim.step(Actions.LEFT)
    assert state.agent_pos == (1, 5)
    assert reward == pytest.approx(-0.01)
    assert not terminated and not truncated


def test_goal_is_not_treated_as_hazard():
    sim = ArmedCorridorSimulator(seed=0)
    sim.reset(forced_fuse_type=FUSE_LONG)
    history = run_actions(
        sim,
        [Actions.RIGHT, Actions.RIGHT, Actions.RIGHT, Actions.RIGHT,
         Actions.RIGHT, Actions.RIGHT, Actions.UP, Actions.UP],
    )
    state, _, terminated, _, info = history[-1]
    assert state.agent_pos == GOAL
    assert state.agent_pos not in SHORT_ROUTE_HAZARD_CELLS
    assert terminated is True
    assert info["success"] is True


def test_route_taken_is_terminal_resolved():
    sim = ArmedCorridorSimulator(seed=0)
    sim.reset(forced_fuse_type=FUSE_SHORT)
    _, _, terminated, _, info = run_actions(
        sim,
        [Actions.RIGHT, Actions.RIGHT, Actions.RIGHT, Actions.RIGHT,
         Actions.RIGHT, Actions.RIGHT, Actions.UP],
    )[-1]
    assert terminated is True
    assert info["route_taken"] == "short"

    sim = ArmedCorridorSimulator(seed=0)
    sim.reset(forced_fuse_type=FUSE_LONG)
    _, _, terminated, _, info = run_actions(
        sim,
        [Actions.RIGHT, Actions.RIGHT, Actions.RIGHT, Actions.RIGHT,
         Actions.UP, Actions.UP, Actions.UP, Actions.RIGHT, Actions.RIGHT, Actions.DOWN],
    )[-1]
    assert terminated is True
    assert info["route_taken"] == "detour"

    sim = ArmedCorridorSimulator(seed=0)
    sim.reset()
    for _ in range(40):
        state, reward, terminated, truncated, info = sim.step(Actions.STAY)
        if terminated or truncated:
            break
    assert truncated is True
    assert info["route_taken"] == "none"


@pytest.mark.parametrize(
    ("policy_fn", "forced", "expected_success"),
    [
        (always_short_policy, FUSE_SHORT, False),
        (always_short_policy, FUSE_LONG, True),
        (always_detour_policy, FUSE_SHORT, True),
        (always_detour_policy, FUSE_LONG, True),
        (fuse_aware_policy, FUSE_SHORT, True),
        (fuse_aware_policy, FUSE_LONG, True),
    ],
)
def test_scripted_policies_have_expected_outcomes(policy_fn, forced, expected_success):
    total_reward, info = run_policy(policy_fn, forced)
    assert info["success"] is expected_success
    if expected_success:
        assert total_reward > 0.0
    else:
        assert total_reward < 0.0


def test_fuse_aware_policy_beats_always_detour_on_mixed_fuses():
    always_detour_returns = []
    fuse_aware_returns = []
    for forced in (FUSE_SHORT, FUSE_LONG):
        detour_reward, _ = run_policy(always_detour_policy, forced)
        aware_reward, _ = run_policy(fuse_aware_policy, forced)
        always_detour_returns.append(detour_reward)
        fuse_aware_returns.append(aware_reward)

    assert np.mean(fuse_aware_returns) > np.mean(always_detour_returns)
