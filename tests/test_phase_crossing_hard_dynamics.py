import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.phase_crossing import (
    make_single_phase_crossing_hard_env,
    make_single_phase_crossing_hard_state_env,
    make_single_phase_crossing_hard_visible_env,
)
from envs.phase_crossing_core import Actions, GOAL, PhaseCrossingSimulator, HARD_HAZARD_ROWS


def run_actions(sim: PhaseCrossingSimulator, actions):
    history = []
    for action in actions:
        state, reward, terminated, truncated, info = sim.step(action)
        history.append((state, reward, terminated, truncated, info))
        if terminated or truncated:
            break
    return history


def run_policy(policy_fn, n_episodes: int = 100):
    returns = []
    successes = 0
    for seed in range(n_episodes):
        sim = PhaseCrossingSimulator(seed=seed, hazard_rows=HARD_HAZARD_ROWS)
        state = sim.reset()
        total_reward = 0.0
        memory = {}
        while True:
            action = policy_fn(sim, state, memory)
            state, reward, terminated, truncated, info = sim.step(action)
            total_reward += reward
            if terminated or truncated:
                returns.append(total_reward)
                successes += int(info.get("success", False))
                break
    return float(np.mean(returns)), successes / n_episodes


def reactive_failure_policy(sim: PhaseCrossingSimulator, state, memory):
    if state.agent_pos == (3, 5):
        return Actions.UP if state.sweeper_x in {1, 3, 5} else Actions.STAY
    if state.agent_pos[1] > GOAL[1]:
        return Actions.UP
    return Actions.STAY


def oracle_policy(sim: PhaseCrossingSimulator, state, memory):
    if state.agent_pos == (3, 5):
        return Actions.UP if sim.safe_to_start_crossing(state) else Actions.STAY
    if state.agent_pos[1] > GOAL[1]:
        return Actions.UP
    return Actions.STAY


def always_forward_policy(sim: PhaseCrossingSimulator, state, memory):
    return Actions.UP if state.agent_pos[1] > GOAL[1] else Actions.STAY


def test_hard_variant_uses_three_hazard_rows():
    sim = PhaseCrossingSimulator(seed=0, hazard_rows=HARD_HAZARD_ROWS)
    assert sim.hazard_rows == HARD_HAZARD_ROWS


def test_hard_variant_requires_longer_safe_window():
    sim = PhaseCrossingSimulator(seed=0, hazard_rows=HARD_HAZARD_ROWS)
    sim.reset(forced_sweeper_x=1, forced_sweeper_direction="right")
    assert sim.safe_to_start_crossing() is False


def test_hard_variant_safe_episode_succeeds():
    sim = PhaseCrossingSimulator(seed=0, hazard_rows=HARD_HAZARD_ROWS)
    sim.reset(forced_sweeper_x=2, forced_sweeper_direction="left")
    history = run_actions(sim, [Actions.UP, Actions.UP, Actions.UP, Actions.UP])
    state, reward, terminated, truncated, info = history[-1]
    assert state.agent_pos == GOAL
    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(1.0)
    assert info["success"] is True


def test_hard_state_and_pixel_shapes():
    env = make_single_phase_crossing_hard_state_env(seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3,)

    env = make_single_phase_crossing_hard_env(seed=0, n_stack=1)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3, 84, 84)

    env = make_single_phase_crossing_hard_env(seed=0, n_stack=4)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (12, 84, 84)


def test_hard_visible_wrapper_distinguishes_direction():
    env = make_single_phase_crossing_hard_visible_env(seed=0, n_stack=1)
    obs_left, _ = env.reset(seed=0, options={"forced_sweeper_x": 1, "forced_sweeper_direction": "left"})
    obs_right, _ = env.reset(seed=0, options={"forced_sweeper_x": 1, "forced_sweeper_direction": "right"})
    arr_left = np.asarray(obs_left[:3]).transpose(1, 2, 0)
    arr_right = np.asarray(obs_right[:3]).transpose(1, 2, 0)
    assert not np.array_equal(arr_left[76, 6], arr_right[76, 6])


def test_hard_oracle_beats_reactive_and_always_forward():
    reactive_return, reactive_success = run_policy(reactive_failure_policy)
    oracle_return, oracle_success = run_policy(oracle_policy)
    forward_return, forward_success = run_policy(always_forward_policy)

    assert oracle_return > reactive_return + 0.10
    assert oracle_return > forward_return + 0.10
    assert oracle_success >= reactive_success
    assert oracle_success > forward_success
