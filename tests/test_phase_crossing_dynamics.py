import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.phase_crossing import (
    make_single_phase_crossing_env,
    make_single_phase_crossing_state_env,
    make_single_phase_crossing_visible_env,
)
from envs.phase_crossing_core import (
    Actions,
    GOAL,
    MAX_STEPS,
    PhaseCrossingSimulator,
    START,
    STEP_REWARD,
    SWEEPER_LEFT,
    SWEEPER_RIGHT,
)


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
        sim = PhaseCrossingSimulator(seed=seed)
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
    if state.agent_pos == START:
        return Actions.UP if state.sweeper_x in {1, 3, 5} else Actions.STAY
    if state.agent_pos[1] > GOAL[1]:
        return Actions.UP
    return Actions.STAY


def oracle_policy(sim: PhaseCrossingSimulator, state, memory):
    if state.agent_pos == START:
        return Actions.UP if sim.safe_to_start_crossing(state) else Actions.STAY
    if state.agent_pos[1] > GOAL[1]:
        return Actions.UP
    return Actions.STAY


def visible_control_policy(sim: PhaseCrossingSimulator, state, memory):
    return oracle_policy(sim, state, memory)


def direction_hud_signature(obs_chw):
    arr = np.asarray(obs_chw).transpose(1, 2, 0)
    return (
        int(arr[76, 6, 0]),
        int(arr[76, 6, 1]),
        int(arr[76, 78, 0]),
        int(arr[76, 78, 1]),
    )


def test_reset_controls_work():
    sim = PhaseCrossingSimulator(seed=0)
    state = sim.reset(forced_sweeper_x=1, forced_sweeper_direction=SWEEPER_LEFT)
    assert state.sweeper_x == 1
    assert state.sweeper_direction == SWEEPER_LEFT


def test_invalid_reset_controls_raise():
    sim = PhaseCrossingSimulator(seed=0)
    with pytest.raises(ValueError):
        sim.reset(forced_sweeper_x=0)
    with pytest.raises(ValueError):
        sim.reset(forced_sweeper_direction="up")


def test_sweeper_wraps_around_track():
    sim = PhaseCrossingSimulator(seed=0)
    sim.reset(forced_sweeper_x=1, forced_sweeper_direction=SWEEPER_LEFT)
    state, *_ = sim.step(Actions.STAY)
    assert state.sweeper_x == 5


def test_unsafe_start_collides_on_second_crossing_step():
    sim = PhaseCrossingSimulator(seed=0)
    sim.reset(forced_sweeper_x=1, forced_sweeper_direction=SWEEPER_RIGHT)
    history = run_actions(sim, [Actions.UP, Actions.UP])
    state, reward, terminated, truncated, info = history[-1]
    assert state.agent_pos == (3, 3)
    assert state.sweeper_x == 3
    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(-1.0)
    assert info["failure_reason"] == "collision"


def test_safe_start_succeeds():
    sim = PhaseCrossingSimulator(seed=0)
    sim.reset(forced_sweeper_x=1, forced_sweeper_direction=SWEEPER_LEFT)
    history = run_actions(sim, [Actions.UP, Actions.UP, Actions.UP, Actions.UP])
    state, reward, terminated, truncated, info = history[-1]
    assert state.agent_pos == GOAL
    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(1.0)
    assert info["success"] is True


def test_timeout_occurs_after_max_steps():
    sim = PhaseCrossingSimulator(seed=0)
    sim.reset(forced_sweeper_x=2, forced_sweeper_direction=SWEEPER_LEFT)
    history = run_actions(sim, [Actions.STAY] * MAX_STEPS)
    _, _, terminated, truncated, info = history[-1]
    assert terminated is False
    assert truncated is True
    assert info["failure_reason"] == "timeout"


def test_rewards_match_step_and_goal():
    sim = PhaseCrossingSimulator(seed=0)
    sim.reset(forced_sweeper_x=2, forced_sweeper_direction=SWEEPER_LEFT)
    _, reward, terminated, truncated, _ = sim.step(Actions.STAY)
    assert reward == pytest.approx(STEP_REWARD)
    assert not terminated and not truncated

    sim = PhaseCrossingSimulator(seed=0)
    sim.reset(forced_sweeper_x=1, forced_sweeper_direction=SWEEPER_LEFT)
    history = run_actions(sim, [Actions.UP, Actions.UP, Actions.UP, Actions.UP])
    _, reward, terminated, truncated, _ = history[-1]
    assert reward == pytest.approx(1.0)
    assert terminated is True
    assert truncated is False


def test_state_wrapper_observation_shape():
    env = make_single_phase_crossing_state_env(seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3,)


def test_pixel_wrapper_stack_shapes():
    env = make_single_phase_crossing_env(seed=0, n_stack=1)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3, 84, 84)

    env = make_single_phase_crossing_env(seed=0, n_stack=4)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (12, 84, 84)


def test_visible_wrapper_exposes_direction_in_hud():
    env = make_single_phase_crossing_visible_env(seed=0, n_stack=1)
    obs_left, _ = env.reset(seed=0, options={"forced_sweeper_x": 1, "forced_sweeper_direction": SWEEPER_LEFT})
    obs_right, _ = env.reset(seed=0, options={"forced_sweeper_x": 1, "forced_sweeper_direction": SWEEPER_RIGHT})
    assert direction_hud_signature(obs_left) != direction_hud_signature(obs_right)


def test_oracle_beats_reactive_policy():
    reactive_return, reactive_success = run_policy(reactive_failure_policy)
    oracle_return, oracle_success = run_policy(oracle_policy)
    visible_return, visible_success = run_policy(visible_control_policy)

    assert oracle_return > reactive_return + 0.10
    assert oracle_success >= reactive_success
    assert visible_return == pytest.approx(oracle_return)
    assert visible_success == pytest.approx(oracle_success)
