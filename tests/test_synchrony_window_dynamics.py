import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.synchrony_window import (
    make_single_synchrony_window_env,
    make_single_synchrony_window_state_env,
    make_single_synchrony_window_visible_env,
)
from envs.synchrony_window_core import (
    Actions,
    GOAL,
    MAX_STEPS,
    START,
    STEP_REWARD,
    SynchronyWindowSimulator,
)


def run_actions(sim: SynchronyWindowSimulator, actions):
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
        sim = SynchronyWindowSimulator(seed=seed)
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


def reactive_failure_policy(sim: SynchronyWindowSimulator, state, memory):
    if state.agent_pos == START:
        return Actions.UP if state.top_mover_x != 4 and state.bottom_mover_x != 4 else Actions.STAY
    if state.agent_pos[1] > GOAL[1]:
        return Actions.UP
    return Actions.STAY


def oracle_policy(sim: SynchronyWindowSimulator, state, memory):
    if state.agent_pos == START:
        return Actions.UP if sim.safe_to_start_crossing(state) else Actions.STAY
    if state.agent_pos[1] > GOAL[1]:
        return Actions.UP
    return Actions.STAY


def always_forward_policy(sim: SynchronyWindowSimulator, state, memory):
    return Actions.UP if state.agent_pos[1] > GOAL[1] else Actions.STAY


def visible_control_policy(sim: SynchronyWindowSimulator, state, memory):
    return oracle_policy(sim, state, memory)


def hud_signature(obs_chw):
    arr = np.asarray(obs_chw).transpose(1, 2, 0)
    return (
        int(arr[76, 8, 0]),
        int(arr[76, 36, 0]),
        int(arr[76, 46, 0]),
        int(arr[76, 74, 0]),
    )


def test_reset_controls_work():
    sim = SynchronyWindowSimulator(seed=0)
    state = sim.reset(
        forced_top_x=2,
        forced_top_direction="right",
        forced_bottom_x=6,
        forced_bottom_direction="left",
    )
    assert state.top_mover_x == 2
    assert state.top_direction == "right"
    assert state.bottom_mover_x == 6
    assert state.bottom_direction == "left"


def test_invalid_reset_controls_raise():
    sim = SynchronyWindowSimulator(seed=0)
    with pytest.raises(ValueError):
        sim.reset(forced_top_x=0)
    with pytest.raises(ValueError):
        sim.reset(forced_bottom_direction="up")


def test_movers_reflect_at_rail_boundary():
    sim = SynchronyWindowSimulator(seed=0)
    state = sim.reset(
        forced_top_x=1,
        forced_top_direction="left",
        forced_bottom_x=7,
        forced_bottom_direction="right",
    )
    state, *_ = sim.step(Actions.STAY)
    assert state.top_mover_x == 2
    assert state.top_direction == "right"
    assert state.bottom_mover_x == 6
    assert state.bottom_direction == "left"


def test_unsafe_start_collides_on_first_crossing_step():
    sim = SynchronyWindowSimulator(seed=0)
    sim.reset(
        forced_top_x=1,
        forced_top_direction="right",
        forced_bottom_x=3,
        forced_bottom_direction="right",
    )
    history = run_actions(sim, [Actions.UP])
    state, reward, terminated, truncated, info = history[-1]
    assert state.agent_pos == (4, 4)
    assert state.bottom_mover_x == 4
    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(-1.0)
    assert info["failure_reason"] == "collision"


def test_safe_start_succeeds():
    sim = SynchronyWindowSimulator(seed=0)
    sim.reset(
        forced_top_x=1,
        forced_top_direction="left",
        forced_bottom_x=1,
        forced_bottom_direction="left",
    )
    history = run_actions(sim, [Actions.UP, Actions.UP, Actions.UP, Actions.UP])
    state, reward, terminated, truncated, info = history[-1]
    assert state.agent_pos == GOAL
    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(1.0)
    assert info["success"] is True


def test_timeout_occurs_after_max_steps():
    sim = SynchronyWindowSimulator(seed=0)
    sim.reset(
        forced_top_x=2,
        forced_top_direction="left",
        forced_bottom_x=6,
        forced_bottom_direction="right",
    )
    history = run_actions(sim, [Actions.STAY] * MAX_STEPS)
    _, _, terminated, truncated, info = history[-1]
    assert terminated is False
    assert truncated is True
    assert info["failure_reason"] == "timeout"


def test_rewards_match_step_and_goal():
    sim = SynchronyWindowSimulator(seed=0)
    sim.reset(
        forced_top_x=2,
        forced_top_direction="left",
        forced_bottom_x=6,
        forced_bottom_direction="right",
    )
    _, reward, terminated, truncated, _ = sim.step(Actions.STAY)
    assert reward == pytest.approx(STEP_REWARD)
    assert not terminated and not truncated

    sim = SynchronyWindowSimulator(seed=0)
    sim.reset(
        forced_top_x=1,
        forced_top_direction="left",
        forced_bottom_x=1,
        forced_bottom_direction="left",
    )
    history = run_actions(sim, [Actions.UP, Actions.UP, Actions.UP, Actions.UP])
    _, reward, terminated, truncated, _ = history[-1]
    assert reward == pytest.approx(1.0)
    assert terminated is True
    assert truncated is False


def test_state_wrapper_observation_shape():
    env = make_single_synchrony_window_state_env(seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (5,)


def test_pixel_wrapper_stack_shapes():
    env = make_single_synchrony_window_env(seed=0, n_stack=1)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3, 84, 84)

    env = make_single_synchrony_window_env(seed=0, n_stack=4)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (12, 84, 84)


def test_visible_wrapper_exposes_direction_pair_in_hud():
    env = make_single_synchrony_window_visible_env(seed=0, n_stack=1)
    obs_a, _ = env.reset(
        seed=0,
        options={
            "forced_top_x": 2,
            "forced_top_direction": "left",
            "forced_bottom_x": 6,
            "forced_bottom_direction": "right",
        },
    )
    obs_b, _ = env.reset(
        seed=0,
        options={
            "forced_top_x": 2,
            "forced_top_direction": "right",
            "forced_bottom_x": 6,
            "forced_bottom_direction": "left",
        },
    )
    assert hud_signature(obs_a) != hud_signature(obs_b)


def test_oracle_beats_reactive_policy():
    reactive_return, reactive_success = run_policy(reactive_failure_policy)
    oracle_return, oracle_success = run_policy(oracle_policy)
    visible_return, visible_success = run_policy(visible_control_policy)

    assert oracle_return > reactive_return + 0.10
    assert oracle_success >= reactive_success
    assert visible_return == pytest.approx(oracle_return)
    assert visible_success == pytest.approx(oracle_success)


def test_oracle_beats_always_forward_policy():
    forward_return, forward_success = run_policy(always_forward_policy)
    oracle_return, oracle_success = run_policy(oracle_policy)
    assert oracle_return > forward_return + 0.10
    assert oracle_success > forward_success
