import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.momentum_corridor import (
    make_single_momentum_corridor_hard_env,
    make_single_momentum_corridor_hard_state_env,
    make_single_momentum_corridor_hard_visible_env,
)
from envs.momentum_corridor_core import (
    Actions,
    GOAL,
    HARD_HAZARD_AGENT_Y,
    HARD_MAX_STEPS,
    HARD_VELOCITY_VALUES,
    MomentumCorridorSimulator,
    START,
)


def run_actions(sim: MomentumCorridorSimulator, actions):
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
        sim = MomentumCorridorSimulator(
            seed=seed,
            hazard_agent_y=HARD_HAZARD_AGENT_Y,
            max_steps=HARD_MAX_STEPS,
            velocity_values=HARD_VELOCITY_VALUES,
        )
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


def reactive_failure_policy(sim: MomentumCorridorSimulator, state, memory):
    if state.agent_pos == START:
        return Actions.UP if state.mover_x in {1, 2, 6, 7} else Actions.STAY
    if state.agent_pos[1] > GOAL[1]:
        return Actions.UP
    return Actions.STAY


def oracle_policy(sim: MomentumCorridorSimulator, state, memory):
    if state.agent_pos == START:
        return Actions.UP if sim.safe_to_start_crossing(state) else Actions.STAY
    if state.agent_pos[1] > GOAL[1]:
        return Actions.UP
    return Actions.STAY


def always_forward_policy(sim: MomentumCorridorSimulator, state, memory):
    return Actions.UP if state.agent_pos[1] > GOAL[1] else Actions.STAY


def test_hard_variant_uses_three_hazard_rows_and_slow_velocity_set():
    sim = MomentumCorridorSimulator(
        seed=0,
        hazard_agent_y=HARD_HAZARD_AGENT_Y,
        max_steps=HARD_MAX_STEPS,
        velocity_values=HARD_VELOCITY_VALUES,
    )
    assert sim.hazard_agent_y == HARD_HAZARD_AGENT_Y
    assert sim.velocity_values == HARD_VELOCITY_VALUES


def test_hard_variant_requires_three_step_safe_window():
    sim = MomentumCorridorSimulator(
        seed=0,
        hazard_agent_y=HARD_HAZARD_AGENT_Y,
        max_steps=HARD_MAX_STEPS,
        velocity_values=HARD_VELOCITY_VALUES,
    )
    sim.reset(forced_mover_x=1, forced_mover_velocity=1)
    assert sim.safe_to_start_crossing() is False


def test_hard_variant_safe_episode_succeeds():
    sim = MomentumCorridorSimulator(
        seed=0,
        hazard_agent_y=HARD_HAZARD_AGENT_Y,
        max_steps=HARD_MAX_STEPS,
        velocity_values=HARD_VELOCITY_VALUES,
    )
    sim.reset(forced_mover_x=6, forced_mover_velocity=1)
    history = run_actions(sim, [Actions.UP, Actions.UP, Actions.UP, Actions.UP])
    state, reward, terminated, truncated, info = history[-1]
    assert state.agent_pos == GOAL
    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(1.0)
    assert info["success"] is True


def test_hard_state_and_pixel_shapes():
    env = make_single_momentum_corridor_hard_state_env(seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3,)

    env = make_single_momentum_corridor_hard_env(seed=0, n_stack=1)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3, 84, 84)

    env = make_single_momentum_corridor_hard_env(seed=0, n_stack=4)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (12, 84, 84)


def test_hard_visible_wrapper_distinguishes_velocity():
    env = make_single_momentum_corridor_hard_visible_env(seed=0, n_stack=1)
    obs_left, _ = env.reset(seed=0, options={"forced_mover_x": 2, "forced_mover_velocity": -1})
    obs_right, _ = env.reset(seed=0, options={"forced_mover_x": 2, "forced_mover_velocity": 1})
    arr_left = np.asarray(obs_left[:3]).transpose(1, 2, 0)
    arr_right = np.asarray(obs_right[:3]).transpose(1, 2, 0)
    assert not np.array_equal(arr_left[76, 6], arr_right[76, 6])


def test_hard_wrapper_rejects_fast_velocity_controls():
    env = make_single_momentum_corridor_hard_state_env(seed=0)
    with pytest.raises(ValueError):
        env.reset(options={"forced_mover_x": 2, "forced_mover_velocity": 2})


def test_hard_oracle_beats_reactive_and_always_forward():
    reactive_return, reactive_success = run_policy(reactive_failure_policy)
    oracle_return, oracle_success = run_policy(oracle_policy)
    forward_return, forward_success = run_policy(always_forward_policy)

    assert oracle_return > reactive_return + 0.10
    assert oracle_return > forward_return + 0.10
    assert oracle_success >= reactive_success
    assert oracle_success > forward_success
