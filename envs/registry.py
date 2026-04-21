"""
Shared environment factory registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import gymnasium as gym

from .armed_corridor import (
    make_armed_corridor_env,
    make_armed_corridor_state_env,
    make_armed_corridor_visible_env,
    make_single_armed_corridor_env,
    make_single_armed_corridor_state_env,
    make_single_armed_corridor_visible_env,
)
from .cartpole import make_cartpole_env, make_single_cartpole_env
from .dynamic_obstacles import make_dynamic_obstacles_env, make_single_dynamic_obstacles_env
from .lunar_lander import (
    make_lunar_lander_env,
    make_lunar_lander_pos_only_env,
    make_lunar_lander_state_env,
    make_single_lunar_lander_env,
    make_single_lunar_lander_pos_only_env,
    make_single_lunar_lander_state_env,
)
from .mountain_car import make_mountain_car_env, make_single_mountain_car_env
from .momentum_corridor import (
    make_momentum_corridor_env,
    make_momentum_corridor_state_env,
    make_momentum_corridor_visible_env,
    make_single_momentum_corridor_env,
    make_single_momentum_corridor_state_env,
    make_single_momentum_corridor_visible_env,
)
from .phase_crossing import (
    make_phase_crossing_env,
    make_phase_crossing_hard_env,
    make_phase_crossing_hard_state_env,
    make_phase_crossing_hard_visible_env,
    make_phase_crossing_state_env,
    make_phase_crossing_visible_env,
    make_single_phase_crossing_env,
    make_single_phase_crossing_hard_env,
    make_single_phase_crossing_hard_state_env,
    make_single_phase_crossing_hard_visible_env,
    make_single_phase_crossing_state_env,
    make_single_phase_crossing_visible_env,
)
from .synchrony_window import (
    make_single_synchrony_window_env,
    make_single_synchrony_window_state_env,
    make_single_synchrony_window_visible_env,
    make_synchrony_window_env,
    make_synchrony_window_state_env,
    make_synchrony_window_visible_env,
)


VecFactory = Callable[..., gym.Env]
SingleFactory = Callable[..., gym.Env]


@dataclass(frozen=True)
class EnvFactorySpec:
    env_name: str
    make_vec: VecFactory
    make_single: SingleFactory
    pass_n_stack: bool = True
    state_like: bool = False


ENV_REGISTRY: Dict[str, EnvFactorySpec] = {
    "cartpole": EnvFactorySpec(
        env_name="cartpole",
        make_vec=make_cartpole_env,
        make_single=make_single_cartpole_env,
        pass_n_stack=True,
    ),
    "mountain_car": EnvFactorySpec(
        env_name="mountain_car",
        make_vec=make_mountain_car_env,
        make_single=make_single_mountain_car_env,
        pass_n_stack=False,
    ),
    "lunar_lander": EnvFactorySpec(
        env_name="lunar_lander",
        make_vec=make_lunar_lander_env,
        make_single=make_single_lunar_lander_env,
        pass_n_stack=True,
    ),
    "lunar_lander_state": EnvFactorySpec(
        env_name="lunar_lander_state",
        make_vec=make_lunar_lander_state_env,
        make_single=make_single_lunar_lander_state_env,
        pass_n_stack=False,
        state_like=True,
    ),
    "lunar_lander_pos_only": EnvFactorySpec(
        env_name="lunar_lander_pos_only",
        make_vec=make_lunar_lander_pos_only_env,
        make_single=make_single_lunar_lander_pos_only_env,
        pass_n_stack=False,
        state_like=True,
    ),
    "dynamic_obstacles": EnvFactorySpec(
        env_name="dynamic_obstacles",
        make_vec=make_dynamic_obstacles_env,
        make_single=make_single_dynamic_obstacles_env,
        pass_n_stack=True,
    ),
    "armed_corridor": EnvFactorySpec(
        env_name="armed_corridor",
        make_vec=make_armed_corridor_env,
        make_single=make_single_armed_corridor_env,
        pass_n_stack=True,
    ),
    "armed_corridor_visible": EnvFactorySpec(
        env_name="armed_corridor_visible",
        make_vec=make_armed_corridor_visible_env,
        make_single=make_single_armed_corridor_visible_env,
        pass_n_stack=True,
    ),
    "armed_corridor_state": EnvFactorySpec(
        env_name="armed_corridor_state",
        make_vec=make_armed_corridor_state_env,
        make_single=make_single_armed_corridor_state_env,
        pass_n_stack=False,
        state_like=True,
    ),
    "phase_crossing": EnvFactorySpec(
        env_name="phase_crossing",
        make_vec=make_phase_crossing_env,
        make_single=make_single_phase_crossing_env,
        pass_n_stack=True,
    ),
    "phase_crossing_visible": EnvFactorySpec(
        env_name="phase_crossing_visible",
        make_vec=make_phase_crossing_visible_env,
        make_single=make_single_phase_crossing_visible_env,
        pass_n_stack=True,
    ),
    "phase_crossing_state": EnvFactorySpec(
        env_name="phase_crossing_state",
        make_vec=make_phase_crossing_state_env,
        make_single=make_single_phase_crossing_state_env,
        pass_n_stack=False,
        state_like=True,
    ),
    "phase_crossing_hard": EnvFactorySpec(
        env_name="phase_crossing_hard",
        make_vec=make_phase_crossing_hard_env,
        make_single=make_single_phase_crossing_hard_env,
        pass_n_stack=True,
    ),
    "phase_crossing_hard_visible": EnvFactorySpec(
        env_name="phase_crossing_hard_visible",
        make_vec=make_phase_crossing_hard_visible_env,
        make_single=make_single_phase_crossing_hard_visible_env,
        pass_n_stack=True,
    ),
    "phase_crossing_hard_state": EnvFactorySpec(
        env_name="phase_crossing_hard_state",
        make_vec=make_phase_crossing_hard_state_env,
        make_single=make_single_phase_crossing_hard_state_env,
        pass_n_stack=False,
        state_like=True,
    ),
    "momentum_corridor": EnvFactorySpec(
        env_name="momentum_corridor",
        make_vec=make_momentum_corridor_env,
        make_single=make_single_momentum_corridor_env,
        pass_n_stack=True,
    ),
    "momentum_corridor_visible": EnvFactorySpec(
        env_name="momentum_corridor_visible",
        make_vec=make_momentum_corridor_visible_env,
        make_single=make_single_momentum_corridor_visible_env,
        pass_n_stack=True,
    ),
    "momentum_corridor_state": EnvFactorySpec(
        env_name="momentum_corridor_state",
        make_vec=make_momentum_corridor_state_env,
        make_single=make_single_momentum_corridor_state_env,
        pass_n_stack=False,
        state_like=True,
    ),
    "synchrony_window": EnvFactorySpec(
        env_name="synchrony_window",
        make_vec=make_synchrony_window_env,
        make_single=make_single_synchrony_window_env,
        pass_n_stack=True,
    ),
    "synchrony_window_visible": EnvFactorySpec(
        env_name="synchrony_window_visible",
        make_vec=make_synchrony_window_visible_env,
        make_single=make_single_synchrony_window_visible_env,
        pass_n_stack=True,
    ),
    "synchrony_window_state": EnvFactorySpec(
        env_name="synchrony_window_state",
        make_vec=make_synchrony_window_state_env,
        make_single=make_single_synchrony_window_state_env,
        pass_n_stack=False,
        state_like=True,
    ),
}


def list_env_names() -> Tuple[str, ...]:
    return tuple(sorted(ENV_REGISTRY.keys()))


def get_env_spec(env_name: str) -> EnvFactorySpec:
    try:
        return ENV_REGISTRY[env_name]
    except KeyError as exc:
        raise ValueError(f"Unknown env: {env_name}") from exc


def resolve_n_stack(env_name: str, temporal_encoding: str) -> Optional[int]:
    spec = get_env_spec(env_name)
    if not spec.pass_n_stack:
        return None
    return 4 if temporal_encoding == "stacked" else 1


def make_env_pair(
    env_name: str,
    n_envs: int,
    seed: int,
    temporal_encoding: str = "none",
) -> tuple[gym.Env, gym.Env, Optional[int]]:
    spec = get_env_spec(env_name)
    n_stack = resolve_n_stack(env_name, temporal_encoding)
    kwargs = {"seed": seed}
    if spec.pass_n_stack:
        kwargs["n_stack"] = n_stack
    vec_env = spec.make_vec(n_envs=n_envs, **kwargs)
    single_env = spec.make_single(**kwargs)
    return vec_env, single_env, n_stack


def make_single_env(
    env_name: str,
    seed: int,
    temporal_encoding: str = "none",
) -> gym.Env:
    spec = get_env_spec(env_name)
    kwargs = {"seed": seed}
    if spec.pass_n_stack:
        kwargs["n_stack"] = resolve_n_stack(env_name, temporal_encoding)
    return spec.make_single(**kwargs)
