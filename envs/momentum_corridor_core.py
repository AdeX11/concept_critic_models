"""
momentum_corridor_core.py - Pure simulator core for Momentum Corridor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Dict, Optional, Tuple

import numpy as np


GRID_WIDTH = 9
GRID_HEIGHT = 7
OBS_SIZE = 84
MAX_STEPS = 18

START = (4, 5)
GOAL = (4, 1)
CORRIDOR_X = 4
CORRIDOR_CELLS = {(CORRIDOR_X, y) for y in range(1, 6)}
HAZARD_AGENT_Y = {3, 4}
MOVER_RAIL_X = tuple(range(1, 8))
MOVER_RAIL_Y = 3
VELOCITY_VALUES = (-2, -1, 1, 2)

STEP_REWARD = -0.02
GOAL_REWARD = 1.0
COLLISION_REWARD = -1.0

CONCEPT_NAMES = [
    "agent_y",
    "at_start",
    "in_hazard_zone",
    "mover_x",
    "mover_velocity_state",
]
TASK_TYPES = ["classification"] * len(CONCEPT_NAMES)
NUM_CLASSES = [7, 2, 2, 7, 4]
TEMPORAL_CONCEPTS = [4]

COLOR_BG = np.array([10, 12, 15], dtype=np.uint8)
COLOR_WALL = np.array([24, 27, 33], dtype=np.uint8)
COLOR_LANE = np.array([88, 98, 116], dtype=np.uint8)
COLOR_CORRIDOR = np.array([216, 220, 224], dtype=np.uint8)
COLOR_GOAL = np.array([228, 221, 112], dtype=np.uint8)
COLOR_AGENT = np.array([34, 35, 40], dtype=np.uint8)
COLOR_MOVER = np.array([255, 118, 67], dtype=np.uint8)


class Actions(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    STAY = 4


ACTION_DELTAS = {
    Actions.UP: (0, -1),
    Actions.RIGHT: (1, 0),
    Actions.DOWN: (0, 1),
    Actions.LEFT: (-1, 0),
    Actions.STAY: (0, 0),
}


VELOCITY_TO_CLASS = {-2: 0, -1: 1, 1: 2, 2: 3}


@dataclass(frozen=True)
class MomentumCorridorState:
    agent_pos: Tuple[int, int] = START
    mover_x: int = 4
    mover_velocity: int = 1
    step_count: int = 0


class MomentumCorridorSimulator:
    """
    Vertical corridor with a horizontally sliding hazard that has hidden momentum.

    The mover occupies the crossing row y=3 and advances with signed velocity in
    {-2, -1, +1, +2}. Safe entry from the start cell depends on where the mover
    will be over the next two steps, so current position alone is insufficient.
    """

    actions = Actions

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.state = MomentumCorridorState()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        forced_mover_x: Optional[int] = None,
        forced_mover_velocity: Optional[int] = None,
    ) -> MomentumCorridorState:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._validate_reset_controls(forced_mover_x, forced_mover_velocity)
        mover_x = forced_mover_x if forced_mover_x is not None else int(self.rng.choice(MOVER_RAIL_X))
        mover_velocity = (
            forced_mover_velocity if forced_mover_velocity is not None else int(self.rng.choice(VELOCITY_VALUES))
        )
        self.state = MomentumCorridorState(
            agent_pos=START,
            mover_x=mover_x,
            mover_velocity=mover_velocity,
            step_count=0,
        )
        return self.state

    def step(self, action: int) -> Tuple[MomentumCorridorState, float, bool, bool, Dict[str, object]]:
        try:
            action_enum = Actions(int(action))
        except ValueError as exc:
            raise ValueError(f"Invalid action: {action}") from exc

        prev_state = self.state
        dx, dy = ACTION_DELTAS[action_enum]
        cand_pos = (prev_state.agent_pos[0] + dx, prev_state.agent_pos[1] + dy)
        next_pos = cand_pos if cand_pos in CORRIDOR_CELLS else prev_state.agent_pos
        mover_x, mover_velocity = self._advance_mover(prev_state.mover_x, prev_state.mover_velocity)

        next_state = replace(
            prev_state,
            agent_pos=next_pos,
            mover_x=mover_x,
            mover_velocity=mover_velocity,
            step_count=prev_state.step_count + 1,
        )

        reward = STEP_REWARD
        terminated = False
        truncated = False
        info: Dict[str, object] = {
            "success": False,
            "mover_x": next_state.mover_x,
            "mover_velocity": next_state.mover_velocity,
        }

        if next_pos == GOAL:
            terminated = True
            reward = GOAL_REWARD
            info["success"] = True
            info["failure_reason"] = None
        elif self._is_collision(next_state):
            terminated = True
            reward = COLLISION_REWARD
            info["failure_reason"] = "collision"
        elif next_state.step_count >= MAX_STEPS:
            truncated = True
            info["failure_reason"] = "timeout"
        else:
            info["failure_reason"] = None

        self.state = next_state
        return next_state, reward, terminated, truncated, info

    def extract_concepts(self, state: Optional[MomentumCorridorState] = None) -> np.ndarray:
        state = self.state if state is None else state
        return np.array(
            [
                state.agent_pos[1],
                int(state.agent_pos == START),
                int(state.agent_pos[1] in HAZARD_AGENT_Y),
                state.mover_x - MOVER_RAIL_X[0],
                VELOCITY_TO_CLASS[state.mover_velocity],
            ],
            dtype=np.float32,
        )

    def get_state_observation(self, state: Optional[MomentumCorridorState] = None) -> np.ndarray:
        state = self.state if state is None else state
        velocity_scalar = (state.mover_velocity + 2.0) / 4.0
        return np.array(
            [
                state.agent_pos[1] / (GRID_HEIGHT - 1),
                (state.mover_x - MOVER_RAIL_X[0]) / (len(MOVER_RAIL_X) - 1),
                velocity_scalar,
            ],
            dtype=np.float32,
        )

    def render_rgb(self, state: Optional[MomentumCorridorState] = None) -> np.ndarray:
        state = self.state if state is None else state
        canvas = np.broadcast_to(COLOR_BG, (OBS_SIZE, OBS_SIZE, 3)).copy()

        cell_w = 9
        cell_h = 12
        x_offset = 1
        y_offset = 0

        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                y0 = y_offset + y * cell_h
                y1 = y0 + cell_h
                x0 = x_offset + x * cell_w
                x1 = x0 + cell_w

                color = COLOR_WALL
                if x in MOVER_RAIL_X and y == MOVER_RAIL_Y:
                    color = COLOR_LANE
                if (x, y) in CORRIDOR_CELLS:
                    color = COLOR_CORRIDOR
                if (x, y) == GOAL:
                    color = COLOR_GOAL
                canvas[y0:y1, x0:x1] = color
                canvas[y0:y1, x0] = COLOR_BG
                canvas[y0:y1, x1 - 1] = COLOR_BG
                canvas[y0, x0:x1] = COLOR_BG
                canvas[y1 - 1, x0:x1] = COLOR_BG

        mx = state.mover_x
        y0 = y_offset + MOVER_RAIL_Y * cell_h
        x0 = x_offset + mx * cell_w
        canvas[y0 + 2:y0 + cell_h - 2, x0 + 1:x0 + cell_w - 1] = COLOR_MOVER

        ax, ay = state.agent_pos
        y0 = y_offset + ay * cell_h
        x0 = x_offset + ax * cell_w
        canvas[y0 + 3:y0 + cell_h - 3, x0 + 2:x0 + cell_w - 2] = COLOR_AGENT
        return canvas

    def safe_to_start_crossing(self, state: Optional[MomentumCorridorState] = None) -> bool:
        state = self.state if state is None else state
        x1, v1 = self._advance_mover(state.mover_x, state.mover_velocity)
        x2, _ = self._advance_mover(x1, v1)
        return x1 != CORRIDOR_X and x2 != CORRIDOR_X

    @staticmethod
    def _validate_reset_controls(
        forced_mover_x: Optional[int],
        forced_mover_velocity: Optional[int],
    ) -> None:
        if forced_mover_x is not None and forced_mover_x not in MOVER_RAIL_X:
            raise ValueError(f"forced_mover_x must be one of {MOVER_RAIL_X}")
        if forced_mover_velocity is not None and forced_mover_velocity not in VELOCITY_VALUES:
            raise ValueError(f"forced_mover_velocity must be one of {VELOCITY_VALUES}")

    @staticmethod
    def _advance_mover(mover_x: int, mover_velocity: int) -> Tuple[int, int]:
        next_x = mover_x + mover_velocity
        next_velocity = mover_velocity
        while next_x < MOVER_RAIL_X[0] or next_x > MOVER_RAIL_X[-1]:
            if next_x < MOVER_RAIL_X[0]:
                next_x = 2 * MOVER_RAIL_X[0] - next_x
                next_velocity = -next_velocity
            elif next_x > MOVER_RAIL_X[-1]:
                next_x = 2 * MOVER_RAIL_X[-1] - next_x
                next_velocity = -next_velocity
        return next_x, next_velocity

    @staticmethod
    def _is_collision(state: MomentumCorridorState) -> bool:
        return state.agent_pos[1] in HAZARD_AGENT_Y and state.mover_x == CORRIDOR_X
