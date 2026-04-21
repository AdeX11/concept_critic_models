"""
phase_crossing_core.py - Pure simulator core for Phase Crossing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Dict, Optional, Tuple

import numpy as np


GRID_WIDTH = 7
GRID_HEIGHT = 7
OBS_SIZE = 84
MAX_STEPS = 15

START = (3, 5)
GOAL = (3, 1)
CORRIDOR_X = 3
CORRIDOR_CELLS = {(CORRIDOR_X, y) for y in range(1, 6)}
HAZARD_AGENT_Y = {3, 4}
SWEEPER_X_VALUES = (1, 2, 3, 4, 5)
SWEEPER_LEFT = "left"
SWEEPER_RIGHT = "right"
SWEEPER_DIRECTIONS = (SWEEPER_LEFT, SWEEPER_RIGHT)

STEP_REWARD = -0.02
GOAL_REWARD = 1.0
COLLISION_REWARD = -1.0

CONCEPT_NAMES = [
    "agent_y",
    "at_start",
    "in_hazard_zone",
    "sweeper_x_phase",
    "sweeper_direction",
]
TASK_TYPES = ["classification"] * len(CONCEPT_NAMES)
NUM_CLASSES = [7, 2, 2, 5, 2]
TEMPORAL_CONCEPTS = [4]

COLOR_BG = np.array([11, 12, 15], dtype=np.uint8)
COLOR_WALL = np.array([24, 27, 33], dtype=np.uint8)
COLOR_LANE = np.array([88, 98, 116], dtype=np.uint8)
COLOR_CORRIDOR = np.array([216, 220, 224], dtype=np.uint8)
COLOR_GOAL = np.array([228, 221, 112], dtype=np.uint8)
COLOR_AGENT = np.array([34, 35, 40], dtype=np.uint8)
COLOR_SWEEPER = np.array([55, 138, 255], dtype=np.uint8)


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


@dataclass(frozen=True)
class PhaseCrossingState:
    agent_pos: Tuple[int, int] = START
    sweeper_x: int = 3
    sweeper_direction: str = SWEEPER_RIGHT
    step_count: int = 0


class PhaseCrossingSimulator:
    """
    Vertical crossing task with a horizontally sweeping blocker.

    The blocker spans the corridor cells y in {3, 4}. Safe entry from the
    start cell depends on where the blocker will be one and two steps later,
    which is ambiguous from a single frame whenever the current blocker
    position is compatible with both directions.
    """

    actions = Actions

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.state = PhaseCrossingState()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        forced_sweeper_x: Optional[int] = None,
        forced_sweeper_direction: Optional[str] = None,
    ) -> PhaseCrossingState:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._validate_reset_controls(forced_sweeper_x, forced_sweeper_direction)
        sweeper_x = forced_sweeper_x if forced_sweeper_x is not None else int(self.rng.choice(SWEEPER_X_VALUES))
        sweeper_direction = (
            forced_sweeper_direction
            if forced_sweeper_direction is not None
            else str(self.rng.choice(SWEEPER_DIRECTIONS))
        )
        self.state = PhaseCrossingState(
            agent_pos=START,
            sweeper_x=sweeper_x,
            sweeper_direction=sweeper_direction,
            step_count=0,
        )
        return self.state

    def step(self, action: int) -> Tuple[PhaseCrossingState, float, bool, bool, Dict[str, object]]:
        try:
            action_enum = Actions(int(action))
        except ValueError as exc:
            raise ValueError(f"Invalid action: {action}") from exc

        prev_state = self.state
        dx, dy = ACTION_DELTAS[action_enum]
        cand_pos = (prev_state.agent_pos[0] + dx, prev_state.agent_pos[1] + dy)
        next_pos = cand_pos if cand_pos in CORRIDOR_CELLS else prev_state.agent_pos

        next_state = replace(
            prev_state,
            agent_pos=next_pos,
            sweeper_x=self._advance_sweeper(prev_state.sweeper_x, prev_state.sweeper_direction),
            step_count=prev_state.step_count + 1,
        )

        reward = STEP_REWARD
        terminated = False
        truncated = False
        info: Dict[str, object] = {
            "success": False,
            "sweeper_direction": next_state.sweeper_direction,
            "sweeper_x": next_state.sweeper_x,
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

    def extract_concepts(self, state: Optional[PhaseCrossingState] = None) -> np.ndarray:
        state = self.state if state is None else state
        return np.array(
            [
                state.agent_pos[1],
                int(state.agent_pos == START),
                int(state.agent_pos[1] in HAZARD_AGENT_Y),
                state.sweeper_x - SWEEPER_X_VALUES[0],
                0 if state.sweeper_direction == SWEEPER_LEFT else 1,
            ],
            dtype=np.float32,
        )

    def get_state_observation(self, state: Optional[PhaseCrossingState] = None) -> np.ndarray:
        state = self.state if state is None else state
        direction_scalar = 0.0 if state.sweeper_direction == SWEEPER_LEFT else 1.0
        return np.array(
            [
                state.agent_pos[1] / (GRID_HEIGHT - 1),
                (state.sweeper_x - SWEEPER_X_VALUES[0]) / (len(SWEEPER_X_VALUES) - 1),
                direction_scalar,
            ],
            dtype=np.float32,
        )

    def render_rgb(self, state: Optional[PhaseCrossingState] = None) -> np.ndarray:
        state = self.state if state is None else state
        canvas = np.broadcast_to(COLOR_BG, (OBS_SIZE, OBS_SIZE, 3)).copy()

        cell_w = 12
        cell_h = 12
        x_offset = 0
        y_offset = 0

        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                y0 = y_offset + y * cell_h
                y1 = y0 + cell_h
                x0 = x_offset + x * cell_w
                x1 = x0 + cell_w

                color = COLOR_WALL
                if x in SWEEPER_X_VALUES and y in HAZARD_AGENT_Y:
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

        sweeper_x = state.sweeper_x
        for sweeper_y in HAZARD_AGENT_Y:
            y0 = y_offset + sweeper_y * cell_h
            x0 = x_offset + sweeper_x * cell_w
            canvas[y0 + 2:y0 + cell_h - 2, x0 + 2:x0 + cell_w - 2] = COLOR_SWEEPER

        ax, ay = state.agent_pos
        y0 = y_offset + ay * cell_h
        x0 = x_offset + ax * cell_w
        canvas[y0 + 3:y0 + cell_h - 3, x0 + 3:x0 + cell_w - 3] = COLOR_AGENT
        return canvas

    def safe_to_start_crossing(self, state: Optional[PhaseCrossingState] = None) -> bool:
        state = self.state if state is None else state
        x1 = self._advance_sweeper(state.sweeper_x, state.sweeper_direction)
        x2 = self._advance_sweeper(x1, state.sweeper_direction)
        return x1 != CORRIDOR_X and x2 != CORRIDOR_X

    @staticmethod
    def _validate_reset_controls(
        forced_sweeper_x: Optional[int],
        forced_sweeper_direction: Optional[str],
    ) -> None:
        if forced_sweeper_x is not None and forced_sweeper_x not in SWEEPER_X_VALUES:
            raise ValueError(f"forced_sweeper_x must be one of {SWEEPER_X_VALUES}")
        if forced_sweeper_direction is not None and forced_sweeper_direction not in SWEEPER_DIRECTIONS:
            raise ValueError(
                f"forced_sweeper_direction must be one of {SWEEPER_DIRECTIONS}"
            )

    @staticmethod
    def _advance_sweeper(sweeper_x: int, sweeper_direction: str) -> int:
        idx = SWEEPER_X_VALUES.index(sweeper_x)
        delta = -1 if sweeper_direction == SWEEPER_LEFT else 1
        return SWEEPER_X_VALUES[(idx + delta) % len(SWEEPER_X_VALUES)]

    @staticmethod
    def _is_collision(state: PhaseCrossingState) -> bool:
        return state.agent_pos[1] in HAZARD_AGENT_Y and state.sweeper_x == CORRIDOR_X
