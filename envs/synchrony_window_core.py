"""
synchrony_window_core.py - Pure simulator core for Synchrony Window.
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
TOP_HAZARD_Y = 3
BOTTOM_HAZARD_Y = 4
HAZARD_AGENT_Y = {TOP_HAZARD_Y, BOTTOM_HAZARD_Y}
MOVER_RAIL_X = tuple(range(1, 8))
DIRECTION_LEFT = "left"
DIRECTION_RIGHT = "right"
DIRECTIONS = (DIRECTION_LEFT, DIRECTION_RIGHT)

STEP_REWARD = -0.02
GOAL_REWARD = 1.0
COLLISION_REWARD = -1.0

CONCEPT_NAMES = [
    "agent_y",
    "at_start",
    "in_hazard_zone",
    "top_mover_x",
    "bottom_mover_x",
    "top_mover_direction",
    "bottom_mover_direction",
]
TASK_TYPES = ["classification"] * len(CONCEPT_NAMES)
NUM_CLASSES = [7, 2, 2, 7, 7, 2, 2]
TEMPORAL_CONCEPTS = [5, 6]

COLOR_BG = np.array([10, 12, 15], dtype=np.uint8)
COLOR_WALL = np.array([24, 27, 33], dtype=np.uint8)
COLOR_LANE = np.array([88, 98, 116], dtype=np.uint8)
COLOR_CORRIDOR = np.array([216, 220, 224], dtype=np.uint8)
COLOR_GOAL = np.array([228, 221, 112], dtype=np.uint8)
COLOR_AGENT = np.array([34, 35, 40], dtype=np.uint8)
COLOR_TOP = np.array([244, 114, 182], dtype=np.uint8)
COLOR_BOTTOM = np.array([99, 102, 241], dtype=np.uint8)


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
    Actions.LEFT: (0 - 1, 0),
    Actions.STAY: (0, 0),
}


DIR_TO_CLASS = {DIRECTION_LEFT: 0, DIRECTION_RIGHT: 1}


@dataclass(frozen=True)
class SynchronyWindowState:
    agent_pos: Tuple[int, int] = START
    top_mover_x: int = 2
    top_direction: str = DIRECTION_RIGHT
    bottom_mover_x: int = 6
    bottom_direction: str = DIRECTION_LEFT
    step_count: int = 0


class SynchronyWindowSimulator:
    """
    Corridor crossing task where two independent movers define the safe window.

    The bottom mover controls safety on the first crossing step (agent enters y=4),
    and the top mover controls safety on the second crossing step (agent enters y=3).
    Whether starting now is safe depends on the future positions of BOTH movers.
    """

    actions = Actions

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.state = SynchronyWindowState()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        forced_top_x: Optional[int] = None,
        forced_top_direction: Optional[str] = None,
        forced_bottom_x: Optional[int] = None,
        forced_bottom_direction: Optional[str] = None,
    ) -> SynchronyWindowState:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._validate_reset_controls(
            forced_top_x,
            forced_top_direction,
            forced_bottom_x,
            forced_bottom_direction,
        )
        top_x = forced_top_x if forced_top_x is not None else int(self.rng.choice(MOVER_RAIL_X))
        bottom_x = forced_bottom_x if forced_bottom_x is not None else int(self.rng.choice(MOVER_RAIL_X))
        top_direction = (
            forced_top_direction if forced_top_direction is not None else str(self.rng.choice(DIRECTIONS))
        )
        bottom_direction = (
            forced_bottom_direction if forced_bottom_direction is not None else str(self.rng.choice(DIRECTIONS))
        )
        self.state = SynchronyWindowState(
            agent_pos=START,
            top_mover_x=top_x,
            top_direction=top_direction,
            bottom_mover_x=bottom_x,
            bottom_direction=bottom_direction,
            step_count=0,
        )
        return self.state

    def step(self, action: int) -> Tuple[SynchronyWindowState, float, bool, bool, Dict[str, object]]:
        try:
            action_enum = Actions(int(action))
        except ValueError as exc:
            raise ValueError(f"Invalid action: {action}") from exc

        prev_state = self.state
        dx, dy = ACTION_DELTAS[action_enum]
        cand_pos = (prev_state.agent_pos[0] + dx, prev_state.agent_pos[1] + dy)
        next_pos = cand_pos if cand_pos in CORRIDOR_CELLS else prev_state.agent_pos

        top_x, top_direction = self._advance_mover(prev_state.top_mover_x, prev_state.top_direction)
        bottom_x, bottom_direction = self._advance_mover(prev_state.bottom_mover_x, prev_state.bottom_direction)

        next_state = replace(
            prev_state,
            agent_pos=next_pos,
            top_mover_x=top_x,
            top_direction=top_direction,
            bottom_mover_x=bottom_x,
            bottom_direction=bottom_direction,
            step_count=prev_state.step_count + 1,
        )

        reward = STEP_REWARD
        terminated = False
        truncated = False
        info: Dict[str, object] = {
            "success": False,
            "top_mover_x": next_state.top_mover_x,
            "top_direction": next_state.top_direction,
            "bottom_mover_x": next_state.bottom_mover_x,
            "bottom_direction": next_state.bottom_direction,
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

    def extract_concepts(self, state: Optional[SynchronyWindowState] = None) -> np.ndarray:
        state = self.state if state is None else state
        return np.array(
            [
                state.agent_pos[1],
                int(state.agent_pos == START),
                int(state.agent_pos[1] in HAZARD_AGENT_Y),
                state.top_mover_x - MOVER_RAIL_X[0],
                state.bottom_mover_x - MOVER_RAIL_X[0],
                DIR_TO_CLASS[state.top_direction],
                DIR_TO_CLASS[state.bottom_direction],
            ],
            dtype=np.float32,
        )

    def get_state_observation(self, state: Optional[SynchronyWindowState] = None) -> np.ndarray:
        state = self.state if state is None else state
        return np.array(
            [
                state.agent_pos[1] / (GRID_HEIGHT - 1),
                (state.top_mover_x - MOVER_RAIL_X[0]) / (len(MOVER_RAIL_X) - 1),
                (state.bottom_mover_x - MOVER_RAIL_X[0]) / (len(MOVER_RAIL_X) - 1),
                float(DIR_TO_CLASS[state.top_direction]),
                float(DIR_TO_CLASS[state.bottom_direction]),
            ],
            dtype=np.float32,
        )

    def render_rgb(self, state: Optional[SynchronyWindowState] = None) -> np.ndarray:
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
                if x in MOVER_RAIL_X and y in (TOP_HAZARD_Y, BOTTOM_HAZARD_Y):
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

        for mover_x, mover_y, color in (
            (state.top_mover_x, TOP_HAZARD_Y, COLOR_TOP),
            (state.bottom_mover_x, BOTTOM_HAZARD_Y, COLOR_BOTTOM),
        ):
            y0 = y_offset + mover_y * cell_h
            x0 = x_offset + mover_x * cell_w
            canvas[y0 + 2:y0 + cell_h - 2, x0 + 1:x0 + cell_w - 1] = color

        ax, ay = state.agent_pos
        y0 = y_offset + ay * cell_h
        x0 = x_offset + ax * cell_w
        canvas[y0 + 3:y0 + cell_h - 3, x0 + 2:x0 + cell_w - 2] = COLOR_AGENT
        return canvas

    def safe_to_start_crossing(self, state: Optional[SynchronyWindowState] = None) -> bool:
        state = self.state if state is None else state
        bottom_x1, _ = self._advance_mover(state.bottom_mover_x, state.bottom_direction)
        top_x1, top_dir1 = self._advance_mover(state.top_mover_x, state.top_direction)
        top_x2, _ = self._advance_mover(top_x1, top_dir1)
        return bottom_x1 != CORRIDOR_X and top_x2 != CORRIDOR_X

    @staticmethod
    def _validate_reset_controls(
        forced_top_x: Optional[int],
        forced_top_direction: Optional[str],
        forced_bottom_x: Optional[int],
        forced_bottom_direction: Optional[str],
    ) -> None:
        if forced_top_x is not None and forced_top_x not in MOVER_RAIL_X:
            raise ValueError(f"forced_top_x must be one of {MOVER_RAIL_X}")
        if forced_bottom_x is not None and forced_bottom_x not in MOVER_RAIL_X:
            raise ValueError(f"forced_bottom_x must be one of {MOVER_RAIL_X}")
        if forced_top_direction is not None and forced_top_direction not in DIRECTIONS:
            raise ValueError(f"forced_top_direction must be one of {DIRECTIONS}")
        if forced_bottom_direction is not None and forced_bottom_direction not in DIRECTIONS:
            raise ValueError(f"forced_bottom_direction must be one of {DIRECTIONS}")

    @staticmethod
    def _advance_mover(mover_x: int, direction: str) -> Tuple[int, str]:
        delta = -1 if direction == DIRECTION_LEFT else 1
        next_x = mover_x + delta
        next_direction = direction
        if next_x < MOVER_RAIL_X[0]:
            next_x = MOVER_RAIL_X[0] + 1
            next_direction = DIRECTION_RIGHT
        elif next_x > MOVER_RAIL_X[-1]:
            next_x = MOVER_RAIL_X[-1] - 1
            next_direction = DIRECTION_LEFT
        return next_x, next_direction

    @staticmethod
    def _is_collision(state: SynchronyWindowState) -> bool:
        if state.agent_pos[1] == TOP_HAZARD_Y:
            return state.top_mover_x == CORRIDOR_X
        if state.agent_pos[1] == BOTTOM_HAZARD_Y:
            return state.bottom_mover_x == CORRIDOR_X
        return False
