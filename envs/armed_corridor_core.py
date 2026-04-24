"""
armed_corridor_core.py - Pure simulator core for Armed Corridor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Dict, Optional, Tuple

import numpy as np


GRID_WIDTH = 9
GRID_HEIGHT = 7
MAX_STEPS = 40
OBS_SIZE = 84

START = (1, 5)
TRIGGER = (3, 5)
JUNCTION = (5, 5)
GOAL = (7, 3)

HALL_CELLS = {(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)}
SHORT_ROUTE_HAZARD_CELLS = {(6, 5), (7, 5), (7, 4)}
DETOUR_CELLS = {(5, 4), (5, 3), (5, 2), (6, 2), (7, 2)}
WALKABLE_CELLS = HALL_CELLS | SHORT_ROUTE_HAZARD_CELLS | DETOUR_CELLS | {GOAL}

FUSE_SHORT = "short"
FUSE_LONG = "long"
FUSE_TYPES = {FUSE_SHORT, FUSE_LONG}
INITIAL_BUDGETS = {
    FUSE_SHORT: 5,
    FUSE_LONG: 8,
}

STEP_REWARD = -0.01
DETOUR_PENALTY = -0.05
GOAL_REWARD = 1.0
COLLAPSE_REWARD = -1.0

CONCEPT_NAMES = [
    "agent_x",
    "agent_y",
    "at_trigger",
    "at_junction",
    "on_short_route",
    "fuse_state",
    "remaining_budget_bin",
]
TASK_TYPES = ["classification"] * len(CONCEPT_NAMES)
NUM_CLASSES = [9, 7, 2, 2, 2, 3, 5]
TEMPORAL_CONCEPTS = [5, 6]

COLOR_BG = np.array([12, 12, 14], dtype=np.uint8)
COLOR_WALL = np.array([24, 26, 30], dtype=np.uint8)
COLOR_FLOOR = np.array([208, 212, 216], dtype=np.uint8)
COLOR_TRIGGER_NEUTRAL = np.array([242, 184, 80], dtype=np.uint8)
COLOR_TRIGGER_SHORT = np.array([214, 48, 49], dtype=np.uint8)
COLOR_TRIGGER_LONG = np.array([42, 157, 88], dtype=np.uint8)
COLOR_GOAL = np.array([232, 221, 117], dtype=np.uint8)
COLOR_AGENT = np.array([33, 33, 38], dtype=np.uint8)


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
class ArmedCorridorState:
    agent_pos: Tuple[int, int] = START
    step_count: int = 0
    triggered: bool = False
    fuse_type: Optional[str] = None
    steps_since_trigger: Optional[int] = None
    cue_visible: bool = False


class ArmedCorridorSimulator:
    """
    Deterministic task dynamics and rendering for Armed Corridor.
    """

    actions = Actions

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.state = ArmedCorridorState()
        self._forced_fuse_type: Optional[str] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        forced_fuse_type: Optional[str] = None,
    ) -> ArmedCorridorState:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._validate_forced_fuse_type(forced_fuse_type)
        self._forced_fuse_type = forced_fuse_type
        self.state = ArmedCorridorState()
        return self.state

    def step(
        self,
        action: int,
    ) -> Tuple[ArmedCorridorState, float, bool, bool, Dict[str, object]]:
        try:
            action_enum = Actions(int(action))
        except ValueError as exc:
            raise ValueError(f"Invalid action: {action}") from exc

        prev_state = self.state
        prev_pos = prev_state.agent_pos
        dx, dy = ACTION_DELTAS[action_enum]
        cand_pos = (prev_pos[0] + dx, prev_pos[1] + dy)
        next_pos = cand_pos if cand_pos in WALKABLE_CELLS else prev_pos

        next_state = replace(
            prev_state,
            agent_pos=next_pos,
            step_count=prev_state.step_count + 1,
            cue_visible=False,
        )

        if not prev_state.triggered and next_pos == TRIGGER:
            fuse_type = self._forced_fuse_type or self.rng.choice(sorted(FUSE_TYPES))
            next_state = replace(
                next_state,
                triggered=True,
                fuse_type=fuse_type,
                steps_since_trigger=0,
                cue_visible=True,
            )
        elif prev_state.triggered:
            next_state = replace(
                next_state,
                triggered=True,
                fuse_type=prev_state.fuse_type,
                steps_since_trigger=(prev_state.steps_since_trigger or 0) + 1,
            )

        reward = STEP_REWARD
        if next_pos in DETOUR_CELLS:
            reward += DETOUR_PENALTY

        terminated = False
        truncated = False
        info: Dict[str, object] = {}

        remaining_budget = self.compute_remaining_budget(next_state)

        if next_pos == GOAL:
            terminated = True
            reward = GOAL_REWARD
            info["success"] = True
            info["route_taken"] = self._route_taken_for_goal(prev_pos)
        elif (
            next_pos in SHORT_ROUTE_HAZARD_CELLS
            and remaining_budget == 0
        ):
            terminated = True
            reward = COLLAPSE_REWARD
            info["success"] = False
            info["failure_reason"] = "collapse"
            info["route_taken"] = "short"
        elif next_state.step_count >= MAX_STEPS:
            truncated = True
            info["success"] = False
            info["failure_reason"] = "timeout"
            info["route_taken"] = "none"

        self.state = next_state
        return next_state, reward, terminated, truncated, info

    def compute_remaining_budget(self, state: Optional[ArmedCorridorState] = None) -> Optional[int]:
        state = self.state if state is None else state
        if not state.triggered or state.fuse_type is None or state.steps_since_trigger is None:
            return None
        initial_budget = INITIAL_BUDGETS[state.fuse_type]
        return max(initial_budget - state.steps_since_trigger, 0)

    def extract_concepts(self, state: Optional[ArmedCorridorState] = None) -> np.ndarray:
        state = self.state if state is None else state
        remaining_budget = self.compute_remaining_budget(state)
        return np.array(
            [
                state.agent_pos[0],
                state.agent_pos[1],
                int(state.agent_pos == TRIGGER),
                int(state.agent_pos == JUNCTION),
                int(state.agent_pos in SHORT_ROUTE_HAZARD_CELLS),
                self._fuse_state_class(state),
                self._remaining_budget_bin(remaining_budget),
            ],
            dtype=np.float32,
        )

    def get_state_observation(self, state: Optional[ArmedCorridorState] = None) -> np.ndarray:
        state = self.state if state is None else state
        return np.array(
            [
                state.agent_pos[0] / (GRID_WIDTH - 1),
                state.agent_pos[1] / (GRID_HEIGHT - 1),
            ],
            dtype=np.float32,
        )

    def render_rgb(self, state: Optional[ArmedCorridorState] = None) -> np.ndarray:
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
                pos = (x, y)
                if pos in WALKABLE_CELLS:
                    color = COLOR_FLOOR
                if pos == GOAL:
                    color = COLOR_GOAL
                if pos == TRIGGER:
                    if state.cue_visible and state.fuse_type == FUSE_SHORT:
                        color = COLOR_TRIGGER_SHORT
                    elif state.cue_visible and state.fuse_type == FUSE_LONG:
                        color = COLOR_TRIGGER_LONG
                    else:
                        color = COLOR_TRIGGER_NEUTRAL
                canvas[y0:y1, x0:x1] = color

                canvas[y0:y1, x0] = COLOR_BG
                canvas[y0:y1, x1 - 1] = COLOR_BG
                canvas[y0, x0:x1] = COLOR_BG
                canvas[y1 - 1, x0:x1] = COLOR_BG

        ax, ay = state.agent_pos
        y0 = y_offset + ay * cell_h
        x0 = x_offset + ax * cell_w
        agent_margin_x = 2
        agent_margin_y = 3
        canvas[
            y0 + agent_margin_y:y0 + cell_h - agent_margin_y,
            x0 + agent_margin_x:x0 + cell_w - agent_margin_x,
        ] = COLOR_AGENT
        return canvas

    @staticmethod
    def _validate_forced_fuse_type(forced_fuse_type: Optional[str]) -> None:
        if forced_fuse_type not in (None, FUSE_SHORT, FUSE_LONG):
            raise ValueError(
                "forced_fuse_type must be one of {'short', 'long', None}"
            )

    @staticmethod
    def _fuse_state_class(state: ArmedCorridorState) -> int:
        if not state.triggered or state.fuse_type is None:
            return 0
        return 1 if state.fuse_type == FUSE_SHORT else 2

    @staticmethod
    def _remaining_budget_bin(remaining_budget: Optional[int]) -> int:
        if remaining_budget is None:
            return 0
        if remaining_budget == 0:
            return 1
        if 1 <= remaining_budget <= 2:
            return 2
        if 3 <= remaining_budget <= 5:
            return 3
        return 4

    @staticmethod
    def _route_taken_for_goal(prev_pos: Tuple[int, int]) -> str:
        if prev_pos == (7, 4):
            return "short"
        if prev_pos == (7, 2):
            return "detour"
        return "none"