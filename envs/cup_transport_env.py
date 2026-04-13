from __future__ import annotations

import numpy as np

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj
from minigrid.minigrid_env import MiniGridEnv

from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, COLORS
from minigrid.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
)


# We create Cup and WaterFountain as standalone classes, not inheriting from WorldObj,
# and implement the necessary minimal interface.

class Cup:
    def __init__(self, color, filled=False, spillability=0.1):
        self.type = "cup"
        self.color = color
        self.filled = filled
        self.init_pos = None
        self.cur_pos = None
        self.contains = None
        self.spillability = spillability  # Probability water spills out (become empty) when running

    def can_pickup(self):
        return True

    def can_overlap(self):
        return False

    def can_contain(self):
        return False

    def see_behind(self):
        return True

    def toggle(self, env, pos):
        if hasattr(self, 'filled') and not self.filled:
            self.filled = True
            return True
        return False

    def spill(self):
        if self.filled:
            self.filled = False

    def encode(self):
        # Return a tuple: (object_idx, color_idx, state)
        object_idx = OBJECT_TO_IDX.get("cup", OBJECT_TO_IDX.get("ball", 6))  # fallback to "ball"
        color_idx = COLOR_TO_IDX[self.color]
        state = 1 if self.filled else 0
        return (object_idx, color_idx, state)

    def render(self, img):
        # Draw the cup so it only occupies the center of the grid block (smaller area)
        cup_color = COLORS[self.color]
        fill_coords(img, point_in_rect(0.25, 0.75, 0.25, 0.75), cup_color)
        
        # Draw the "liquid" center: blue if filled, grey if not filled
        if self.filled:
            center_color = COLORS["blue"]
        else:
            center_color = COLORS["grey"]

        # Center smaller than the cup (e.g., inner 0.38-0.62)
        fill_coords(img, point_in_rect(0.38, 0.62, 0.38, 0.62), center_color)
 

class WaterFountain:
    def __init__(self):
        self.type = "fountain"
        self.color = "blue"
        self.state = 0
        self.init_pos = None
        self.cur_pos = None
        self.contains = None

    def can_overlap(self):
        return False

    def can_pickup(self):
        return False

    def can_contain(self):
        return False

    def see_behind(self):
        return True

    def toggle(self, env, pos):
        return False

    def encode(self):
        object_idx = OBJECT_TO_IDX.get("fountain", OBJECT_TO_IDX.get("box", 7))  # fallback to "box"
        color_idx = COLOR_TO_IDX[self.color]
        return (object_idx, color_idx, self.state)

    def render(self, img):
        c = COLORS[self.color]

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

class CupTransportEnv(MiniGridEnv):
    """
    ## Description

    The agent must pick up each cup, bring it to the water fountain and fill it (using "toggle"),
    and finally deliver the filled cup to somewhere along the right wall. All cups must be filled and delivered.

    ## Mission Space

    "Pick up each cup, fill it at the fountain (left-center), and deliver to right wall."

    ## Action Space

    | Num | Name         | Action                        |
    |-----|--------------|-------------------------------|
    | 0   | left         | Turn left                     |
    | 1   | right        | Turn right                    |
    | 2   | forward      | Move forward                  |
    | 3   | pickup       | Pick up an object             |
    | 4   | drop         | Drop an object                |
    | 5   | toggle       | Fill cup (at fountain)        |
    | 6   | run          | Run two steps forward         |
    | 7   | done         | Unused                        |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` custom for cups: 0 not filled, 1 filled

    ## Rewards

    A reward is given for each cup successfully filled and delivered to right wall.
    The total reward is the sum over all cups: for each cup, '1 - 0.9 * (step_count / max_steps)'.
    No reward for failing to fill/deliver some cups.

    ## New Features

    - Action 'run': moves two blocks forward if possible; if blocked, moves only up to possible points.
    - Cup 'spillability': chance water in cup spills (becomes empty) when agent uses 'run' while carrying a filled cup.
    """

    # Workaround for environments lacking MiniGridEnv.Actions:
    class Actions:
        left = 0
        right = 1
        forward = 2
        pickup = 3
        drop = 4
        toggle = 5
        run = 6
        done = 7

    def __init__(self, min_height=3, max_height=6, max_steps=None, **kwargs):
        self.width = 12
        self.height = self._rand_int(min_height+2, max_height+3)
        self.num_cups = self.height - 3
        self.delivered = set()
        self.fountain_pos = None
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[],
        )
        if max_steps is None:
            max_steps = int(5 * self.width * self.height)
        super().__init__(
            mission_space=mission_space,
            width=self.width,
            height=self.height,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Pick up each cup, fill it at the fountain (left-center), and deliver to right wall."

    def _cell_in_bounds(self, x: int, y: int) -> bool:
        """MiniGrid ``Grid`` has no ``in_bounds``; keep checks aligned with ``grid.get`` assertions."""
        return 0 <= x < self.grid.width and 0 <= y < self.grid.height

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        # Walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        center_y = height // 2

        # Place the water fountain
        fountain_x, fountain_y = 1, center_y
        fountain = WaterFountain()
        self.put_obj(fountain, fountain_x, fountain_y)
        self.fountain_pos = (fountain_x, fountain_y)

        # Precompute the left wall cup positions: avoid the fountain cell itself!
        cup_spots = []
        # range for y: bottom to top
        for y in range(1, height-1):
            if y != center_y:
                cup_spots.append(y)
        assert len(cup_spots) == self.num_cups

        # Colors for cups
        cup_colors = [COLOR_NAMES[i % len(COLOR_NAMES)] for i in range(self.num_cups)]

        self.cup_positions = []
        # Sample a different spillability for each cup (range [0.05, 0.4])
        cup_spillabilities = self.np_random.uniform(low=0.05, high=0.4, size=self.num_cups)
        for i, y in enumerate(cup_spots):
            cup_x = 1
            spillability = float(cup_spillabilities[i])
            cup = Cup(cup_colors[i], filled=False, spillability=spillability)
            # Place the cup, and save its color for tracking purposes
            self.put_obj(cup, cup_x, y)
            self.cup_positions.append(((cup_x, y), cup_colors[i]))

        # Place agent, just to the right of fountain
        agent_y = fountain_y
        self.place_agent(top=(fountain_x+1, agent_y), size=(1,1), rand_dir=False) # right of fountain
        self.agent_dir = 2 # facing left
   
        self.right_wall_x = width - 2

        self.cups_delivered = [False]*self.num_cups
        self.cups_filled = [False]*self.num_cups
        self.cups_indices = {}  # maps (x,y) -> cup index for reward tracking

        for idx, (pos, _) in enumerate(self.cup_positions):
            self.cups_indices[pos] = idx

        self.mission = self._gen_mission()

    def step(self, action):
        # Custom step logic replacing super().step(action) for new action 'run'
        obs = None
        reward = 0
        terminated = False
        truncated = False
        info = {}

        step_action = action
        agent_moved = False

        carrying = self.carrying
        step_reward = 0

        # Actions:
        if step_action == self.Actions.left:
            self.agent_dir = (self.agent_dir - 1) % 4
        elif step_action == self.Actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4
        elif step_action == self.Actions.forward:
            fwd_pos = (self.agent_pos[0] + self.dir_vec[0], self.agent_pos[1] + self.dir_vec[1])
            if self._cell_in_bounds(*fwd_pos) and self.grid.get(*fwd_pos) is None:
                self.agent_pos = fwd_pos
                agent_moved = True
        elif step_action == self.Actions.run:
            # Try to move 2 steps forward or as far as possible (stopping at wall or object)
            steps_moved = 0
            for _ in range(2):
                fwd_pos = (self.agent_pos[0] + self.dir_vec[0], self.agent_pos[1] + self.dir_vec[1])
                if self._cell_in_bounds(*fwd_pos) and self.grid.get(*fwd_pos) is None:
                    self.agent_pos = fwd_pos
                    steps_moved += 1
                else:
                    break
            if steps_moved > 0:
                agent_moved = True
            # If agent is carrying a filled cup, maybe spill
            if self.carrying and isinstance(self.carrying, Cup):
                cup = self.carrying
                if cup.filled:
                    spill_prob = getattr(cup, 'spillability', 0.1)
                    rand_val = self.np_random.random()
                    if rand_val < spill_prob:
                        cup.spill()
        elif step_action == self.Actions.pickup:
            in_front_pos = (self.agent_pos[0] + self.dir_vec[0], self.agent_pos[1] + self.dir_vec[1])
            obj = (
                self.grid.get(*in_front_pos)
                if self._cell_in_bounds(*in_front_pos)
                else None
            )
            if obj and hasattr(obj, 'can_pickup') and obj.can_pickup() and self.carrying is None:
                self.carrying = obj
                self.grid.set(*in_front_pos, None)
        elif step_action == self.Actions.drop:
            drop_pos = (self.agent_pos[0] + self.dir_vec[0], self.agent_pos[1] + self.dir_vec[1])
            if self.carrying and self._cell_in_bounds(*drop_pos) and self.grid.get(*drop_pos) is None:
                self.grid.set(*drop_pos, self.carrying)
                self.carrying.cur_pos = drop_pos
                obj = self.carrying
                self.carrying = None
        elif step_action == self.Actions.toggle:
            in_front_pos = (self.agent_pos[0] + self.dir_vec[0], self.agent_pos[1] + self.dir_vec[1])
            obj = (
                self.grid.get(*in_front_pos)
                if self._cell_in_bounds(*in_front_pos)
                else None
            )
            # If toggle and at fountain, fill the cup if carrying
            if self.carrying and isinstance(self.carrying, Cup):
                # Only if agent is next to the fountain
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = self.agent_pos[0]+dx, self.agent_pos[1]+dy
                    if (nx, ny) == self.fountain_pos:
                        if not self.carrying.filled:
                            self.carrying.toggle(self, self.agent_pos)
                        break
        # (done is just a no-op)

        self.step_count += 1

        # Cup delivery reward check and delivery logic
        # If just dropped a cup, check if it's along the right wall and filled
        if step_action == self.Actions.drop and carrying and isinstance(carrying, Cup):
            drop_x, drop_y = self.agent_pos[0] + self.dir_vec[0], self.agent_pos[1] + self.dir_vec[1]
            if drop_x == self.right_wall_x and self._cell_in_bounds(drop_x, drop_y):
                obj = self.grid.get(drop_x, drop_y)
                idx = None
                # Find which cup it is (by position, color, or identity)
                for i, ((_, _), color) in enumerate(self.cup_positions):
                    # If cup colors and obj colors match and it's filled and not already delivered
                    if hasattr(obj, "color") and obj.color == color and getattr(obj, "filled", False) and not self.cups_delivered[i]:
                        idx = i
                        break
                if idx is not None:
                    self.cups_delivered[idx] = True
                    step_reward += 1 - 0.9 * (self.step_count / self.max_steps)

        # Terminate when all cups are delivered (ignore order)
        if all(self.cups_delivered):
            terminated = True

        if terminated:
            # Final reward is sum of delivered cups, step_count considered for each
            reward += sum(1 - 0.9 * (self.step_count / self.max_steps) for d in self.cups_delivered if d)
        else:
            reward += step_reward

        # Build observation
        obs = self.gen_obs()
        # Standard MiniGrid API expects obs, reward, terminated, truncated, info
        truncated = self.step_count >= self.max_steps
        if truncated:
            terminated = True
        return obs, reward, terminated, truncated, info