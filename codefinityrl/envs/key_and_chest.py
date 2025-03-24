import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image, ImageDraw


class KeyAndChestEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None):
        self.grid_size = 7
        self.action_space = spaces.Discrete(4)
        self.action_descriptions = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.observation_space = spaces.Tuple(
            (
                spaces.Discrete(self.grid_size, start=1),
                spaces.Discrete(self.grid_size, start=1),
                spaces.Discrete(2),
            )
        )

        self._walls = {
            (1, 1, "h"),
            (1, 2, "v"),
            (1, 5, "v"),
            (1, 6, "v"),
            (2, 1, "v"),
            (2, 2, "v"),
            (2, 4, "v"),
            (2, 5, "h"),
            (2, 6, "h"),
            (3, 1, "v"),
            (3, 4, "h"),
            (3, 4, "v"),
            (3, 5, "h"),
            (4, 3, "h"),
            (4, 4, "v"),
            (4, 5, "h"),
            (4, 6, "v"),
            (4, 7, "v"),
            (5, 1, "h"),
            (5, 2, "h"),
            (5, 2, "v"),
            (5, 3, "h"),
            (5, 4, "h"),
            (5, 5, "h"),
            (5, 6, "h"),
            (6, 3, "h"),
            (6, 3, "v"),
            (6, 4, "v"),
            (6, 5, "h"),
            (6, 6, "h"),
            (7, 2, "h"),
        }
        self._key_pos = (5, 7)
        self._chest_pos = (7, 1)

        self._agent_pos = None
        self._has_key = None

        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_pos = (1, 1)
        self._has_key = False

        return self._get_obs(), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.simulate_step(
            self._get_obs(), action
        )
        self._agent_pos = (obs[0], obs[1])
        self._has_key = obs[2]
        return obs, reward, terminated, truncated, info

    def simulate_step(self, state, action):
        x, y, key = state
        if action == 0 and ((x, y - 1, "h") not in self._walls):
            new_pos = (x, y - 1)
        elif action == 1 and ((x, y, "h") not in self._walls):
            new_pos = (x, y + 1)
        elif action == 2 and ((x - 1, y, "v") not in self._walls):
            new_pos = (x - 1, y)
        elif action == 3 and ((x, y, "v") not in self._walls):
            new_pos = (x + 1, y)
        else:
            new_pos = (x, y)

        agent_pos = (x, y)
        pos_changed = False
        if 0 < new_pos[0] < self.grid_size + 1 and 0 < new_pos[1] < self.grid_size + 1:
            agent_pos = new_pos
            pos_changed = True

        has_key = key
        if agent_pos == self._key_pos:
            has_key = True

        if agent_pos == self._chest_pos and has_key and pos_changed:
            return self._get_obs(agent_pos, has_key), 0, True, False, {}

        return self._get_obs(agent_pos, has_key), -1.0, False, False, {}

    def _get_obs(self, agent_pos=None, has_key=None):
        if agent_pos is None and has_key is None:
            agent_pos = self._agent_pos
            has_key = self._has_key
        return agent_pos[0], agent_pos[1], has_key

    def states(self):
        for i in range(1, self.grid_size + 1):
            for j in range(1, self.grid_size + 1):
                if (i, j) != self._key_pos:
                    yield self._get_obs((i, j), False)
                yield self._get_obs((i, j), True)

    def actions(self):
        for i in range(4):
            yield i

    def render(self):
        if self.render_mode == "rgb_array":
            img = Image.new(
                "RGB", (self.grid_size * 50 + 8, self.grid_size * 50 + 8), (0, 0, 0)
            )
            draw = ImageDraw.Draw(img)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    draw.rectangle(
                        [(1 + i * 51, 1 + j * 51), (50 + i * 51, 50 + j * 51)],
                        fill=(255, 255, 255),
                    )

            draw.rectangle(
                [(0, 0), (self.grid_size * 50 + 7, self.grid_size * 50 + 7)],
                outline=(0, 0, 0),
                width=3,
            )
            for wall in self._walls:
                if wall[2] == "h":
                    draw.line(
                        [
                            ((wall[0] - 1) * 51, wall[1] * 51),
                            (wall[0] * 51, wall[1] * 51),
                        ],
                        fill=(0, 0, 0),
                        width=5,
                    )
                elif wall[2] == "v":
                    draw.line(
                        [
                            (wall[0] * 51, (wall[1] - 1) * 51),
                            (wall[0] * 51, wall[1] * 51),
                        ],
                        fill=(0, 0, 0),
                        width=5,
                    )

            if not self._has_key:
                key_coords = (
                    3 + (self._key_pos[0] - 1) * 51,
                    3 + (self._key_pos[1] - 1) * 51,
                )
                draw.circle(
                    (key_coords[0] + 35, key_coords[1] + 10), 10, fill=(220, 220, 50)
                )
                draw.line(
                    [
                        (key_coords[0] + 35, key_coords[1] + 10),
                        (key_coords[0] + 1, key_coords[1] + 44),
                    ],
                    fill=(220, 220, 50),
                    width=5,
                )
                draw.line(
                    [
                        (key_coords[0] + 7, key_coords[1] + 38),
                        (key_coords[0] + 12, key_coords[1] + 43),
                    ],
                    fill=(220, 220, 50),
                    width=5,
                )
                draw.line(
                    [
                        (key_coords[0] + 12, key_coords[1] + 33),
                        (key_coords[0] + 17, key_coords[1] + 38),
                    ],
                    fill=(220, 220, 50),
                    width=5,
                )

            chest_coords = (
                3 + (self._chest_pos[0] - 1) * 51,
                3 + (self._chest_pos[1] - 1) * 51,
            )
            draw.rectangle(
                [
                    (chest_coords[0], chest_coords[1]),
                    (chest_coords[0] + 45, chest_coords[1] + 45),
                ],
                fill=(139, 69, 19),
                outline=(192, 192, 192),
                width=3,
            )
            draw.line(
                [
                    (chest_coords[0], chest_coords[1] + 15),
                    (chest_coords[0] + 45, chest_coords[1] + 15),
                ],
                fill=(192, 192, 192),
                width=3,
            )
            draw.rectangle(
                [
                    (chest_coords[0] + 18, chest_coords[1] + 17),
                    (chest_coords[0] + 27, chest_coords[1] + 30),
                ],
                fill=(192, 192, 192),
                width=3,
            )
            draw.circle(
                (chest_coords[0] + 23, chest_coords[1] + 22), 2.5, fill=(0, 0, 0)
            )
            draw.line(
                [
                    (chest_coords[0] + 22, chest_coords[1] + 22),
                    (chest_coords[0] + 22, chest_coords[1] + 27),
                ],
                fill=(0, 0, 0),
                width=2,
            )

            agent_coords = (
                3 + (self._agent_pos[0] - 1) * 51,
                3 + (self._agent_pos[1] - 1) * 51,
            )
            draw.rectangle(
                [
                    (agent_coords[0] + 19, agent_coords[1]),
                    (agent_coords[0] + 26, agent_coords[1] + 7),
                ],
                fill=(0, 0, 0),
            )
            draw.rectangle(
                [
                    (agent_coords[0] + 10, agent_coords[1] + 8),
                    (agent_coords[0] + 35, agent_coords[1] + 30),
                ],
                fill=(0, 0, 0),
            )
            draw.rectangle(
                [
                    (agent_coords[0] + 16, agent_coords[1] + 31),
                    (agent_coords[0] + 29, agent_coords[1] + 45),
                ],
                fill=(0, 0, 0),
            )

            return np.array(img)
