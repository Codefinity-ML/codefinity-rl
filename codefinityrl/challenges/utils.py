import io

import imageio
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from IPython.display import display, Video
from IPython.display import display_html
from matplotlib.patches import Polygon
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from seaborn.utils import relative_luminance


def display_solution(code: str):
    lexer = get_lexer_by_name("python", stripall=True)
    formatter = HtmlFormatter(style="monokai")
    style = f"""<style>{formatter.get_style_defs(".output_html")}</style>"""
    display_html(style + highlight(code, lexer, formatter), raw=True)


def display_check(correct: bool, text: str):
    display_html(
        f"""<span style="color: {"green" if correct else "red"}">{text}</span>""",
        raw=True,
    )


def value_dicts_close(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False

    for key in dict1:
        val1, val2 = dict1[key], dict2[key]
        if not np.isclose(val1, val2):
            return False

    return True


def np_random_generator(fixed_rng: bool) -> np.random.Generator:
    if fixed_rng:
        return np.random.default_rng(seed=42)
    else:
        return np.random.default_rng()


class VideoRecord:
    def __init__(self, fps: int):
        self.fps = fps
        self.frames = []

    def add_frame(self, frame: np.ndarray):
        self.frames.append(frame)

    def play(self):
        buffer = io.BytesIO()
        imageio.mimwrite(
            buffer, self.frames, format="mp4", fps=self.fps, macro_block_size=1
        )
        buffer.seek(0)
        display(
            Video(
                buffer.read(),
                embed=True,
                mimetype="video/mp4",
                html_attributes="controls autoplay",
            )
        )


def add_walls(ax, walls):
    for wall in walls:
        i, j, orientation = wall
        if orientation == "h":
            ax.plot([i - 1, i], [j, j], color="black", linewidth=1)
        elif orientation == "v":
            ax.plot([i, i], [j - 1, j], color="black", linewidth=1)


def plot_values(values_dict):
    if len(next(iter(values_dict))) == 3:
        plot_state_values(values_dict)
    else:
        plot_action_values(values_dict)


def plot_state_values(values_dict):
    env = gym.make("codefinityrl:KeyAndChest-v0", render_mode="rgb_array")
    env.reset()

    values = np.full((2, 7, 7), np.nan)
    for x, y, key in values_dict.keys():
        values[int(key), y - 1, x - 1] = values_dict[(x, y, key)]

    vmin = np.min(values, where=~np.isnan(values), initial=np.inf)
    vmax = np.max(values, where=~np.isnan(values), initial=-np.inf)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for key, title in zip([0, 1], ["No key", "Key"]):
        axes[key].set_aspect("equal")

        for x in range(8):
            axes[key].axvline(x, color="lightgray", linewidth=1)
        for y in range(8):
            axes[key].axhline(y, color="lightgray", linewidth=1)

        sns.heatmap(
            values[key],
            ax=axes[key],
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".1f",
            square=True,
            xticklabels=False,
            yticklabels=False,
            cbar=False,
            cmap="coolwarm",
        )

        add_walls(axes[key], env.unwrapped._walls)
        axes[key].set_title(title)
        axes[key].set_xlim(0, 7)
        axes[key].set_ylim(0, 7)
        axes[key].invert_yaxis()
        axes[key].set_xticks([])
        axes[key].set_yticks([])
        axes[key].axis("off")

    axes[2].imshow(env.render())
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    plt.show()


def plot_action_values(values_dict):
    env = gym.make("codefinityrl:KeyAndChest-v0", render_mode="rgb_array")
    env.reset()

    values = np.full((2, 7, 7, 4), np.nan)
    for (x, y, key), action in values_dict.keys():
        values[int(key), y - 1, x - 1, action] = values_dict[((x, y, key), action)]

    vmin = np.min(values, where=~np.isnan(values), initial=np.inf)
    vmax = np.max(values, where=~np.isnan(values), initial=-np.inf)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for key, title in zip([0, 1], ["No key", "Key"]):
        axes[key].set_aspect("equal")

        for x in range(8):
            axes[key].axvline(x, color="lightgray", linewidth=1)
        for y in range(8):
            axes[key].axhline(y, color="lightgray", linewidth=1)

        for y in range(7):
            for x in range(7):
                center = (x + 0.5, y + 0.5)
                triangles = {
                    0: [(x, y), (x + 1, y), center],
                    1: [(x, y + 1), (x + 1, y + 1), center],
                    2: [(x, y + 1), (x, y), center],
                    3: [(x + 1, y + 1), (x + 1, y), center],
                }

                for action, vertices in triangles.items():
                    if not np.isnan(values[key, y, x, action]):
                        val = values[key, y, x, action]
                        color = cmap(norm(val))
                        lum = relative_luminance(color)
                        text_color = "0.15" if lum > 0.408 else "w"
                        poly = Polygon(vertices, facecolor=color)
                        axes[key].add_patch(poly)
                        cx = sum(v[0] for v in vertices) / 3
                        cy = sum(v[1] for v in vertices) / 3
                        axes[key].text(
                            cx,
                            cy,
                            f"{val:.1f}",
                            color=text_color,
                            ha="center",
                            va="center",
                            fontsize=6,
                        )

        add_walls(axes[key], env.unwrapped._walls)
        axes[key].set_title(title)
        axes[key].set_xlim(0, 7)
        axes[key].set_ylim(0, 7)
        axes[key].invert_yaxis()
        axes[key].set_xticks([])
        axes[key].set_yticks([])
        axes[key].axis("off")

    axes[2].imshow(env.render())
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    plt.tight_layout()
    plt.show()


def plot_policy(policy_dict):
    env = gym.make("codefinityrl:KeyAndChest-v0", render_mode="rgb_array")
    env.reset()

    policy = np.full((2, 7, 7), np.nan)
    for x, y, key in policy_dict.keys():
        policy[int(key), y - 1, x - 1] = policy_dict[(x, y, key)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    arrow_deltas = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0)}

    for key, title in zip([0, 1], ["No key", "Key"]):
        axes[key].set_aspect("equal")

        for x in range(8):
            axes[key].axvline(x, color="lightgray", linewidth=1)
        for y in range(8):
            axes[key].axhline(y, color="lightgray", linewidth=1)

        for y in range(7):
            for x in range(7):
                action = policy[key, y, x]
                if not np.isnan(action):
                    cx, cy = x + 0.5, y + 0.5
                    dx, dy = arrow_deltas[action]
                    axes[key].annotate(
                        "",
                        xy=(cx + dx, cy + dy),
                        xytext=(cx - dx, cy - dy),
                        arrowprops=dict(arrowstyle="->", lw=1.5),
                        ha="center",
                        va="center",
                    )

        add_walls(axes[key], env.unwrapped._walls)
        axes[key].set_title(title)
        axes[key].set_xlim(0, 7)
        axes[key].set_ylim(0, 7)
        axes[key].invert_yaxis()
        axes[key].set_xticks([])
        axes[key].set_yticks([])
        axes[key].axis("off")

    axes[2].imshow(env.render())
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    plt.tight_layout()
    plt.show()
