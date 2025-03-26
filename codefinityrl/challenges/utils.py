import io

import imageio
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Video
from IPython.display import display_markdown, display_html
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter


def display_hint(text: str):
    display_markdown(text, raw=True)


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


def plot_values(values_dict):
    env = gym.make("codefinityrl:KeyAndChest-v0", render_mode="rgb_array")
    env.reset()

    values = np.full((2, 7, 7), np.nan)
    for state in env.unwrapped.states():
        values[int(state[2]), state[1] - 1, state[0] - 1] = values_dict[state]
    for state in env.unwrapped.terminal_states():
        values[int(state[2]), state[1] - 1, state[0] - 1] = values_dict[state]

    vmin = np.min(values, where=~np.isnan(values), initial=np.inf)
    vmax = np.max(values, where=~np.isnan(values), initial=-np.inf)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sns.heatmap(
        values[0],
        ax=axes[0],
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
    axes[0].set_title("No key")

    sns.heatmap(
        values[1],
        ax=axes[1],
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
    axes[1].set_title("Key")

    axes[2].imshow(env.render())
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    plt.plot()


def plot_policy(policy_dict):
    env = gym.make("codefinityrl:KeyAndChest-v0", render_mode="rgb_array")
    env.reset()

    policy = np.full((2, 7, 7), np.nan)
    policy_annots = np.full((2, 7, 7), "")
    for state in env.unwrapped.states():
        policy[int(state[2]), state[1] - 1, state[0] - 1] = policy_dict[state]
        policy_annots[int(state[2]), state[1] - 1, state[0] - 1] = (
            env.unwrapped.action_descriptions[policy_dict[state]][0]
        )

    vmin = np.min(policy, where=~np.isnan(policy), initial=np.inf)
    vmax = np.max(policy, where=~np.isnan(policy), initial=-np.inf)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sns.heatmap(
        policy[0],
        ax=axes[0],
        vmin=vmin,
        vmax=vmax,
        annot=policy_annots[0],
        fmt="",
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        cmap=sns.color_palette("hls", 4),
    )
    axes[0].set_title("No key")

    sns.heatmap(
        policy[1],
        ax=axes[1],
        vmin=vmin,
        vmax=vmax,
        annot=policy_annots[1],
        fmt="",
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        cmap=sns.color_palette("hls", 4),
    )
    axes[1].set_title("Key")

    axes[2].imshow(env.render())
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    plt.plot()
