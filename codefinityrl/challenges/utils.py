import numpy as np
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
