import gymnasium as gym
from codefinityrl.challenges.utils import display_solution, display_check


def solution1():
    code = """
solution_actions = [3, 1, 1, 1, 1, 3, 3, 3, 3, 3, 1, 1, 2, 2, 3, 3, 0, 0, 2, 2, 2, 2, 2, 0, 0, 3, 3, 3, 3, 0, 0, 3]
"""
    display_solution(code)


def check1(solution_actions):
    env = gym.make("codefinityrl:KeyAndChest-v0")
    env.reset()

    is_done = False
    done = False
    for action in solution_actions:
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        is_done = is_done or done
    if not is_done:
        display_check(False, "You didn't solve the environment")
    if not done:
        display_check(False, "You unnecessarily moved after solving the environment")
    if done:
        display_check(True, "Correct! Here is the first part of the key: pSUS2")
