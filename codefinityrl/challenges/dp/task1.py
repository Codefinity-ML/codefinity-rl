import gymnasium as gym
from codefinityrl.challenges.utils import display_solution
from codefinityrl.tests import test_case, test_case_context_var, TestFailure


def solution1():
    code = """
solution_actions = [3, 1, 1, 1, 1, 3, 3, 3, 3, 3, 1, 1, 2, 2, 3, 3, 0, 0, 2, 2, 2, 2, 2, 0, 0, 3, 3, 3, 3, 0, 0, 3]
"""
    display_solution(code)


@test_case("Correct! Here is the first part of the key: pSUS2")
def check1(solution_actions):
    test_case_context = test_case_context_var.get()

    env = gym.make("codefinityrl:KeyAndChest-v0")
    env.reset()

    test_case_context.set_test("The environment is solved")
    is_done = False
    done = False
    for action in solution_actions:
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        is_done = is_done or done

    if not is_done:
        raise TestFailure

    test_case_context.set_test("No moves after the environment is solved")
    if not done:
        raise TestFailure
