import numpy as np

from codefinityrl.challenges.mc.impls import _Decay
from codefinityrl.challenges.utils import (
    display_solution,
)
from codefinityrl.tests import test_case, test_case_context_var, TestFailure


def solution1():
    code = """
class Decay:
    def __init__(self, steps: int, high: float = 1, low: float = 0):
        self.steps = steps
        self.high = high
        self.low = low
        self.decay_rate = (high - low) / steps
        self.value = high

    def step(self):
        self.value = max(self.low, self.value - self.decay_rate)

    def __call__(self):
        return self.value
"""
    display_solution(code)


def _test_params(params: dict, user_decay_cls):
    user_decay = user_decay_cls(**params)
    correct_decay = _Decay(**params)

    if not np.isclose(user_decay(), correct_decay()):
        return False

    for _ in range(100):
        user_decay.step()
        correct_decay.step()
        if not np.isclose(user_decay(), correct_decay()):
            return False

    return True


@test_case("Correct! Here is the first part of the key: styOX")
def check1(user_decay_cls):
    test_case_context = test_case_context_var.get()

    params_1 = {"steps": 99, "high": 1, "low": 0}
    params_2 = {"steps": 99, "high": 0.5, "low": 0.5}
    params_3 = {"steps": 99, "high": 0.7, "low": 0.3}

    test_case_context.set_test(
        f"Decay works correctly for this configuration: {params_1}"
    )
    if not _test_params(params_1, user_decay_cls):
        raise TestFailure

    test_case_context.set_test(
        f"Decay works correctly for this configuration: {params_2}"
    )
    if not _test_params(params_2, user_decay_cls):
        raise TestFailure

    test_case_context.set_test(
        f"Decay works correctly for this configuration: {params_3}"
    )
    if not _test_params(params_3, user_decay_cls):
        raise TestFailure
