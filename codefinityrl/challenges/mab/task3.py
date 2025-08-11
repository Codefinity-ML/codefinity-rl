import numpy as np

from codefinityrl.challenges.utils import display_solution
from codefinityrl.challenges.mab.impls import _test_agent, _UpperConfidenceBoundAgent
from codefinityrl.tests import test_case, test_case_context_var, TestFailure


def solution3():
    code = """
class UpperConfidenceBoundAgent:
    def __init__(self, n_arms: int, confidence: float):
        self.n_arms = n_arms
        self.confidence = confidence
        self.Q = np.zeros(self.n_arms)
        self.N = np.zeros(self.n_arms)
        self.t = 0

    def select_action(self):
        self.t += 1

        for action in range(self.n_arms):
            if self.N[action] == 0:
                return action

        return np.argmax(self.Q + self.confidence * np.sqrt(np.log(self.t) / self.N))

    def update(self, action: int, reward: float):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]
"""
    return display_solution(code)


def _test_params(env_params: dict, agent_cls, agent_params: dict):
    _, average_return_correct = _test_agent(
        env_params, _UpperConfidenceBoundAgent, agent_params, 10
    )
    _, average_return_actual = _test_agent(env_params, agent_cls, agent_params, 10)
    return np.isclose(average_return_correct, average_return_actual)


@test_case("Correct! Here is the third part of the key: aFlt9")
def check3(agent_cls):
    test_case_context = test_case_context_var.get()

    env_params = {
        "id": "codefinityrl:MultiArmedBanditStationary-v0",
        "max_episode_steps": 1000,
        "n_arms": 10,
    }
    agent_params = {"n_arms": 10, "confidence": 0.2}

    test_case_context.set_test("Algorithm works correctly")
    if not _test_params(env_params, agent_cls, agent_params):
        raise TestFailure
