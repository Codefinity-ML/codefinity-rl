import numpy as np

from codefinityrl.challenges.mab.impls import _test_agent, _EpsilonGreedyAgent
from codefinityrl.challenges.utils import display_solution, display_check


def solution2():
    code = """
class EpsilonGreedyAgent:
    def __init__(
        self,
        n_arms: int,
        epsilon: float,
        alpha: float | str = "mean",
        optimistic: float = 0,
        fixed_rng: bool = True,
    ):
        self.np_random = np_random_generator(fixed_rng)

        self.n_arms = n_arms
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = np.full(self.n_arms, optimistic, dtype=np.float64)
        if alpha == "mean":
            self.N = np.zeros(self.n_arms)

    def select_action(self):
        if self.np_random.uniform() < self.epsilon:
            return self.np_random.integers(self.n_arms)
        else:
            return np.argmax(self.Q)

    def update(self, action: int, reward: float):
        if self.alpha == "mean":
            self.N[action] += 1
            self.Q[action] += (reward - self.Q[action]) / self.N[action]
        else:
            self.Q[action] += self.alpha * (reward - self.Q[action])
"""
    return display_solution(code)


def _test_params(env_params: dict, agent_cls, agent_params: dict):
    _, average_return_correct = _test_agent(
        env_params, _EpsilonGreedyAgent, agent_params, 10
    )
    _, average_return_actual = _test_agent(env_params, agent_cls, agent_params, 10)
    return np.isclose(average_return_correct, average_return_actual)


def check2(agent_cls):
    env_params = {
        "id": "codefinityrl:MultiArmedBanditStationary-v0",
        "max_episode_steps": 1000,
        "n_arms": 10,
    }
    agent_params_1 = {"n_arms": 10, "epsilon": 0.2}
    agent_params_2 = {"n_arms": 10, "epsilon": 0.2, "alpha": 0.1}
    agent_params_3 = {"n_arms": 10, "epsilon": 0.2, "alpha": 0.1, "optimistic": 1}
    agent = agent_cls(**agent_params_1)
    if not all(
        x is not None
        for x in [agent.n_arms, agent.epsilon, agent.alpha, agent.Q, agent.N]
    ):
        display_check(False, "Some of object attributes are not initialized")
        return
    if not _test_params(env_params, agent_cls, agent_params_1):
        display_check(
            False,
            "Your algorithm works incorrectly with mean estimates for action values",
        )
        return
    if not _test_params(env_params, agent_cls, agent_params_2):
        display_check(False, "Your algorithm works incorrectly with constant step size")
        return
    if not _test_params(env_params, agent_cls, agent_params_3):
        display_check(False, "Your algorithm works incorrectly with optimisic values")
        return
    display_check(True, "Correct! Here is the second part of the key: uivu0")
