import numpy as np

from codefinityrl.challenges.utils import display_solution, display_check
from codefinityrl.challenges.mab.impls import _test_agent, _GradientBanditsAgent
from codefinityrl.math import softmax


def solution4():
    code = """
class GradientBanditsAgent:
    def __init__(self, n_arms: int, alpha: float, fixed_rng: bool = True):
        self.np_random = np_random_generator(fixed_rng)

        self.n_arms = n_arms
        self.alpha = alpha
        self.H = np.zeros(n_arms)
        self.reward_avg = 0
        self.t = 0

    def select_action(self):
        probs = softmax(self.H)
        return self.np_random.choice(self.n_arms, p=probs)

    def update(self, action: int, reward: float):
        self.t += 1
        self.reward_avg += (reward - self.reward_avg) / self.t

        probs = softmax(self.H)

        self.H -= self.alpha * (reward - self.reward_avg) * probs
        self.H[action] += self.alpha * (reward - self.reward_avg)
"""
    return display_solution(code)


def _test_params(env_params: dict, agent_cls, agent_params: dict):
    _, average_return_correct = _test_agent(
        env_params, _GradientBanditsAgent, agent_params, 10
    )
    _, average_return_actual = _test_agent(env_params, agent_cls, agent_params, 10)
    return np.isclose(average_return_correct, average_return_actual)


def check4(user_softmax, agent_cls):
    env_params = {
        "id": "codefinityrl:MultiArmedBanditStationary-v0",
        "max_episode_steps": 1000,
        "n_arms": 10,
    }
    agent_params = {"n_arms": 10, "alpha": 0.2}
    agent = agent_cls(**agent_params)
    if not np.array_equal(
        user_softmax(np.array([1, 2, 3])), softmax(np.array([1, 2, 3]))
    ):
        display_check(False, "Your softmax implementation is incorrect")
    if not all(
        x is not None
        for x in [agent.n_arms, agent.alpha, agent.H, agent.reward_avg, agent.t]
    ):
        display_check(False, "Some of object attributes are not initialized")
        return
    if not _test_params(env_params, agent_cls, agent_params):
        display_check(False, "Your algorithm works incorrectly")
        return
    display_check(True, "Correct! Here is the fourth part of the key: 5pLxz")
