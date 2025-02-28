from codefinityrl.challenges.utils import display_hint, display_solution, display_check
from codefinityrl.challenges.mab.impls import _test_agent, _UpperConfidenceBoundAgent


def hint3():
    hint = """
These functions may help you to write an agent:
- `np.zeros`
- `np.argmax`
- `np.sqrt`
- `np.log`
"""
    return display_hint(hint)


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
    return average_return_correct == average_return_actual


def check3(agent_cls):
    env_params = {
        "id": "codefinityrl:MultiArmedBanditStationary-v0",
        "max_episode_steps": 1000,
        "n_arms": 10,
    }
    agent_params = {"n_arms": 10, "confidence": 0.2}
    agent = agent_cls(**agent_params)
    if not all(
        x is not None
        for x in [agent.n_arms, agent.confidence, agent.Q, agent.N, agent.t]
    ):
        display_check(False, "Some of object attributes are not initialized")
        return
    if not _test_params(env_params, agent_cls, agent_params):
        display_check(False, "Your algorithm works incorrectly")
        return
    display_check(True, "Correct! Here is the third part of the key: aFlt9")
