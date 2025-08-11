import gymnasium as gym
from codefinityrl.challenges.utils import (
    display_solution,
    value_dicts_close,
)
from codefinityrl.challenges.dp.impls import _ValueIterationAgent
from codefinityrl.tests import test_case, test_case_context_var, TestFailure


def solution3():
    code = """
class ValueIterationAgent:
    def __init__(self, env: gym.Env):
        self.states = env.unwrapped.states
        self.terminal_states = env.unwrapped.terminal_states
        self.actions = env.unwrapped.actions
        self.model = env.unwrapped.simulate_step

        self.policy = {}
        self.values = {}
        for state in self.states():
            self.values[state] = 0
        for state in self.terminal_states():
            self.values[state] = 0

    def evaluate_policy(self, theta: float, gamma: float):
        delta = np.inf
        while delta > theta:
            delta = 0
            for state in self.states():
                v = self.values[state]
                max_q = -np.inf
                for action in self.actions():
                    next_state, reward, _, _, _ = self.model(state, action)
                    max_q = max(max_q, reward + gamma * self.values[next_state])
                self.values[state] = max_q
                delta = max(delta, abs(v - max_q))

    def improve_policy(self, gamma: float):
        for state in self.states():
            best_action = None
            best_action_value = -np.inf
            for action in self.actions():
                next_state, reward, _, _, _ = self.model(state, action)
                action_value = reward + gamma * self.values[next_state]
                if best_action_value < action_value:
                    best_action = action
                    best_action_value = action_value
            self.policy[state] = best_action

    def train(self, theta: float, gamma: float):
        self.evaluate_policy(theta, gamma)
        self.improve_policy(gamma)

    def __call__(self, state):
        return self.policy[state]
"""
    display_solution(code)


@test_case("Correct! Here is the third part of the key: 5pEdG")
def check3(user_agent_cls):
    test_case_context = test_case_context_var.get()

    env = gym.make("codefinityrl:KeyAndChest-v0")

    user_agent = user_agent_cls(env)
    correct_agent = _ValueIterationAgent(env)

    test_case_context.set_test("Policy evaluation is implemented correctly")
    user_agent.evaluate_policy(theta=1e-6, gamma=0.99)
    correct_agent.evaluate_policy(theta=1e-6, gamma=0.99)

    if not value_dicts_close(user_agent.values, correct_agent.values):
        raise TestFailure

    test_case_context.set_test("Policy improvement is implemented correctly")
    user_agent.improve_policy(gamma=0.99)
    correct_agent.improve_policy(gamma=0.99)

    if user_agent.policy != correct_agent.policy:
        raise TestFailure

    test_case_context.set_test("Training is implemented correctly")
    user_agent = user_agent_cls(env)
    correct_agent = _ValueIterationAgent(env)

    user_agent.train(theta=1e-6, gamma=0.99)
    correct_agent.train(theta=1e-6, gamma=0.99)

    if (
        not value_dicts_close(user_agent.values, correct_agent.values)
        or user_agent.policy != correct_agent.policy
    ):
        raise TestFailure
