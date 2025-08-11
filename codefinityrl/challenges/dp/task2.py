import types

import gymnasium as gym
from codefinityrl.challenges.utils import (
    display_solution,
    value_dicts_close,
)
from codefinityrl.challenges.dp.impls import _PolicyIterationAgent
from codefinityrl.tests import test_case, test_case_context_var, TestFailure


def solution2():
    code = """
class PolicyIterationAgent:
    def __init__(self, env: gym.Env):
        self.states = env.unwrapped.states
        self.terminal_states = env.unwrapped.terminal_states
        self.actions = env.unwrapped.actions
        self.model = env.unwrapped.simulate_step

        self.policy = {}
        self.values = {}
        for state in self.states():
            self.values[state] = 0
            self.policy[state] = 0
        for state in self.terminal_states():
            self.values[state] = 0

    def evaluate_policy(self, theta: float, gamma: float):
        delta = np.inf
        while delta > theta:
            delta = 0
            for state in self.states():
                v = self.values[state]
                next_state, reward, _, _, _ = self.model(state, self.policy[state])
                self.values[state] = reward + gamma * self.values[next_state]
                delta = max(delta, abs(v - self.values[state]))

    def improve_policy(self, gamma: float) -> bool:
        is_policy_stable = True
        for state in self.states():
            a = self.policy[state]
            best_action = None
            best_action_value = -np.inf
            for action in self.actions():
                next_state, reward, _, _, _ = self.model(state, action)
                action_value = reward + gamma * self.values[next_state]
                if best_action_value < action_value:
                    best_action = action
                    best_action_value = action_value
            self.policy[state] = best_action
            if best_action != a:
                is_policy_stable = False
        return is_policy_stable

    def train(self, theta: float, gamma: float):
        is_policy_stable = False
        while not is_policy_stable:
            self.evaluate_policy(theta, gamma)
            is_policy_stable = self.improve_policy(gamma)

    def __call__(self, state):
        return self.policy[state]
"""
    display_solution(code)


@test_case("Correct! Here is the second part of the key: U2Iq0")
def check2(user_agent_cls):
    test_case_context = test_case_context_var.get()

    env = gym.make("codefinityrl:KeyAndChest-v0")

    test_case_context.set_test("Policy evaluation is implemented correctly")
    user_agent = _PolicyIterationAgent(env)
    user_agent.evaluate_policy = types.MethodType(
        user_agent_cls.evaluate_policy, user_agent
    )
    correct_agent = _PolicyIterationAgent(env)

    is_optimal = False
    while not is_optimal:
        user_agent.evaluate_policy(theta=1e-6, gamma=0.99)
        correct_agent.evaluate_policy(theta=1e-6, gamma=0.99)

        if not value_dicts_close(user_agent.values, correct_agent.values):
            raise TestFailure

        user_agent.improve_policy(gamma=0.99)
        is_optimal = correct_agent.improve_policy(gamma=0.99)

    test_case_context.set_test("Policy improvement is implemented correctly")
    user_agent = _PolicyIterationAgent(env)
    user_agent.improve_policy = types.MethodType(
        user_agent_cls.improve_policy, user_agent
    )
    correct_agent = _PolicyIterationAgent(env)

    is_optimal = False
    while not is_optimal:
        user_agent.evaluate_policy(theta=1e-6, gamma=0.99)
        correct_agent.evaluate_policy(theta=1e-6, gamma=0.99)

        user_agent.improve_policy(gamma=0.99)
        is_optimal = correct_agent.improve_policy(gamma=0.99)

        if user_agent.policy != correct_agent.policy:
            raise TestFailure

    test_case_context.set_test(
        "Policy improvement correctly identifies if policy is optimal"
    )
    user_agent = _PolicyIterationAgent(env)
    user_agent.improve_policy = types.MethodType(
        user_agent_cls.improve_policy, user_agent
    )
    correct_agent = _PolicyIterationAgent(env)

    while not is_optimal:
        user_agent.evaluate_policy(theta=1e-6, gamma=0.99)
        correct_agent.evaluate_policy(theta=1e-6, gamma=0.99)

        is_optimal_user = user_agent.improve_policy(gamma=0.99)
        is_optimal = correct_agent.improve_policy(gamma=0.99)

        if is_optimal_user != is_optimal:
            raise TestFailure

    test_case_context.set_test("Training loop is implemented correctly")
    user_agent = user_agent_cls(env)
    correct_agent = _PolicyIterationAgent(env)

    user_agent.train(theta=1e-6, gamma=0.99)
    correct_agent.train(theta=1e-6, gamma=0.99)

    if (
        not value_dicts_close(user_agent.values, correct_agent.values)
        or user_agent.policy != correct_agent.policy
    ):
        raise TestFailure
