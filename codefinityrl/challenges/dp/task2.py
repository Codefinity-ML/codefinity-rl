import gymnasium as gym
from codefinityrl.challenges.utils import (
    display_hint,
    display_solution,
    display_check,
    value_dicts_close,
)
from codefinityrl.challenges.dp.impls import _PolicyIterationAgent


def hint2():
    hint = """
Hints:
- To implement policy evaluation loop, you may want to initialize delta
to np.inf before the while loop;
- For policy improvement, argmax is better implemented by an inner
loop over actions. To find the best action, use additional variables to
store the action value and the current best action.
"""
    display_hint(hint)


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


def check2(user_agent_cls):
    env = gym.make("codefinityrl:KeyAndChest-v0")

    user_agent = user_agent_cls(env)
    correct_agent = _PolicyIterationAgent(env)

    is_optimal = False
    while not is_optimal:
        user_agent.evaluate_policy(theta=1e-6, gamma=0.99)
        correct_agent.evaluate_policy(theta=1e-6, gamma=0.99)

        if not value_dicts_close(user_agent.values, correct_agent.values):
            display_check(False, "Policy evaluation is implemented incorrectly")
            return

        is_optimal_user = user_agent.improve_policy(gamma=0.99)
        is_optimal = correct_agent.improve_policy(gamma=0.99)

        if user_agent.policy != correct_agent.policy:
            display_check(False, "Policy improvement is implemented incorrectly")
            return

        if is_optimal_user != is_optimal:
            display_check(
                False,
                "Policy improvement doesn't identify correctly if current policy is optimal",
            )
            return

    user_agent = user_agent_cls(env)
    correct_agent = _PolicyIterationAgent(env)

    user_agent.train(theta=1e-6, gamma=0.99)
    correct_agent.train(theta=1e-6, gamma=0.99)

    if (
        not value_dicts_close(user_agent.values, correct_agent.values)
        or user_agent.policy != correct_agent.policy
    ):
        display_check(False, "Training loop is implemented incorrectly")
        return

    display_check(True, "Correct! Here is the second part of the key: U2Iq0")
