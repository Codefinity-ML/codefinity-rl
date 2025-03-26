import numpy as np
import gymnasium as gym


class _PolicyIterationAgent:
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


class _ValueIterationAgent:
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
