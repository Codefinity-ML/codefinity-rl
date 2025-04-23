import numpy as np
import gymnasium as gym
from codefinityrl.challenges import np_random_generator


class _SARSA:
    def __init__(self, env: gym.Env):
        self.np_random = np_random_generator(True)

        self.env = env

        self.policy = {}
        self.values = {}
        self.counts = {}

    def init_state(self, state):
        if state not in self.policy:
            self.policy[state] = 0
            for action in range(self.env.action_space.n):
                self.values[(state, action)] = 0
                self.counts[(state, action)] = 0

    def update_policy(self, state):
        best_action = None
        best_action_value = -np.inf
        for action in range(self.env.action_space.n):
            if best_action_value < self.values[(state, action)]:
                best_action_value = self.values[(state, action)]
                best_action = action
        self.policy[state] = best_action

    def get_action(self, state, epsilon: float = 0):
        if self.np_random.random() < epsilon:
            return self.np_random.integers(self.env.action_space.n)
        else:
            return self.policy[state]

    def train(self, gamma: float, episodes: int, epsilon: float, alpha: float):
        for e in range(episodes):
            state, _ = self.env.reset()
            self.init_state(state)

            action = self.get_action(state, epsilon)

            done = False
            while not done:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.init_state(next_state)
                next_action = self.get_action(next_state, epsilon)
                self.values[(state, action)] += alpha * (
                    reward
                    + gamma * self.values[(next_state, next_action)]
                    - self.values[(state, action)]
                )

                self.update_policy(state)

                state = next_state
                action = next_action
                done = terminated or truncated

    def __call__(self, state):
        if state not in self.policy:
            return self.np_random.integers(self.env.action_space.n)
        return self.get_action(state)


class _QLearning:
    def __init__(self, env: gym.Env):
        self.np_random = np_random_generator(True)

        self.env = env

        self.policy = {}
        self.values = {}
        self.counts = {}

    def init_state(self, state):
        if state not in self.policy:
            self.policy[state] = 0
            for action in range(self.env.action_space.n):
                self.values[(state, action)] = 0
                self.counts[(state, action)] = 0

    def update_policy(self, state):
        best_action = None
        best_action_value = -np.inf
        for action in range(self.env.action_space.n):
            if best_action_value < self.values[(state, action)]:
                best_action_value = self.values[(state, action)]
                best_action = action
        self.policy[state] = best_action

    def get_action(self, state, epsilon: float = 0):
        if self.np_random.random() < epsilon:
            return self.np_random.integers(self.env.action_space.n)
        else:
            return self.policy[state]

    def train(self, gamma: float, episodes: int, epsilon: float, alpha: float):
        for e in range(episodes):
            state, _ = self.env.reset()
            self.init_state(state)

            done = False
            while not done:
                action = self.get_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.init_state(next_state)
                next_action = self.policy[next_state]
                self.values[(state, action)] += alpha * (
                    reward
                    + gamma * self.values[(next_state, next_action)]
                    - self.values[(state, action)]
                )

                self.update_policy(state)

                state = next_state
                done = terminated or truncated

    def __call__(self, state):
        if state not in self.policy:
            return self.np_random.integers(self.env.action_space.n)
        return self.get_action(state)
