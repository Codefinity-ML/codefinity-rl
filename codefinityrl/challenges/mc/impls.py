import numpy as np
import gymnasium as gym
from codefinityrl.challenges import np_random_generator


class _Decay:
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


class _OnPolicyMonteCarloControl:
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

    def get_episode(self, epsilon: float):
        episode = []
        state, _ = self.env.reset()

        done = False
        while not done:
            self.init_state(state)
            action = self.get_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            episode.append((state, action, reward))

            state = next_state
            done = terminated or truncated

        return episode

    def train(self, gamma: float, episodes: int, epsilon):
        for _ in range(episodes):
            episode = self.get_episode(epsilon())

            G = 0
            for state, action, reward in reversed(episode):
                G = reward + gamma * G

                self.counts[(state, action)] += 1
                self.values[(state, action)] = self.values[(state, action)] + (
                    1.0 / self.counts[(state, action)]
                ) * (G - self.values[(state, action)])

                self.update_policy(state)

            epsilon.step()

    def __call__(self, state):
        if state not in self.policy:
            return self.np_random.integers(self.env.action_space.n)
        return self.get_action(state)


class _OffPolicyMonteCarloControl:
    def __init__(self, env: gym.Env):
        self.np_random = np_random_generator(True)

        self.env = env

        self.policy = {}
        self.values = {}
        self.importances = {}

    def init_state(self, state):
        if state not in self.policy:
            self.policy[state] = 0
            for action in range(self.env.action_space.n):
                self.values[(state, action)] = 0
                self.importances[(state, action)] = 0

    def update_policy(self, state):
        best_action = None
        best_action_value = -np.inf
        for action in range(self.env.action_space.n):
            if best_action_value < self.values[(state, action)]:
                best_action_value = self.values[(state, action)]
                best_action = action
        self.policy[state] = best_action

    def get_target_action(self, state):
        return self.policy[state]

    def get_behavior_action(self, state, epsilon: float):
        if self.np_random.random() < epsilon:
            return self.np_random.integers(self.env.action_space.n)
        else:
            return self.policy[state]

    def get_episode(self, epsilon: float):
        episode = []
        state, _ = self.env.reset()

        done = False
        while not done:
            self.init_state(state)
            action = self.get_behavior_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            episode.append((state, action, reward))

            state = next_state
            done = terminated or truncated

        return episode

    def train(self, gamma: float, episodes: int, epsilon, target_epsilon: float):
        for _ in range(episodes):
            episode = self.get_episode(epsilon())

            G = 0
            W = 1
            for state, action, reward in reversed(episode):
                G = reward + gamma * G
                self.importances[(state, action)] += W
                self.values[(state, action)] = self.values[(state, action)] + (
                    W / self.importances[(state, action)]
                ) * (G - self.values[(state, action)])

                self.update_policy(state)

                if action == self.policy:
                    W *= (
                        1 - target_epsilon + target_epsilon / self.env.action_space.n
                    ) / (1 - epsilon() + epsilon() / self.env.action_space.n)
                else:
                    W *= (target_epsilon / self.env.action_space.n) / (
                        epsilon() / self.env.action_space.n
                    )
                if W == 0:
                    break

            epsilon.step()

    def __call__(self, state):
        if state not in self.policy:
            return self.np_random.integers(self.env.action_space.n)
        return self.get_target_action(state)
