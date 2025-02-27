import numpy as np
import gymnasium as gym
from codefinityrl.challenges.utils import np_random_generator
from codefinityrl.math import softmax


class _EpsilonGreedyAgent:
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


class _UpperConfidenceBoundAgent:
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


class _GradientBanditsAgent:
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
        self.reward_avg = 0

        probs = softmax(self.H)

        self.H -= self.alpha * (reward - self.reward_avg) * probs
        self.H[action] += self.alpha * (reward - self.reward_avg)


def _test_agent(env_params: dict, agent_cls, agent_params: dict, test_envs: int):
    optimal_action_rate = np.zeros(env_params["max_episode_steps"])
    average_return = 0

    env = gym.make(**env_params)
    for test_case in range(test_envs):
        agent = agent_cls(**agent_params)
        env.reset(seed=test_case)

        t = 0
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action()
            _, reward, terminated, truncated, info = env.step(action)
            agent.update(action, reward)
            total_reward += reward

            if info["is_action_optimal"]:
                optimal_action_rate[t] += 1.0 / test_envs
            t += 1

            done = terminated or truncated
        average_return += total_reward / test_envs
    return optimal_action_rate, average_return
