import gymnasium as gym
from codefinityrl.spaces import Empty


class MultiArmedBanditStationaryEnv(gym.Env):
    def __init__(self, n_arms: int):
        self.observation_space = Empty()
        self.action_space = gym.spaces.Discrete(n_arms)

        self.n_arms = n_arms

        self.values = None

    def step(self, action: int):
        is_action_optimal = action == self.values.argmax()

        reward = self.values[action] + self.np_random.normal()
        return None, reward, False, False, {"is_action_optimal": is_action_optimal}

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.values = self.np_random.normal(size=self.n_arms)

        return None, {}


class MultiArmedBanditDynamicEnv(gym.Env):
    def __init__(self, n_arms: int, drift_interval: int):
        self.observation_space = Empty()
        self.action_space = gym.spaces.Discrete(n_arms)

        self.n_arms = n_arms
        self.drift_interval = drift_interval

        self.values = None
        self.values_drift = None
        self.t = None

    def step(self, action: int):
        is_action_optimal = action == self.values.argmax()

        self.t += 1
        if self.t % self.drift_interval == 0:
            self.values_drift = (
                self.np_random.normal(size=self.n_arms) - self.values
            ) / self.drift_interval

        reward = self.values[action] + self.np_random.normal()
        self.values += self.values_drift

        return None, reward, False, False, {"is_action_optimal": is_action_optimal}

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.values = self.np_random.normal(size=self.n_arms)
        self.values_drift = (
            self.np_random.normal(size=self.n_arms) - self.values
        ) / self.drift_interval
        self.t = 0

        return None, {}
