import gymnasium as gym
from codefinityrl.challenges.utils import (
    display_solution,
    display_check,
    value_dicts_close,
)

from codefinityrl.challenges.mc.impls import _OffPolicyMonteCarloControl, _Decay


def solution3():
    code = """
class OffPolicyMonteCarloControl:
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
        for e in range(episodes):
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
"""
    display_solution(code)


def check3(user_agent_cls):
    env = gym.make("codefinityrl:KeyAndChest-v0")

    user_agent = user_agent_cls(env)
    test_state = (1, 1, False)
    user_agent.init_state(test_state)

    if user_agent.policy[test_state] != 0:
        display_check(False, "init_state incorrectly initializes policy")
        return
    for a in range(env.action_space.n):
        if user_agent.values[(test_state, a)] != 0:
            display_check(False, "init_state incorrectly initializes values")
            return
        if user_agent.importances[(test_state, a)] != 0:
            display_check(False, "init_state incorrectly initializes importances")
            return

    user_agent.values[(test_state, 0)] = -5
    user_agent.values[(test_state, 1)] = 5
    user_agent.update_policy(test_state)

    if user_agent.policy[test_state] != 1:
        display_check(
            False, "update_policy does not pick the action with the highest value"
        )
        return

    user_agent.values[(test_state, 0)] = 5
    user_agent.update_policy(test_state)

    if user_agent.policy[test_state] != 0:
        display_check(
            False, "update_policy does not pick the first action with the highest value"
        )
        return

    if user_agent.get_target_action(test_state) != user_agent.policy[test_state]:
        display_check(False, "get_target_action must return greedy action")
        return

    if (
        user_agent.get_behavior_action(test_state, epsilon=0)
        != user_agent.policy[test_state]
    ):
        display_check(
            False, "get_behavior_action must return greedy action when epsilon is zero"
        )
        return

    random_actions = [
        user_agent.get_behavior_action(test_state, epsilon=1) for _ in range(100)
    ]
    if 1 not in random_actions or 2 not in random_actions or 3 not in random_actions:
        display_check(
            False,
            "get_behavior_action must return a random action with probability epsilon",
        )
        return

    user_agent = user_agent_cls(env)
    correct_agent = _OffPolicyMonteCarloControl(env)

    user_episode = user_agent.get_episode(epsilon=1)
    correct_episode = correct_agent.get_episode(epsilon=1)

    if len(user_episode) != len(correct_episode):
        display_check(False, "get_episode generates episodes incorrectly")
        return

    for i in range(len(correct_episode)):
        if user_episode[i] != correct_episode[i]:
            display_check(False, "get_episode generates episodes incorrectly")
            return

    user_agent = user_agent_cls(env)
    correct_agent = _OffPolicyMonteCarloControl(env)

    user_agent_eps = _Decay(200, 1, 0.8)
    correct_agent_eps = _Decay(200)

    user_agent.train(0.99, 200, user_agent_eps, 0.01, 1, 0.8)
    correct_agent.train(0.99, 200, correct_agent_eps, 0.01)

    if (
        not value_dicts_close(user_agent.values, correct_agent.values)
        or user_agent.policy != correct_agent.policy
    ):
        display_check(False, "Training loop is implemented incorrectly")
        return

    display_check(True, "Correct! Here is the next part of the key: 6g7EG")
