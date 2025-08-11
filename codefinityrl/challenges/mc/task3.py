import gymnasium as gym
from codefinityrl.challenges.utils import (
    display_solution,
    value_dicts_close,
)

from codefinityrl.challenges.mc.impls import _OffPolicyMonteCarloControl, _Decay
from codefinityrl.tests import test_case, test_case_context_var, TestFailure


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
"""
    display_solution(code)


@test_case("Correct! Here is the third part of the key: 6g7EG")
def check3(user_agent_cls):
    test_case_context = test_case_context_var.get()

    env = gym.make("codefinityrl:KeyAndChest-v0")

    test_case_context.set_test("init_state correctly initializes policy")
    user_agent = user_agent_cls(env)
    test_state = (1, 1, False)
    user_agent.init_state(test_state)

    if user_agent.policy[test_state] != 0:
        raise TestFailure

    test_case_context.set_test("init_state correctly initializes values")
    for a in range(env.action_space.n):
        if user_agent.values[(test_state, a)] != 0:
            raise TestFailure

    test_case_context.set_test("init_state correctly initializes importances")
    for a in range(env.action_space.n):
        if user_agent.importances[(test_state, a)] != 0:
            raise TestFailure

    test_case_context.set_test("update_policy picks the action with the highest value")
    user_agent.values[(test_state, 0)] = -5
    user_agent.values[(test_state, 1)] = 5
    user_agent.update_policy(test_state)

    if user_agent.policy[test_state] != 1:
        raise TestFailure

    test_case_context.set_test(
        "update_policy picks the first action with the highest value"
    )
    user_agent.values[(test_state, 0)] = 5
    user_agent.update_policy(test_state)

    if user_agent.policy[test_state] != 0:
        raise TestFailure

    test_case_context.set_test("get_target_action returns greedy action")
    if user_agent.get_target_action(test_state) != user_agent.policy[test_state]:
        raise TestFailure

    test_case_context.set_test(
        "get_behavior_action returns greedy action when epsilon is zero"
    )
    if (
        user_agent.get_behavior_action(test_state, epsilon=0)
        != user_agent.policy[test_state]
    ):
        raise TestFailure

    test_case_context.set_test(
        "get_behavior_action returns a random action with probability epsilon"
    )
    random_actions = [
        user_agent.get_behavior_action(test_state, epsilon=1) for _ in range(1000)
    ]
    if 1 not in random_actions or 2 not in random_actions or 3 not in random_actions:
        raise TestFailure

    test_case_context.set_test("get_episode generates episodes correctly")
    user_agent = user_agent_cls(env)
    correct_agent = _OffPolicyMonteCarloControl(env)

    user_episode = user_agent.get_episode(epsilon=1)
    correct_episode = correct_agent.get_episode(epsilon=1)

    if len(user_episode) != len(correct_episode):
        raise TestFailure

    for i in range(len(correct_episode)):
        if user_episode[i] != correct_episode[i]:
            raise TestFailure

    test_case_context.set_test("Training loop is implemented correctly")
    user_agent = user_agent_cls(env)
    correct_agent = _OffPolicyMonteCarloControl(env)

    user_agent_eps = _Decay(200, 1, 0.8)
    correct_agent_eps = _Decay(200, 1, 0.8)

    user_agent.train(0.99, 200, user_agent_eps, 0.01)
    correct_agent.train(0.99, 200, correct_agent_eps, 0.01)

    if (
        not value_dicts_close(user_agent.values, correct_agent.values)
        or user_agent.policy != correct_agent.policy
    ):
        raise TestFailure
