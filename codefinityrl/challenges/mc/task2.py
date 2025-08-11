import gymnasium as gym
from codefinityrl.challenges.utils import (
    display_solution,
    value_dicts_close,
)

from codefinityrl.challenges.mc.impls import _OnPolicyMonteCarloControl, _Decay
from codefinityrl.tests import test_case, test_case_context_var, TestFailure


def solution2():
    code = """
class OnPolicyMonteCarloControl:
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
"""
    display_solution(code)


@test_case("Correct! Here is the second part of the key: LZbET")
def check2(user_agent_cls):
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

    test_case_context.set_test("init_state correctly initializes counts")
    for a in range(env.action_space.n):
        if user_agent.counts[(test_state, a)] != 0:
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

    test_case_context.set_test("get_action returns greedy action when epsilon is zero")
    if user_agent.get_action(test_state, epsilon=0) != user_agent.policy[test_state]:
        raise TestFailure

    test_case_context.set_test(
        "get_action returns a random action with probability epsilon"
    )
    random_actions = [user_agent.get_action(test_state, epsilon=1) for _ in range(1000)]
    if 1 not in random_actions or 2 not in random_actions or 3 not in random_actions:
        raise TestFailure

    test_case_context.set_test("get_episode generates episodes correctly")
    user_agent = user_agent_cls(env)
    correct_agent = _OnPolicyMonteCarloControl(env)

    user_episode = user_agent.get_episode(epsilon=1)
    correct_episode = correct_agent.get_episode(epsilon=1)

    if len(user_episode) != len(correct_episode):
        raise TestFailure

    for i in range(len(correct_episode)):
        if user_episode[i] != correct_episode[i]:
            raise TestFailure

    test_case_context.set_test("Training loop is implemented correctly")
    user_agent = user_agent_cls(env)
    correct_agent = _OnPolicyMonteCarloControl(env)

    user_agent_eps = _Decay(200, 1, 0.8)
    correct_agent_eps = _Decay(200, 1, 0.8)

    user_agent.train(0.99, 200, user_agent_eps)
    correct_agent.train(0.99, 200, correct_agent_eps)

    if (
        not value_dicts_close(user_agent.values, correct_agent.values)
        or user_agent.policy != correct_agent.policy
    ):
        raise TestFailure
