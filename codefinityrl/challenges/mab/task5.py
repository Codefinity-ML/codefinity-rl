import numpy as np

from codefinityrl.challenges.utils import display_solution
from codefinityrl.challenges.mab.impls import (
    _test_agent,
    _EpsilonGreedyAgent,
    _UpperConfidenceBoundAgent,
    _GradientBanditsAgent,
)
from codefinityrl.tests import test_case, test_case_context_var, TestFailure


def solution5():
    code = """
def test_agent(env_params: dict, agent_cls, agent_params: dict, test_envs: int):
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
"""
    return display_solution(code)


def _test_params(user_test_agent, env_params: dict, agent_cls, agent_params: dict):
    optimal_action_rate_correct, average_return_correct = _test_agent(
        env_params, agent_cls, agent_params, 10
    )
    optimal_action_rate_actual, average_return_actual = user_test_agent(
        env_params, agent_cls, agent_params, 10
    )
    return np.all(
        np.isclose(optimal_action_rate_correct, optimal_action_rate_actual)
    ), np.isclose(average_return_correct, average_return_actual)


@test_case("Correct! Here is the fifth part of the key: ouePl")
def check5(user_test_agent):
    test_case_context = test_case_context_var.get()

    env_params_1 = {
        "id": "codefinityrl:MultiArmedBanditStationary-v0",
        "max_episode_steps": 1000,
        "n_arms": 10,
    }
    env_params_2 = {
        "id": "codefinityrl:MultiArmedBanditDynamic-v0",
        "max_episode_steps": 5000,
        "n_arms": 10,
        "drift_interval": 500,
    }
    agent_params_1 = {"n_arms": 10, "epsilon": 0.2, "alpha": 0.1, "optimistic": 1}
    agent_params_2 = {"n_arms": 10, "confidence": 0.2}
    agent_params_3 = {"n_arms": 10, "alpha": 0.2}

    test_case_context.set_test(
        "Your test returns correct optimal_action_rate for this config: stationary environment, epsilon-greedy agent"
    )
    oar, ar = _test_params(
        user_test_agent, env_params_1, _EpsilonGreedyAgent, agent_params_1
    )
    if not oar:
        raise TestFailure

    test_case_context.set_test(
        "Your test returns correct average_return for this config: stationary environment, epsilon-greedy agent"
    )
    if not ar:
        raise TestFailure

    test_case_context.set_test(
        "Your test returns correct optimal_action_rate for this config: stationary environment, UCB agent"
    )
    oar, ar = _test_params(
        user_test_agent, env_params_1, _UpperConfidenceBoundAgent, agent_params_2
    )
    if not oar:
        raise TestFailure

    test_case_context.set_test(
        "Your test returns correct average_return for this config: stationary environment, UCB agent"
    )
    if not ar:
        raise TestFailure

    test_case_context.set_test(
        "Your test returns correct optimal_action_rate for this config: stationary environment, gradient bandits agent"
    )
    oar, ar = _test_params(
        user_test_agent, env_params_1, _GradientBanditsAgent, agent_params_3
    )
    if not oar:
        raise TestFailure

    test_case_context.set_test(
        "Your test returns correct average_return for this config: stationary environment, gradient bandits agent"
    )
    if not ar:
        raise TestFailure

    test_case_context.set_test(
        "Your test returns correct optimal_action_rate for this config: dynamic environment, epsilon-greedy agent"
    )
    oar, ar = _test_params(
        user_test_agent, env_params_2, _EpsilonGreedyAgent, agent_params_1
    )
    if not oar:
        raise TestFailure

    test_case_context.set_test(
        "Your test returns correct average_return for this config: dynamic environment, epsilon-greedy agent"
    )
    if not ar:
        raise TestFailure
