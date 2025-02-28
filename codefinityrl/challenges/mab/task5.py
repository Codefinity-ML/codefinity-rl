import numpy as np

from codefinityrl.challenges.utils import display_hint, display_solution, display_check
from codefinityrl.challenges.mab.impls import (
    _test_agent,
    _EpsilonGreedyAgent,
    _UpperConfidenceBoundAgent,
    _GradientBanditsAgent,
)


def hint5():
    hint = """
`env_params` has `"max_episode_steps"` key, use it to initialize `optimal_action_rate` with `np.zeros`.
You should also use dictionary unpacking to avoid writing a lot of code
"""
    return display_hint(hint)


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
    return np.array_equal(
        optimal_action_rate_correct, optimal_action_rate_actual
    ), average_return_correct == average_return_actual


def check5(user_test_agent):
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

    oar, ar = _test_params(
        user_test_agent, env_params_1, _EpsilonGreedyAgent, agent_params_1
    )
    if not oar or not ar:
        display_check(
            False,
            f"""Your test returns incorrect {"optimal_action_rate" if not oar else ""}{" and " if not oar and not ar else ""}{"average_return" if not ar else ""} for this config: stationary environment, epsilon-greedy agent""",
        )
        return
    oar, ar = _test_params(
        user_test_agent, env_params_1, _UpperConfidenceBoundAgent, agent_params_2
    )
    if not oar or not ar:
        display_check(
            False,
            f"""Your test returns incorrect {"optimal_action_rate" if not oar else ""}{" and " if not oar and not ar else ""}{"average_return" if not ar else ""} for this config: stationary environment, UCB agent""",
        )
        return
    oar, ar = _test_params(
        user_test_agent, env_params_1, _GradientBanditsAgent, agent_params_3
    )
    if not oar or not ar:
        display_check(
            False,
            f"""Your test returns incorrect {"optimal_action_rate" if not oar else ""}{" and " if not oar and not ar else ""}{"average_return" if not ar else ""} for this config: stationary environment, gradient bandits agent""",
        )
        return
    oar, ar = _test_params(
        user_test_agent, env_params_2, _EpsilonGreedyAgent, agent_params_1
    )
    if not oar or not ar:
        display_check(
            False,
            f"""Your test returns incorrect {"optimal_action_rate" if not oar else ""}{" and " if not oar and not ar else ""}{"average_return" if not ar else ""} for this config: dynamic environment, epsilon-greedy agent""",
        )
        return
    display_check(True, "Correct! Here is the fifth part of the key: ouePl")
