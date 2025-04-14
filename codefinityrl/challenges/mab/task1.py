from codefinityrl.challenges.utils import display_solution, display_check
from codefinityrl.envs import MultiArmedBanditStationaryEnv, MultiArmedBanditDynamicEnv


def solution1():
    code = """
env_stationary = gym.make(
    "codefinityrl:MultiArmedBanditStationary-v0", 
    max_episode_steps=1000, 
    n_arms=10
)

env_dynamic = gym.make(
    "codefinityrl:MultiArmedBanditDynamic-v0", 
    max_episode_steps=1000, 
    n_arms=10, 
    drift_interval=500
)
"""
    display_solution(code)


def check1(env_stationary, env_dynamic):
    if not (
        hasattr(env_stationary, "unwrapped")
        and isinstance(env_stationary.unwrapped, MultiArmedBanditStationaryEnv)
    ):
        display_check(False, "You did not create a stationary environment")
    elif not (
        hasattr(env_dynamic, "unwrapped")
        and isinstance(env_dynamic.unwrapped, MultiArmedBanditDynamicEnv)
    ):
        display_check(False, "You did not create a dynamic environment")
    else:
        display_check(True, "Correct! Here is the first part of the key: WKVkH")
