from codefinityrl.challenges.utils import display_solution
from codefinityrl.envs import MultiArmedBanditStationaryEnv, MultiArmedBanditDynamicEnv
from codefinityrl.tests import test_case, test_case_context_var, TestFailure


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


@test_case("Correct! Here is the first part of the key: WKVkH")
def check1(env_stationary, env_dynamic):
    test_case_context = test_case_context_var.get()

    test_case_context.set_test("Stationary environment created")
    if not (
        hasattr(env_stationary, "unwrapped")
        and isinstance(env_stationary.unwrapped, MultiArmedBanditStationaryEnv)
    ):
        raise TestFailure

    test_case_context.set_test("Dynamic environment created")
    if not (
        hasattr(env_dynamic, "unwrapped")
        and isinstance(env_dynamic.unwrapped, MultiArmedBanditDynamicEnv)
    ):
        raise TestFailure
