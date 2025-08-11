import contextvars
import functools
import traceback
from codefinityrl.challenges.utils import display_check


class TestFailure(Exception):
    pass


class TestCaseContext:
    def __init__(self):
        self.successful_tests = []
        self.current_test = None

    def set_test(self, test: str):
        if self.current_test is not None:
            self.successful_tests.append(self.current_test)
        self.current_test = test


test_case_context_var = contextvars.ContextVar("test_case_context")


def test_case(success_message: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            test_case_context_var.set(TestCaseContext())
            try:
                test_case_context_var.get().set_test("Test initialized")
                func(*args, **kwargs)
            except TestFailure:
                test_case_context = test_case_context_var.get()
                for test in test_case_context.successful_tests:
                    display_check(True, test)
                display_check(False, test_case_context.current_test)
            except Exception:
                test_case_context = test_case_context_var.get()
                for test in test_case_context.successful_tests:
                    display_check(True, test)
                display_check(False, test_case_context.current_test)
                print(
                    f"Test failed with the following exception:\n{traceback.format_exc()}"
                )
            else:
                display_check(True, f"{success_message}")

        return wrapper

    return decorator
