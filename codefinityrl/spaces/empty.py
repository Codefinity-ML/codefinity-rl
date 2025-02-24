from typing import Any
from gymnasium.spaces import Space


class Empty(Space):
    """Empty observation space

    Used for stateless environments.
    """

    def __init__(self):
        super().__init__()

    def sample(self, mask: None = None, probability: None = None):
        return None

    def contains(self, x: Any) -> bool:
        if x is None:
            return True
        else:
            return False
