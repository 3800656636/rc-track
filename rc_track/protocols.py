from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BanditEnvironment(Protocol):
    """Minimal environment interface expected by the experiment runner."""

    T: int
    K: int

    def pull(self, action: int) -> float:
        """Return the reward observed after selecting ``action``."""
        ...
