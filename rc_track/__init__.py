"""Reference implementation for RC-Track*.

The learner core follows Zhang (2026), Algorithm 1:
"Toward Optimal Switching Regret for Multi-Armed Bandits with Oblivious Adversary".
The reward clipping / loss transformation follows the RC-Track* manuscript.
"""

from .learner import RCTrackStarLearner
from .comparator import (
    dyadic_windows,
    hysteresis_comparator,
    switch_count,
)
from .environment import GaussianRisingConcaveBandit
from .runner import run_episode, run_oblivious_loss_table
from .metrics import best_switch_path, switching_regret
from .protocols import BanditEnvironment

__all__ = [
    "RCTrackStarLearner",
    "GaussianRisingConcaveBandit",
    "dyadic_windows",
    "hysteresis_comparator",
    "switch_count",
    "run_episode",
    "run_oblivious_loss_table",
    "best_switch_path",
    "switching_regret",
    "BanditEnvironment",
]
