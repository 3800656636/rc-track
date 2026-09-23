from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def softmax_log_weights(log_weights: np.ndarray) -> np.ndarray:
    """Stable normalization of log-weights into a probability vector."""
    log_weights = np.asarray(log_weights, dtype=float)
    if log_weights.ndim != 1 or log_weights.size == 0:
        raise ValueError("log_weights must be a non-empty 1-D array")
    max_log = np.max(log_weights)
    shifted = np.exp(log_weights - max_log)
    total = float(np.sum(shifted))
    if not math.isfinite(total) or total <= 0.0:
        raise FloatingPointError("Could not normalize log-weights")
    return shifted / total


def categorical_sample(rng: np.random.Generator, probabilities: np.ndarray) -> int:
    """Sample an integer action from a probability vector."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("probabilities must be a non-empty 1-D array")
    total = float(np.sum(probabilities))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("probabilities must have a positive finite sum")
    p = probabilities / total
    # Guard against tiny negative roundoff after mixture operations.
    p = np.maximum(p, 0.0)
    p /= np.sum(p)
    return int(rng.choice(len(p), p=p))


def validate_probability_vector(p: np.ndarray, *, tol: float = 1e-10) -> None:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("probability vector must be a non-empty 1-D array")
    if not np.all(np.isfinite(p)):
        raise ValueError("probability vector contains non-finite values")
    if np.any(p < -tol):
        raise ValueError("probability vector contains a negative entry")
    if abs(float(np.sum(p)) - 1.0) > tol:
        raise ValueError("probability vector does not sum to one")


def canonical_dyadic_starts(t: int, horizon: int) -> Iterable[tuple[int, int, int]]:
    """Yield (start, end, length) for canonical dyadic intervals starting at t.

    The definition matches Zhang (2026), Eq. (4):
        J_{k,h} = {k 2^h + 1, ..., (k+1) 2^h}.
    """
    if not (1 <= t <= horizon):
        raise ValueError("t must be in {1, ..., horizon}")
    h = 0
    while True:
        length = 1 << h
        end = t + length - 1
        if end > horizon:
            break
        if (t - 1) % length == 0:
            yield t, end, length
        h += 1
