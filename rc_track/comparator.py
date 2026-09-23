from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


def dyadic_windows(T: int) -> list[tuple[int, int]]:
    """Return the paper's dyadic windows as zero-based half-open intervals.

    The manuscript uses one-based inclusive windows
        I_w = {2^{w-1}, ..., min(2^w-1, T)}.
    This function returns equivalent Python slices ``[start, end)``.
    """
    if T <= 0:
        raise ValueError("T must be positive")
    W = int(math.ceil(math.log2(T + 1)))
    windows: list[tuple[int, int]] = []
    for w in range(1, W + 1):
        start_1 = 2 ** (w - 1)
        end_1 = min(2**w - 1, T)
        if start_1 <= T:
            windows.append((start_1 - 1, end_1))
    return windows


def hysteresis_comparator(
    means: np.ndarray,
    deltas: Mapping[int, float] | Sequence[float] | float,
    *,
    tie_break: int | None = None,
) -> np.ndarray:
    """Construct the theoretical hysteresis comparator from the manuscript.

    Parameters
    ----------
    means:
        Array of shape ``(T, K)`` containing deterministic arm means.
    deltas:
        Threshold(s) indexed by one-based window number. A scalar applies to
        every window. A sequence must have length equal to the number of
        windows.
    tie_break:
        Optional fixed arm index used only when multiple means are tied.
        If omitted, NumPy's first-maximum rule is used.
    """
    mu = np.asarray(means, dtype=float)
    if mu.ndim != 2:
        raise ValueError("means must have shape (T, K)")
    T, K = mu.shape
    if T == 0 or K == 0:
        raise ValueError("means must be non-empty")
    windows = dyadic_windows(T)

    def delta_for_window(w: int) -> float:
        if np.isscalar(deltas):
            value = float(deltas)
        elif isinstance(deltas, Mapping):
            if w not in deltas:
                raise KeyError(f"Missing threshold delta_{w}")
            value = float(deltas[w])
        else:
            values = list(deltas)
            if len(values) != len(windows):
                raise ValueError("threshold sequence length must equal number of windows")
            value = float(values[w - 1])
        if value <= 0:
            raise ValueError("all thresholds must be positive")
        return value

    def argmax_row(row: np.ndarray) -> int:
        maxima = np.flatnonzero(row == np.max(row))
        if tie_break is not None and tie_break in maxima:
            return int(tie_break)
        return int(maxima[0])

    comparator = np.empty(T, dtype=int)
    for w, (start, end) in enumerate(windows, start=1):
        delta = delta_for_window(w)
        current = argmax_row(mu[start])
        comparator[start] = current
        for t in range(start, end - 1):
            gap_next = float(np.max(mu[t + 1]) - mu[t + 1, current])
            if gap_next <= 3.0 * delta:
                comparator[t + 1] = current
            else:
                current = argmax_row(mu[t + 1])
                comparator[t + 1] = current
    return comparator


def switch_count(sequence: Sequence[int]) -> int:
    seq = np.asarray(sequence)
    if seq.ndim != 1:
        raise ValueError("sequence must be one-dimensional")
    if len(seq) <= 1:
        return 0
    return int(np.sum(seq[1:] != seq[:-1]))
