from __future__ import annotations

from typing import Sequence

import numpy as np


def best_switch_path(losses: np.ndarray, max_switches: int) -> tuple[np.ndarray, float]:
    """Compute an exact best comparator with at most ``max_switches`` switches.

    Parameters
    ----------
    losses:
        Deterministic loss table of shape ``(T, K)``.
    max_switches:
        Maximum number of arm changes allowed in the comparator.

    Returns
    -------
    path, value:
        A zero-based arm path and its cumulative loss. The dynamic program is
        exact and is intended for diagnostics/evaluation, not for the online
        learner itself.
    """
    ell = np.asarray(losses, dtype=float)
    if ell.ndim != 2 or ell.shape[0] == 0 or ell.shape[1] == 0:
        raise ValueError("losses must have shape (T, K) with T,K > 0")
    if not np.all(np.isfinite(ell)):
        raise ValueError("losses must contain only finite values")
    if np.any(ell < 0.0) or np.any(ell > 1.0):
        raise ValueError("losses must lie in [0, 1]")
    if not isinstance(max_switches, (int, np.integer)) or max_switches < 0:
        raise ValueError("max_switches must be a nonnegative integer")

    T, K = ell.shape
    S = min(int(max_switches), T - 1)

    # dp[s, k] = best loss through the current time, ending at arm k,
    # with exactly s switches. Backpointers store the previous switch count
    # and arm so the optimal path can be reconstructed.
    dp = np.full((S + 1, K), np.inf, dtype=float)
    dp[0] = ell[0]
    prev_s = np.full((T, S + 1, K), -1, dtype=np.int16)
    prev_k = np.full((T, S + 1, K), -1, dtype=np.int32)

    for t in range(1, T):
        new = np.full_like(dp, np.inf)
        for s in range(S + 1):
            for k in range(K):
                # Stay on k.
                stay = dp[s, k]
                best_value = stay
                best_prev_s = s
                best_prev_k = k

                # Switch into k from the best arm != k with one fewer switch.
                if s > 0:
                    row = dp[s - 1]
                    order = np.argsort(row)
                    for candidate in order:
                        candidate = int(candidate)
                        if candidate != k:
                            switch = row[candidate]
                            if switch < best_value:
                                best_value = switch
                                best_prev_s = s - 1
                                best_prev_k = candidate
                            break

                new[s, k] = best_value + ell[t, k]
                prev_s[t, s, k] = best_prev_s
                prev_k[t, s, k] = best_prev_k
        dp = new

    end_s, end_k = min(
        ((s, k) for s in range(S + 1) for k in range(K)),
        key=lambda pair: dp[pair],
    )
    value = float(dp[end_s, end_k])

    path = np.empty(T, dtype=int)
    s, k = int(end_s), int(end_k)
    path[-1] = k
    for t in range(T - 1, 0, -1):
        s_prev = int(prev_s[t, s, k])
        k_prev = int(prev_k[t, s, k])
        k = k_prev
        s = s_prev
        path[t - 1] = k

    return path, value


def switching_regret(
    losses: np.ndarray,
    actions: Sequence[int],
    max_switches: int,
) -> float:
    """Return regret against the best comparator with at most ``max_switches`` switches."""
    ell = np.asarray(losses, dtype=float)
    act = np.asarray(actions, dtype=int)
    if ell.ndim != 2 or act.ndim != 1 or act.shape[0] != ell.shape[0]:
        raise ValueError("losses/actions have incompatible shapes")
    if np.any(act < 0) or np.any(act >= ell.shape[1]):
        raise ValueError("actions contain an invalid arm index")
    learner_loss = float(np.sum(ell[np.arange(ell.shape[0]), act]))
    _, comparator_loss = best_switch_path(ell, max_switches)
    return learner_loss - comparator_loss
