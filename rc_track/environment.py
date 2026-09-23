from __future__ import annotations

import numpy as np


class GaussianRisingConcaveBandit:
    """Synthetic restless rising-concave bandit environment.

    The mean curve is generated as
        mu_i(t) = offset_i + amplitude_i * (1 - exp(-rate_i * t)),
    with nonnegative offsets/amplitudes chosen so that means stay in [0, 1].
    Its increments are nonnegative and non-increasing, so it satisfies the
    rising-concave structural assumptions used by RC-Track*.
    """

    def __init__(
        self,
        means: np.ndarray,
        sigma: float,
        seed: int | None = None,
    ):
        self.means = np.asarray(means, dtype=float)
        if self.means.ndim != 2:
            raise ValueError("means must have shape (T, K)")
        if np.any(self.means < 0.0) or np.any(self.means > 1.0):
            raise ValueError("means must lie in [0, 1]")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)
        self.t = 0

    @property
    def T(self) -> int:
        return self.means.shape[0]

    @property
    def K(self) -> int:
        return self.means.shape[1]

    def reset(self) -> None:
        self.t = 0

    def pull(self, action: int) -> float:
        if not (0 <= action < self.K):
            raise ValueError("action out of range")
        if self.t >= self.T:
            raise RuntimeError("environment horizon exhausted")
        reward = float(self.means[self.t, action] + self.rng.normal(0.0, self.sigma))
        self.t += 1
        return reward

    @classmethod
    def from_exponential_curves(
        cls,
        T: int,
        offsets: np.ndarray,
        amplitudes: np.ndarray,
        rates: np.ndarray,
        sigma: float,
        seed: int | None = None,
    ) -> "GaussianRisingConcaveBandit":
        offsets = np.asarray(offsets, dtype=float)
        amplitudes = np.asarray(amplitudes, dtype=float)
        rates = np.asarray(rates, dtype=float)
        if offsets.ndim != amplitudes.ndim or offsets.ndim != rates.ndim or offsets.ndim != 1:
            raise ValueError("offsets, amplitudes, and rates must be one-dimensional")
        if not (len(offsets) == len(amplitudes) == len(rates)):
            raise ValueError("curve parameter arrays must have equal length")
        if T <= 0:
            raise ValueError("T must be positive")
        if np.any(offsets < 0) or np.any(amplitudes < 0) or np.any(rates <= 0):
            raise ValueError("offsets/amplitudes must be nonnegative and rates positive")
        if np.any(offsets + amplitudes > 1.0 + 1e-12):
            raise ValueError("offset_i + amplitude_i must be at most one")

        t = np.arange(1, T + 1, dtype=float)[:, None]
        means = offsets[None, :] + amplitudes[None, :] * (1.0 - np.exp(-rates[None, :] * t))
        return cls(means=means, sigma=sigma, seed=seed)
