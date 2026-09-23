from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._utils import (
    canonical_dyadic_starts,
    categorical_sample,
    softmax_log_weights,
    validate_probability_vector,
)


@dataclass
class _IntervalState:
    start: int
    end: int
    eta: float
    x: np.ndarray  # arm coordinates; missing mass is retained on the main learner

    @property
    def arm_mass(self) -> float:
        return float(np.sum(self.x))

    @property
    def main_mass(self) -> float:
        return float(max(0.0, 1.0 - self.arm_mass))


class RCTrackStarLearner:
    """Bandit learner used by RC-Track*.

    This class implements the *online* part of RC-Track*: the universal
    switching-bandit learner from Zhang (2026), Algorithm 1, run on the
    clipped-loss transformation from the manuscript.

    The hysteresis comparator is a theoretical comparator and is therefore
    implemented separately in :mod:`rc_track.comparator`.

    Parameters
    ----------
    K:
        Number of arms.
    T:
        Known horizon.
    sigma:
        Sub-Gaussian scale used by the manuscript's clipping threshold.
    seed:
        Optional NumPy RNG seed.
    """

    def __init__(self, K: int, T: int, sigma: float, seed: int | None = None):
        if K < 2:
            raise ValueError("K must be at least 2")
        if T <= K:
            raise ValueError("The manuscript assumes T > K")
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        self.K = int(K)
        self.T = int(T)
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)

        self.B_T = 1.0 + self.sigma * np.sqrt(
            8.0 * np.log(max(2.0, 2.0 * self.K * self.T))
        )
        self.L = int(np.ceil(20.0 * np.log(max(2.0, 2.0 * self.K * self.T))))
        self.alpha = 1.0 / (100.0 * self.L**2)
        self.Q = 1000.0

        # Zhang (2026), Algorithm 1.
        self.eta_1 = 100.0 * self.L / np.sqrt(self.K * self.T)
        self.epoch = 1
        self.eta = self.eta_1
        self.credit = 0.0

        self.q = np.full(self.K, 1.0 / self.K, dtype=float)
        self._active: list[_IntervalState] = []
        self.t = 1
        self._pending: dict[str, Any] | None = None
        self.last_action: int | None = None
        self.last_probability: np.ndarray | None = None
        self.last_branch: str | None = None

    @property
    def active_intervals(self) -> tuple[tuple[int, int], ...]:
        """Active canonical intervals as ``(start, end)`` pairs."""
        return tuple((s.start, s.end) for s in self._active)

    @property
    def epoch_learning_rate(self) -> float:
        return self.eta

    @property
    def active_interval_count(self) -> int:
        return len(self._active)

    def clip_reward(self, reward: float) -> float:
        """Clip one observed reward according to the manuscript."""
        return float(np.clip(reward, -self.B_T, self.B_T))

    def reward_to_loss(self, reward: float) -> float:
        """Map one observed reward to the bounded loss in [0, 1]."""
        y = self.clip_reward(reward)
        loss = 1.0 - (y + self.B_T) / (2.0 * self.B_T)
        # Protect against only floating-point endpoint spillover.
        return float(np.clip(loss, 0.0, 1.0))

    def _launch_intervals(self) -> None:
        for start, end, _length in canonical_dyadic_starts(self.t, self.T):
            u = float(self.rng.random())
            # NumPy may return exactly zero; the theoretical distribution has
            # probability zero at that point, so use the smallest positive
            # representable float only for numerical robustness.
            u = max(u, np.finfo(float).tiny)
            interval_eta = self.alpha * self.eta / u
            x = np.full(self.K, 1.0 / (self.K * (self.T + 1)), dtype=float)
            self._active.append(
                _IntervalState(start=start, end=end, eta=interval_eta, x=x)
            )

    def _challenge_distribution(self) -> np.ndarray:
        if not self._active:
            return self.q.copy()
        arm_mix = np.zeros(self.K, dtype=float)
        total_arm_mass = 0.0
        for state in self._active:
            arm_mix += state.x
            total_arm_mass += state.arm_mass
        p = (1.0 - self.alpha * total_arm_mass) * self.q + self.alpha * arm_mix
        p = np.maximum(p, 0.0)
        p /= np.sum(p)
        return p

    def begin_round(self) -> tuple[int, np.ndarray, str]:
        """Sample the action for the current round.

        Returns
        -------
        action, probability, branch
            ``action`` is zero-based. ``branch`` is either ``"main"`` or
            ``"challenge"``.
        """
        if self.t > self.T:
            raise RuntimeError("The horizon has already ended")
        if self._pending is not None:
            raise RuntimeError("observe() must be called before begin_round()")

        self._launch_intervals()
        main_round = bool(self.rng.random() < 0.5)
        if main_round:
            p = self.q.copy()
            branch = "main"
        else:
            p = self._challenge_distribution()
            branch = "challenge"

        validate_probability_vector(p)
        action = categorical_sample(self.rng, p)
        self._pending = {"p": p, "action": action, "branch": branch}
        self.last_action = action
        self.last_probability = p.copy()
        self.last_branch = branch
        return action, p.copy(), branch

    def _update_main(self, loss: float, action: int, p: np.ndarray) -> None:
        q_action = float(self.q[action])
        if q_action <= 0.0:
            raise FloatingPointError("Main distribution assigned zero mass")
        estimated = np.zeros(self.K, dtype=float)
        estimated[action] = 2.0 * loss / q_action
        log_weights = np.log(self.q) - self.eta * estimated
        q_tilde = softmax_log_weights(log_weights)
        self.q = (1.0 - 1.0 / self.T) * q_tilde + 1.0 / (self.K * self.T)
        self.q /= np.sum(self.q)

    def _update_interval(self, state: _IntervalState, loss: float, action: int, p: np.ndarray) -> None:
        denominator = float(p[action] + state.eta)
        if denominator <= 0.0:
            raise FloatingPointError("Invalid implicit-exploration denominator")
        estimated = np.zeros(self.K, dtype=float)
        estimated[action] = loss / denominator
        baseline = float(np.dot(self.q, estimated))
        z = estimated - baseline

        # Completed vector is (main-mass, arm masses), and the main-mass
        # coordinate has zero loss difference. This is the closed-form
        # entropy-OMD update used by Zhang (2026).
        completed = np.concatenate(([state.main_mass], state.x))
        gradient = np.concatenate(([0.0], z))
        log_weights = np.full(self.K + 1, -np.inf, dtype=float)
        positive = completed > 0.0
        log_weights[positive] = np.log(completed[positive]) - state.eta * gradient[positive]
        updated = softmax_log_weights(log_weights)
        state.x = updated[1:]

        # Numerical cleanup; the completed mass remains a probability vector.
        total = float(np.sum(state.x))
        if total > 1.0:
            state.x /= total * (1.0 + 1e-15)

    def _finish_round(self, restart: bool) -> None:
        if restart:
            self.epoch += 1
            self.eta = 2.0 * self.eta
            self.q.fill(1.0 / self.K)
            self.credit = 0.0
            self._active.clear()
        else:
            self._active = [state for state in self._active if state.end != self.t]
        self.t += 1

    def act(self) -> tuple[int, np.ndarray, str]:
        """Alias for :meth:`begin_round` for conventional bandit code."""
        return self.begin_round()

    def update(self, loss: float) -> dict[str, Any]:
        """Alias for :meth:`observe` for conventional bandit code."""
        return self.observe(loss)

    def observe(self, loss: float) -> dict[str, Any]:
        """Provide the selected loss after :meth:`begin_round`.

        Returns a dictionary with diagnostics useful for experiments.
        """
        if self._pending is None:
            raise RuntimeError("begin_round() must be called before observe()")
        if not np.isfinite(loss) or not (0.0 <= float(loss) <= 1.0):
            raise ValueError("loss must be finite and lie in [0, 1]")

        pending = self._pending
        self._pending = None
        p = pending["p"]
        action = int(pending["action"])
        branch = pending["branch"]

        z_t = 0.0
        if branch == "main":
            self._update_main(float(loss), action, p)
        else:
            self.q = self.q.copy()  # explicit: frozen on challenge rounds
            for state in self._active:
                self._update_interval(state, float(loss), action, p)
            z_t = float(loss) * (self.q[action] / p[action] - 1.0)
            self.credit += z_t

        restart = self.credit >= self.Q * self.K * self.T * self.eta and self.t < self.T
        diagnostics = {
            "t": self.t,
            "action": action,
            "probability": p.copy(),
            "branch": branch,
            "loss": float(loss),
            "credit_increment": z_t,
            "credit": float(self.credit),
            "epoch": self.epoch,
            "eta": float(self.eta),
            "active_intervals": self.active_interval_count,
            "restart": bool(restart),
        }
        self._finish_round(restart=restart)
        return diagnostics

    def step_loss(self, loss: float) -> tuple[int, dict[str, Any]]:
        """Convenience API: sample an action and feed back its bounded loss."""
        action, p, branch = self.begin_round()
        diagnostics = self.observe(loss)
        diagnostics["probability"] = p
        diagnostics["branch"] = branch
        return action, diagnostics

    def step_reward(self, reward: float) -> tuple[int, dict[str, Any]]:
        """Convenience API: sample an action and feed back an observed reward."""
        loss = self.reward_to_loss(reward)
        return self.step_loss(loss)
