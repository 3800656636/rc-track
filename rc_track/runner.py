from __future__ import annotations

from typing import Any

import numpy as np

from .learner import RCTrackStarLearner


def run_episode(
    learner: RCTrackStarLearner,
    environment: Any,
) -> dict[str, Any]:
    """Run one complete bandit episode.

    The environment must expose ``T``, ``K``, ``means`` and ``pull(action)``.
    The learner only receives the reward of the sampled arm; the mean table is
    used only by this helper to compute diagnostic pseudo-regret.
    """
    if learner.K != environment.K or learner.T != environment.T:
        raise ValueError("learner and environment dimensions must match")
    means = np.asarray(environment.means, dtype=float)
    if means.shape != (learner.T, learner.K):
        raise ValueError("environment.means has the wrong shape")

    actions = np.empty(learner.T, dtype=int)
    rewards = np.empty(learner.T, dtype=float)
    losses = np.empty(learner.T, dtype=float)
    expected_rewards = np.empty(learner.T, dtype=float)
    pseudo_regret = np.empty(learner.T, dtype=float)
    branches: list[str] = []
    diagnostics: list[dict[str, Any]] = []

    for t in range(learner.T):
        action, _p, branch = learner.begin_round()
        reward = environment.pull(action)
        loss = learner.reward_to_loss(reward)
        info = learner.observe(loss)

        actions[t] = action
        rewards[t] = reward
        losses[t] = loss
        expected_rewards[t] = means[t, action]
        pseudo_regret[t] = float(np.max(means[t]) - means[t, action])
        branches.append(branch)
        diagnostics.append(info)

    return {
        "actions": actions,
        "rewards": rewards,
        "losses": losses,
        "expected_rewards": expected_rewards,
        "pseudo_regret": pseudo_regret,
        "cumulative_pseudo_regret": np.cumsum(pseudo_regret),
        "branches": branches,
        "diagnostics": diagnostics,
    }


def run_oblivious_loss_table(
    learner: RCTrackStarLearner,
    losses: np.ndarray,
) -> dict[str, Any]:
    """Run the learner against a deterministic oblivious loss table.

    This is the direct diagnostic interface for the switching-bandit setting
    of Zhang (2026). The learner only receives the loss of the sampled arm.
    """
    table = np.asarray(losses, dtype=float)
    if table.shape != (learner.T, learner.K):
        raise ValueError("losses must have shape (T, K)")
    if not np.all(np.isfinite(table)):
        raise ValueError("losses must contain only finite values")
    if np.any(table < 0.0) or np.any(table > 1.0):
        raise ValueError("losses must lie in [0, 1]")

    actions = np.empty(learner.T, dtype=int)
    sampled_losses = np.empty(learner.T, dtype=float)
    branches: list[str] = []
    probabilities = np.empty((learner.T, learner.K), dtype=float)
    diagnostics: list[dict[str, Any]] = []

    for t in range(learner.T):
        action, probability, branch = learner.begin_round()
        loss = float(table[t, action])
        info = learner.observe(loss)
        actions[t] = action
        sampled_losses[t] = loss
        probabilities[t] = probability
        branches.append(branch)
        diagnostics.append(info)

    return {
        "actions": actions,
        "losses": sampled_losses,
        "cumulative_loss": np.cumsum(sampled_losses),
        "probabilities": probabilities,
        "branches": branches,
        "diagnostics": diagnostics,
    }
