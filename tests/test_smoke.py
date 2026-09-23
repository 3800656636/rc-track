import numpy as np

from rc_track import (
    GaussianRisingConcaveBandit,
    RCTrackStarLearner,
    dyadic_windows,
    hysteresis_comparator,
    run_episode,
    switch_count,
)


def test_dyadic_windows_cover_horizon():
    for T in range(2, 65):
        windows = dyadic_windows(T)
        covered = []
        for s, e in windows:
            covered.extend(range(s, e))
        assert covered == list(range(T))


def test_hysteresis_comparator_valid():
    T, K = 16, 3
    means = np.zeros((T, K), dtype=float)
    means[:, 0] = np.linspace(0.1, 0.2, T)
    means[:, 1] = np.linspace(0.1, 0.5, T)
    means[:, 2] = np.linspace(0.05, 0.15, T)
    u = hysteresis_comparator(means, 0.01)
    assert u.shape == (T,)
    assert np.all((0 <= u) & (u < K))
    assert 0 <= switch_count(u) <= T - 1


def test_learner_probability_and_episode():
    T, K = 32, 4
    means = np.full((T, K), 0.2)
    env = GaussianRisingConcaveBandit(means, sigma=0.02, seed=1)
    learner = RCTrackStarLearner(K, T, sigma=0.02, seed=1)
    result = run_episode(learner, env)
    assert result["actions"].shape == (T,)
    assert np.isfinite(result["rewards"]).all()
    for d in result["diagnostics"]:
        p = np.asarray(d["probability"])
        assert np.isfinite(p).all()
        assert np.isclose(np.sum(p), 1.0)
        assert np.all(p >= 0.0)
