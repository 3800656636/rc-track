import numpy as np

from rc_track import best_switch_path, switching_regret


def test_best_switch_path_zero_switches():
    losses = np.array(
        [
            [0.4, 0.2],
            [0.3, 0.1],
            [0.5, 0.2],
        ],
        dtype=float,
    )
    path, value = best_switch_path(losses, 0)
    assert np.all(path == 1)
    assert np.isclose(value, 0.5)


def test_best_switch_path_and_regret():
    losses = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )
    path, value = best_switch_path(losses, 1)
    assert np.array_equal(path, np.array([0, 1, 1]))
    assert np.isclose(value, 0.0)
    assert np.isclose(switching_regret(losses, [0, 0, 0], 1), 2.0)
