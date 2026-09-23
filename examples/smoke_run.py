"""Small reproducibility smoke test for RC-Track*."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rc_track import GaussianRisingConcaveBandit, RCTrackStarLearner, run_episode


def main() -> None:
    T, K = 256, 5
    env = GaussianRisingConcaveBandit.from_exponential_curves(
        T=T,
        offsets=np.array([0.10, 0.08, 0.06, 0.04, 0.02]),
        amplitudes=np.array([0.35, 0.55, 0.40, 0.65, 0.50]),
        rates=np.array([0.030, 0.012, 0.020, 0.010, 0.016]),
        sigma=0.05,
        seed=7,
    )
    learner = RCTrackStarLearner(K=K, T=T, sigma=0.05, seed=7)
    result = run_episode(learner, env)
    print(f"Final pseudo-regret: {result['cumulative_pseudo_regret'][-1]:.4f}")
    print(f"Final epoch: {result['diagnostics'][-1]['epoch']}")
    print(f"Number of restarts: {sum(d['restart'] for d in result['diagnostics'])}")


if __name__ == "__main__":
    main()
