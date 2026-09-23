"""Small deterministic-loss sanity check for the switching learner."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rc_track import RCTrackStarLearner, run_oblivious_loss_table, switching_regret


T, K = 2000, 5
rng = np.random.default_rng(0)
losses = rng.uniform(0.2, 0.8, size=(T, K))
losses[:700, 0] -= 0.15
losses[700:1400, 2] -= 0.15
losses[1400:, 4] -= 0.15
losses = np.clip(losses, 0.0, 1.0)

learner = RCTrackStarLearner(K=K, T=T, sigma=1.0, seed=0)
result = run_oblivious_loss_table(learner, losses)
regret = switching_regret(losses, result["actions"], max_switches=2)

print(f"Learner cumulative loss: {result['cumulative_loss'][-1]:.4f}")
print(f"Regret to best 2-switch comparator: {regret:.4f}")
print(f"Final epoch: {result['diagnostics'][-1]['epoch']}")
