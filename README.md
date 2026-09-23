# RC-Track code

Reference implementation for *Regret Bounds for Restless Rising Concave Bandits via Switching*.

This directory contains a compact research implementation for the RC-Track$^\star$ paper. The online learner follows Algorithm 1 of Zhang (2026) and is wrapped with the reward clipping and loss transformation used by RC-Track$^\star$.

## Components

- `rc_track/learner.py`: universal switching-bandit learner. It implements the fixed-share main learner, canonical dyadic interval subroutines, randomized interval learning rates, challenge rounds, credit accumulation, and rate-doubling restarts.
- `rc_track/comparator.py`: geometric windows, hysteresis comparator, and switch-count utilities used in the theoretical analysis.
- `rc_track/environment.py`: synthetic restless rising-concave Gaussian environment for experiments and smoke tests.
- `rc_track/runner.py`: runners for stochastic reward environments and deterministic oblivious loss tables.
- `rc_track/metrics.py`: exact best-comparator dynamic program for small-to-medium diagnostic instances.
- `rc_track/protocols.py`: minimal environment protocol for plugging in a user-defined simulator.
- `examples/smoke_run.py`: end-to-end example.
- `tests/`: basic correctness and smoke tests.

## Install

```bash
pip install -e ".[test]"
```

## Test

```bash
pytest -q
```

## Online API

```python
from rc_track import RCTrackStarLearner

learner = RCTrackStarLearner(K=5, T=10000, sigma=0.05, seed=0)

for _ in range(10000):
    action, probability, branch = learner.act()
    reward = environment.pull(action)
    learner.update(learner.reward_to_loss(reward))
```

For direct loss-table experiments:

```python
import numpy as np
from rc_track import RCTrackStarLearner, run_oblivious_loss_table

losses = np.random.default_rng(0).uniform(0.0, 1.0, size=(1000, 5))
learner = RCTrackStarLearner(K=5, T=1000, sigma=1.0, seed=0)
result = run_oblivious_loss_table(learner, losses)
```

For a diagnostic benchmark against the best comparator with at most $S$ switches:

```python
from rc_track import switching_regret

reg = switching_regret(losses, result["actions"], max_switches=10)
```

The exact comparator routine is intended for evaluation and sanity checks; it is not used by the online learner.

## Attribution

The switching learner is implemented from the published algorithm specification in:

Mengxiao Zhang. *Toward Optimal Switching Regret for Multi-Armed Bandits with Oblivious Adversary*. arXiv:2609.13547, 2026.

The clipping and reward-to-loss transformation, hysteresis comparator, and rising-concave environment utilities are specific to RC-Track$^\star$.
