# Bayesian optimisation scenarios

Entry point: `cicd/system_tests/test_bayesian_optimiser.py`, retaining the
original two public-service scenarios.

- `test_bayesian_optimisation` creates a two-dimensional quartic objective,
  evaluates seeded trials, exercises a model-selected proposal and a random
  proposal, then records a nonfinite measurement.
- `test_bayesian_optimiser` creates mixed continuous/integer/categorical trials,
  simulates repeated measurements with fixed JAX keys, checks continuous bounds
  and exercises visualisation with the headless Matplotlib backend in CI.

These are existing smoke/integration scenarios, not proofs of global optimality
or comprehensive nonfinite-input handling. They use the real implementation,
may compile expensive numerical programs, and are retained as release gates.
The separate demo exercises the lighter measurement/serialization workflow.

## Real-sampler smoke budgets

The quartic scenario uses four seeded observations followed by one
model-selected proposal, then retains the random proposal and nonfinite
measurement path. It asserts finite, in-domain proposals, finite objective
values, recorded measurements and a fresh trial after seed exhaustion. This
exercises production posterior sampling and acquisition search; small budgets
are integration coverage, not evidence of posterior convergence or optimality.

The service receives these configurable budgets (no monkeypatched sampler):

| Environment variable | Default |
| --- | --- |
| `BOJAXNS_SYSTEM_STEPS` | 5 (seeded observations + one model proposal) |
| `BOJAXNS_SYSTEM_NUM_SEARCH` | 128 |
| `BOJAXNS_SYSTEM_BATCH_SIZE` | 8 |
| `BOJAXNS_SYSTEM_POSTERIOR_SAMPLES` | 8 |
| `BOJAXNS_SYSTEM_ROOTS` | 16 |
| `BOJAXNS_SYSTEM_MAX_SAMPLES` | 4096 |

Steps must be at least three, budgets positive and search size divisible by
batch size. Production defaults are unchanged. The mixed-domain scenario
retains its fixed-key measurement simulation and continuous-domain assertion;
its local user fixture is a dataclass. Figures are closed, and sampler output
from the quartic scenario is isolated under pytest's `tmp_path`.
