# Bojaxns invariants

These are existing contracts exercised by the retained unit tests, not copied
nested-sampling claims. Exact invariant text is keyed in the coverage ledger.

## Parameters and experiment state

- Invariant: Parameter names within one parameter space are unique.
- Invariant: Supported parameter prior schemas preserve their values through JSON serialization.
- Invariant: Prior transforms produce values inside the declared continuous, integer and categorical domains.
- Invariant: Supported parameter values can be mapped to unit-cube coordinates and transformed back to the same values.
- Invariant: Experiment JSON round trips preserve trials and measurements, and trial parameter names must match the experiment parameter space.
- Invariant: For finite observation variances, masked and unmasked GP calculations agree numerically.
- Invariant: Masking the tested observation subsets with infinite variance leaves finite GP likelihoods and predictions.
- Invariant: Latin hypercube designs place one sample in each stratum along every dimension.

## Migration regression contracts

- Invariant: Predictive mixture covariance includes both conditional covariance and covariance of conditional means.
- Invariant: Masked observations do not determine GP empirical prior scales or the expected-improvement incumbent.
- Invariant: Terminal lookahead states have zero continuation value and leave their observation data unchanged.
- Invariant: Latin hypercube sampling is reproducible for a supplied seed without mutating NumPy global random state.
- Invariant: Acquisition search returns the requested candidate count and the same optimum with or without retained history.
- Invariant: Experiments with fewer than two distinct usable trial means continue random exploration.
