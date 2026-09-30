# Bojaxns context

Bojaxns provides Bayesian optimisation over continuous, integer and categorical
parameters. It uses JAXNS to marginalise Gaussian-process model parameters and
provides acquisition and multi-step-lookahead machinery.

`BayesianOptimisation` manages experiments, trials and measurements.
`ParameterSpace` owns the parameter schema and its prior transformation.
The Gaussian-process formulation owns predictions, acquisitions and lookahead;
nested-sampling internals remain the responsibility of installed JAXNS 3.0.0.
Registered JAXNS `PureDataclassPytree` dataclasses own parameter and experiment
state, with explicit host validation and native pytree JSON serialization. NumPy/SciPy own host-side sampling utilities.

This repository is a library, not a distributed service or trading system.
Do not import another project's runtime, secret configuration, scientific
invariants, research results or deployment conventions merely to share tooling.

Prefer clear scientific intent, preserved public APIs, deterministic evidence
and measured performance. See [source ownership](docs/design/SOURCE_LAYOUT.md),
[requirements](docs/design/REQUIREMENTS.md), [invariants](docs/design/INVARIANTS.md)
and [CI/CD](cicd/README.md).
