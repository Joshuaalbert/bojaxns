# System tests

Run `conda run -n bojaxns_py python -m pytest cicd/system_tests`.
The original optimiser scenarios are retained in `test_bayesian_optimiser.py`.
Their composed contract is described in
[the scenario specification](../../docs/design/system_tests/bayesian_optimisation.md).
They exercise actual numerical work and can be substantially slower than units.
