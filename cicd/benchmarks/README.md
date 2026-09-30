# Benchmarks

The existing masked/unmasked GP timing experiment is preserved in
`gaussian_conditional_predictive.py`. Run deliberately:

```bash
conda run -n bojaxns_py python cicd/benchmarks/gaussian_conditional_predictive.py
```

It warms and synchronizes the two JAX paths before timing; it is not a PR
performance gate. Record device, dependency versions, precision, shapes and
compile versus steady timing when reporting results. No benchmark result or
performance improvement is claimed by this repository-layout migration.
