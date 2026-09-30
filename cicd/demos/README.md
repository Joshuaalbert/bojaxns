# Demos

Run from the repository root after installation:

```bash
conda run -n bojaxns_py python -m cicd.demos.experiment_round_trip
```

The demo creates an experiment, evaluates two seeded trials, records their
measurements and round-trips the public JSON schema. It performs no external
IO, writes no credentials or artifacts, and does not claim optimisation quality.
