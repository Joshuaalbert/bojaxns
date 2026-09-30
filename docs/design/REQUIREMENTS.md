# Bojaxns design requirements

## Package and public API

- Production lives in `src/bojaxns/`; package discovery must not install CI,
  docs, benchmarks or test fixtures.
- `pyproject.toml` owns metadata, runtime dependencies and tests/docs/visualisation
  extras. Requirement files are compatibility entry points only.
- Preserve version 1.1.1 and the existing public import surface during this
  migration, except the explicitly replaced Pydantic serialization APIs. Python >=3.10 follows JAXNS 3.0.0 metadata; the requested
  local environment is Python 3.14. Compatibility is verified, not inferred
  from that declaration.
- The project uses Apache-2.0, as explicitly authorized for this migration.
  LICENSE and package metadata must agree in built distributions.
- Registered `PureDataclassPytree` dataclasses own experiment/parameter state.
  JAXNS owns `to_json()` / `from_json()` persistence; host boundaries own explicit
  validation. Existing scientific array containers are NamedTuples; array fields
  carry adjacent symbolic shape comments.
  The [v3 migration contract](JAXNS_V3.md) records persistence, precision and
  scientific boundary choices.

## Scientific ownership

- Parameter transforms belong in `parameter_space.py`; trial/measurement
  orchestration belongs in `service.py`.
- Gaussian-process predictions, acquisitions and lookahead belong in
  `gaussian_process_formulation/`; nested sampling belongs in the installed
  JAXNS 3.0.0 dependency. Its API migration requires numerical verification
  independently of the file-layout checks.
- Preserve model, precision, masking and random-key semantics unless their
  change is separately specified and tested.
- Keep static shape validation at its JAX tracing boundary. Benchmark claims
  distinguish compilation, device execution and synchronization.

## Verification and hygiene

- Every active unit test has invariant or explicit non-invariant ownership.
- Reviewer checks run on PRs; system and complete-invariant gates also run
  before a develop-to-main release. Demos run on develop pushes.
- System scenarios have matching specs under `system_tests/`; benchmarks
  measure real production paths and are not shared-runner timing assertions.
- Preserve existing user docs/examples and local edits.
- Never copy sibling credentials, machine-local auth/configuration, sessions,
  generated outputs or scientific evidence into this scaffold.
