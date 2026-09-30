# Bojaxns CI/CD

Adapted from the sibling JAXNS lifecycle structure, with Bojaxns-owned tests
and contracts. Production is discovered only under `src/`.

| Directory | Role |
| --- | --- |
| `tests/` | Deterministic unit and numerical regression tests |
| `reviewer_autochecks/` | Structure, shape documentation and coverage-ledger integrity |
| `system_tests/` | Composed public-API scenarios with design specifications |
| `pre_release_autochecks/` | Require every documented invariant to have unit coverage |
| `demos/` | Executable, network-independent public-API examples |
| `benchmarks/` | Explicit performance measurements, outside PR timing gates |
| `testing/` | Shared deterministic setup, not a second optimiser |

Use `conda run -n bojaxns_py python -m pip install -e '.[tests]'`.
Run `python -m pytest` through that environment for units; explicitly name the
reviewer or system directory for those checks. The pre-release entry point is
`python cicd/pre_release_autochecks/check_all_invariants_covered.py`.

Feature PRs run units and reviewer checks. Develop pushes run demos.
Develop-to-main PRs and main pushes also run system and pre-release checks.
The coverage ledger is exact-text ownership, not a claim of code coverage:
each unit test is mapped to documented invariants or explicitly classified
non-invariant. Tests are not removed or skipped to satisfy the ledger.

The former in-package tests are retained here. The two expensive optimiser
scenarios moved to system tests with configurable smoke budgets; the 1,000-repeat timing function moved to
benchmarks. Their lifecycle changed, not their scientific assertions.

Current dependency decisions and historical compatibility findings are in
[DEPENDENCIES.md](../docs/design/DEPENDENCIES.md). Reviewer checks and builds
are separately runnable even when the scientific runtime cannot import.
