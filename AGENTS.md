# Agent Notes (bojaxns)

Read [LEARNINGS.md](LEARNINGS.md). For orientation, use
[COMMON_CONTEXT.md](COMMON_CONTEXT.md); current code and design contracts remain
authoritative.

## Repository and environment

- Production code: `src/bojaxns/`. Package metadata and dependencies:
  `pyproject.toml`, using setuptools and the src layout.
- Local development environment: `bojaxns_py` (currently Python 3.11.16);
  Python 3.14 is a CI target, not a locally verified interpreter.
  Check `conda run -n bojaxns_py python --version` before changing an environment.
  Never silently recreate or downgrade a user's environment.
- Python >=3.10 is required by JAXNS 3.0.0; this is a dependency-driven
  minimum, not a tooling preference.
- Run Python and tools through `conda run -n bojaxns_py ...`.
- Install editable with `conda run -n bojaxns_py python -m pip install -e '.[tests]'`.
- Use the installed JAXNS 3.0.0 dependency, not the sibling checkout.
  The tested TFP distribution is tfp-nightly==0.26.0.dev20260930.

## Code intent and ownership

Prefer linear code with concise comments explaining non-obvious mathematics,
array axes, JAX static/dynamic boundaries and downstream consequences.
Read the relevant design document completely before changing its contract.
Reuse an existing implementation when its ownership and semantics match;
do not add abstractions solely to reduce line count.

Keep public `bojaxns` imports working. Existing package re-exports are an API:
do not empty `__init__.py` merely because another repository does so.
Production owns numerical behaviour; CI, demos and benchmarks must call it,
not contain a second optimiser. Dataclasses own validated experiment and
parameter schemas and explicit JSON serialization; NamedTuples own existing JAX array containers.

Use absolute imports, explicit types, four spaces, descriptive names and
Google-style API docstrings. Add adjacent symbolic shape comments to array
fields. Validate at the boundary that owns the contract. Do not move a static
JAX tracing check into every device execution or introduce host synchronization
without measured need. Use the bundled performance-and-intent review skill
for performance, tracing, numerical or maintainability reviews.

## Verification

- Units: `conda run -n bojaxns_py python -m pytest cicd/tests`.
- Reviewer checks: `conda run -n bojaxns_py python -m pytest cicd/reviewer_autochecks`.
- System tests: `conda run -n bojaxns_py python -m pytest cicd/system_tests`.
- Release ledger: `conda run -n bojaxns_py python cicd/pre_release_autochecks/check_all_invariants_covered.py`.
- Lint: `conda run -n bojaxns_py ruff check src cicd` and
  `conda run -n bojaxns_py flake8 src cicd --select E9,F63,F7,F82`.
- Build: `conda run -n bojaxns_py python -m build`; verify the built wheel from
  outside the checkout, not only editable imports.
- Regressions should fail before the fix and pass afterward. Keep tests
  deterministic, isolate network/time, and put generated output in `tmp_path`.
- Every active unit test belongs in either the exact-text invariant ledger or
  the non-invariant ledger, not both. Do not invent scientific coverage.
- Benchmarks are explicit measurements, not noisy timing gates in PR CI.

## Collaboration and hygiene

Preserve unrelated and concurrent edits. For requested PR work, use a dedicated
branch/worktree from the requested base and link a tracking issue. When the user
explicitly asks to update this checkout, work here without assuming permission
to commit, push, publish or open a PR. After a merged PR, clean up its worktree.

Identify ownership before changing a shared contract. Explain any scientific
or subsystem boundary crossing and coordinate before implementing it.
Do not infer broader authority from persistence instructions.

Never copy credentials, auth state, local sessions, private machine settings,
`.env` files, editor settings or generated research artifacts from siblings.
Keep dependency declarations in `pyproject.toml`; requirement files are only
compatibility entry points. Publishing requires explicit user authorization.
