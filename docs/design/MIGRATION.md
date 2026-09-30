# Historical repository modernisation verification — 2026-09-27

This is the historical report from the original uncommitted tree, not current
issue #22 validation. Its dependency, Python-floor and machine-configuration
statements describe that earlier snapshot. See [DEPENDENCIES.md](DEPENDENCIES.md)
and [REQUIREMENTS.md](REQUIREMENTS.md) for current contracts. The authorized Apache licence text is retained in issue #22.

The requested migration is local and uncommitted. No release was uploaded.

## Scope

- Move all 12 production modules under `src/bojaxns`, retaining public imports.
- Move the original tests/fixture under `cicd`; preserve two composed optimiser
  scenarios as system tests and the timing experiment as an explicit benchmark.
- Replace `setup.py` with PEP 621/639 metadata and tests/docs/visualisation
  extras in `pyproject.toml`. Requirement files delegate to that metadata.
- Adapt the sibling JAXNS Codex config, rules, review skill and coverage-ledger
  checks; author Bojaxns-specific requirements/invariants/ownership docs.
- Change the licence to Apache-2.0 at the user's explicit request, copying the
  sibling's licence text and checking the built distribution's metadata.
- Update CI, Sphinx/Read the Docs paths and the local build/publish helper.
  Upload now requires `--upload`; no publishing or credential copying occurred.

Only explicitly selected tracked JAXNS files were used. Local `.idea` content
was preserved, and no auth/session files, `.env`, private keys or generated
research results were imported. The source repositories were not modified.

## Checks

- New structure regressions before migration: **4 failed** as expected.
- Final reviewer autochecks: **5 passed**.
- Full documented-invariant coverage gate: **passed**.
- Ruff, strict Flake8 fatal checks, shell syntax and diff whitespace: **passed**.
- Bundled skill validator and actual Codex rule evaluation: **passed**;
  the named environment command is allowed and `git reset --hard` is forbidden.
- Workflow YAML and Codex TOML parsing: **passed**.
- Source AST comparison: **all 12 production modules unchanged**; only array
  shape comments were added to three relocated modules.
- Isolated PEP 517 sdist and wheel build, followed by Twine checks: **passed**.
- Wheel inventory: exactly the original production modules, Apache-2.0 licence
  metadata and text; no tests, CI, docs, Codex configuration or credentials.
- Wheel installed under a separate `/tmp` target and its package path resolved
  independently of the src checkout.

Artifacts: `/tmp/bojaxns-modern-dist-20260927/`; independent wheel target:
`/tmp/bojaxns-wheel-smoke-20260927/`.

## Explicit limitations

The local interpreter used for these checks reported **Python 3.11.16** at
`/home/albert/miniconda3/envs/bojaxns_py/bin/python`, despite the user's intended
Python 3.14 environment. No Python 3.14 test result is claimed. Documentation
and CI include the requested 3.14 target; the original >=3.9 metadata floor
remains unchanged.

The original scientific dependency stack already fails with freshly resolved
dependencies. Unit collection after migration has four import errors from
TFP's removed `pytype_aval_mappings` use and JAXNS 2.4.4's removed top-level
`jax.tree_map` import. The same baseline import failure was reproduced before
moving files. See [DEPENDENCIES.md](DEPENDENCIES.md).

Numerical units, system tests, demos and an actual wheel runtime import are
therefore **not validated**. CI preserves those checks rather than skipping
or marking them allowed failures. Modernising that scientific stack is a
separate compatibility task, not evidence supplied by this layout migration.
