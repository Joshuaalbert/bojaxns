# Dependency decisions

`pyproject.toml` is the single source of package requirements. Issue #22
explicitly migrates the runtime to **JAXNS 3.0.0** as well as relocating code
and tests. Python **>=3.10** follows JAXNS 3.0.0's declared minimum; it is not
a tooling-only change. The existing local environment reports Python 3.11.16.
CI targets Python 3.10–3.14 without recreating that environment.

| Dependency | Consumer |
| --- | --- |
| JAX / JAXLIB / Chex | Compiled numerical work, arrays and PRNGs |
| JAXNS 3.0.0 | Prior transforms, nested sampling and posterior resampling |
| tfp-nightly 0.26.0.dev20260930 | JAX distributions and GP kernels |
| NumPy / SciPy | Host arrays and sampling utilities |
| mctx | Multi-step search |
| Matplotlib | Plotting APIs |
| jaxctx >=1.2.0 | Direct context and categorical-prior APIs |

The tested TFP nightly is pinned for reproducibility and imports under
`tensorflow_probability`. Do not also install the stable
`tensorflow-probability` distribution: both provide the same import package.
A source import audit found no etils consumers, so it is no longer a direct
dependency. JAXNS may still require it transitively.
Pydantic is replaced by validated dataclasses with explicit JSON serialization.
pyDOE2 is replaced by SciPy qmc; Latin hypercube construction belongs to
production sampling utilities, not CI or demos. Preserve deterministic seeding and stratum coverage.

Tests/build tools, documentation tools and optional Graphviz visualisation
have separate extras. `requirements*.txt` and `docs/requirements.txt` delegate
to those extras rather than duplicate constraints. Public `bojaxns` imports
remain part of the compatibility contract.

## Historical baseline

On 2026-09-27 the original unbounded requirements resolved JAX/JAXLIB 0.10.2,
TFP 0.25.0, JAXNS 2.4.4 and Pydantic 2.13.5. Importing that implementation
failed because TFP accessed removed JAX internals. This predates relocation;
the old failure is not an excuse to skip current scientific tests.

Builds and structural checks do not establish numerical compatibility. Validate
units, real-sampler system scenarios, demos and imports from an independently
installed wheel. The [historical report](MIGRATION.md) is not current evidence.
