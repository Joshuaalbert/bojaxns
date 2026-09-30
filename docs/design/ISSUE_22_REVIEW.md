# Issue #22 performance and code-intent review

Review date: 2026-10-01. Scope: the JAXNS v3/schema/layout migration against
develop, including the bug fixes specified in issue #22. The repository's
`performance-and-intent-review-flow` skill was applied during implementation
and by independent correctness and performance reviewers.

## Findings resolved

- Warm-up filtered nonfinite measurements differently from the fitter's existing
  penalty transformation. The service now reuses the fitter's prior validation
  and falls back only on `NotEnoughData`. Regression: observations `[0]` versus
  `[10, NaN x 10]` give two zero penalised means. A sampler hard-stop RuntimeError
  still propagates without creating a trial.
- Diagonal-only predictions constructed a full candidate covariance. Both masked
  and unmasked paths now use the kernel diagonal and column squared norms of
  the triangular solve. Jitted diagonal/full comparisons, including masks, pass
  at `rtol=atol=1e-12`. The incorrect subtraction comment on the posterior mean
  was corrected to match the conditional-Gaussian addition.
- Ordinary acquisition search retained candidate history solely for automatic
  plotting. History is now optional; default trial creation retains only the
  optimum. Full history remains available through the existing search methods.
- Other issue findings are covered by parameter/schema, covariance, masking,
  terminal lookahead, and service regressions. The independent correctness
  re-review reported no outstanding confirmed P1/P2 findings in its scope.

## Compiler observations

Optimised CPU HLO already places training Cholesky and observation-only solves
outside the candidate loop. Candidate solves operate on `[S,N,batch]` right-hand
sides. A manual covariance cache would duplicate compiler reuse, so none was
introduced. Kernel selection and observation-noise semantics remain unchanged.

Before the diagonal change, both GP programs retained a `[100,100]` covariance
dot. That dot is absent afterward. Compiler temporary memory is effectively
unchanged at this shape; removing that operation is not evidence of lower peak
memory. History omission materially reduces compiler-reported output storage,
while temporary storage changes only minimally.

## Measurement

CPU (`cpu:0`), Python 3.11.16, JAX/JAXLIB 0.10.2, float64, x64 enabled,
default matmul precision and no custom XLA/thread flags. Measurements use two
warmups followed by nine calls with full output synchronization. Lowering and
compilation are measured separately from warmed execution. This is shared-machine
wall time including dispatch, not a controlled hardware performance comparison.
No speedup is claimed: all before/after ranges overlap.

GP shapes: N=100 observations, D=1 coordinate, M=100 predictions. The comparison
below is before versus after the diagonal-only implementation change.

| Path | Median milliseconds before [min–max] | After [min–max] | Compile seconds before / after | Temporary bytes before / after |
| --- | --- | --- | --- | --- |
| Unmasked | 28.54 [24.81–45.40] | 22.12 [16.07–35.38] | 0.541 / 0.577 | 320064 / 320016 |
| Masked | 10.68 [8.21–13.91] | 12.01 [10.92–17.27] | 0.491 / 0.396 | 240068 / 240080 |

Masked/unmasked outputs agree within `rtol=1e-6, atol=1e-7`; maximum absolute
mean/variance discrepancy after the change is 8.9e-16 / 2.3e-16. Lowering times
after the change were 0.451 / 0.172 seconds, respectively.

Final acquisition benchmark: N=32, D=3, S=16 hyperparameter draws, candidate
batch=16, search=1024. It uses production EI and JAXNS resampling with supplied
hyperparameters, excluding posterior inference. Both modes return the same
optimum; the moderate-case optimum is also unchanged by the diagonal rewrite.

| History | Median milliseconds [min–max] | Lower / compile seconds | Compiler temporary / output bytes |
| --- | --- | --- | --- |
| Retained | 174.92 [109.42–267.82] | 1.049 / 1.517 | 696400 / 32832 |
| Omitted | 173.32 [124.88–301.72] | 0.589 / 1.268 | 696384 / 48 |

A larger history comparison was also measured before the diagonal rewrite:
N=64,D=6,S=32,batch=32,search=4096. History/no-history output storage was
229464/72 bytes; median runtime was 10.18/8.80 seconds with overlapping ranges
[9.22–11.88]/[7.77–10.26]. This is not a final-code speedup measurement.

Compiler bytes are static estimates and exclude unreported library/runtime
workspaces. No device peak-memory profile or GPU measurement was made. A
bounded real JAXNS posterior solve and the two existing service/system scenarios
validate integration, not posterior convergence or optimiser optimality.

## Reproduction and evidence limits

From the installed checkout:

```bash
conda run -n bojaxns_py python -m cicd.benchmarks.gaussian_conditional_predictive
conda run -n bojaxns_py python -m cicd.benchmarks.acquisition_search
```

Both tools accept `--output` and `--hlo-dir`; the search tool also accepts
`--larger`. Raw JSON and HLO from this review were written under `/tmp`, not
committed as generated research artifacts. The tables preserve the measured
summary; reruns should establish their own hardware/configuration baseline.

The original dependency combination could not import on modern JAX. Regressions
were demonstrated using isolated original functions where needed, and fixed GP
calculations were checked against dense conditioning and full covariance.
There is no claim of identical random trajectories or a statistical old-v2 versus
new-v3 convergence comparison. The documented noise-model and mean-fantasy
questions remain outside this migration's scientific changes.

## Native pytree persistence follow-up

The requested API correction removes `SerialisableBaseModel`, its codec/schema
helpers and the generated schema fixture. Experiment/parameter records directly
inherit registered, slotted JAXNS `PureDataclassPytree`; callers use inherited
`to_json()` / `from_json()`. Numeric values are children and identity/date/type
metadata are auxiliary data. No replacement serialization framework was added.

Validation now runs explicitly at host service, optimiser and parameter-model
boundaries. Constructors and inherited unflatten accept tracers and batched
leaves without scalar conversion or device synchronization. Native JSON restores
NumPy leaves; host validation normalises scalar arrays before orchestration.
The existing flat parameter adapter remains a plain dataclass with its existing
closure ownership; it did not need a new pytree interface.

The JIT/tree-map/vmap regression fails against the previous `b0ddd8f` wheel
because its records are opaque object leaves. It passes after migration across
all eleven record classes, including native batched JSON restoration and static
datetime preservation. Mutable-input/service checks cover resumed snapshots,
invalid requests, invalid observations and invalid externally supplied values.

Validation: 49 unit/system tests and 5 reviewer checks passed on Python 3.11.16;
ledger, Ruff, fatal Flake8, build and Twine checks passed. An independently
installed wheel completed the native snapshot/measurement workflow outside the
checkout. The README cookie example completed six recipes with five simulated
tasters per recipe and saved/resumed its state, using real posterior inference
with bounded test budgets (`num_search=128`, `batch_size=8`, `S=8`,
`root_allocation_degree=16`, `max_samples=4096`). This verifies API composition,
not cookie quality, default-budget runtime or optimisation convergence.

The follow-up changes host records/persistence, not GP kernels or acquisition
arithmetic. The earlier numerical benchmark results above were not rerun and
are not measurements of serialization performance.

Independent follow-up review reported no outstanding confirmed P1/P2 findings.
It ran 20 focused tests and the unmodified README example with flat simulated
scores through exploration and native snapshot save/resume.
