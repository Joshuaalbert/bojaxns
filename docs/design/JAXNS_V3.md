# JAXNS v3 migration contract

Issue [#22](https://github.com/Joshuaalbert/bojaxns/issues/22) combines the src-layout
and TOML packaging migration with replacement of Pydantic and the JAXNS v2 API.
The structural migration alone did not establish numerical compatibility.

## Schemas and persistence

Experiment and parameter records are registered, slotted dataclasses inheriting
JAXNS `PureDataclassPytree`. Numeric values are children; names, type tags,
identifiers and datetimes are static auxiliary metadata. Mutable trial dictionaries
remain host orchestration state. Existing scientific NamedTuples retain their
array semantics.

Persistence uses the inherited `to_json()` and `from_json()` methods directly.
`to_json()` returns a dictionary: use `json.dump(record.to_json(), file)` and
`Record.from_json(json.load(file))` for files. JAXNS preserves array dtype and
bytes (including infinities/NaNs), and restores leaves as host NumPy arrays.
Its tree definition contains pickled metadata: load only trusted snapshots in
a compatible Python environment. This replaces the old named-field JSON format;
there is no `parse_raw`, JSON Schema generator, or compatibility codec.

Construction and pytree reconstruction are data-only, so JAX can rebuild records
with tracers and batched leaves. Call `.validate()` for standalone host records.
The service validates requests, experiments and measurements before use; parameter
model construction and inversion validate their inputs. Validation normalises
host scalar arrays and nested dictionaries and rechecks mutable state. It is not
part of JAX unflatten, device execution, or snapshot decoding. Invalid domains,
duplicate names, incorrect U lengths and impossible category inversions fail at
these host boundaries. Numeric scalars require the declared kind rather than
string/bool coercion. Nonfinite objective measurements retain the existing
failed-measurement penalty contract.

## Coordinate order and precision

Persisted Bojaxns `U_value` is a flat vector in parameter declaration order:
one coordinate per continuous/integer parameter, and K coordinates per
K-category Gumbel-max parameter. The adapter packs those slices into JAXCTX
named collections, independently of pytree key sorting. Integer priors expose
physical integers, including their lower-bound offset, in the X collection.

JAXNS 3.0.0 with JAXCTX 1.2.0 uses JAXCTX's default float32 native U coordinates.
The Bojaxns flat adapter calls the public JAXCTX transform with explicit
`mp_policy.measure_dtype` (float64), preserving existing persisted-coordinate
precision. Distributions and GP arithmetic use that measure dtype. The adapter
and native JAXNS sample/transform contracts are tested separately; enabling JAX
x64 does not by itself change the dependency's native U contract.

Continuous infinite uncertainty uses Uniform; integer infinite uncertainty
uses uniform categorical logits. Integer CDF inversion samples its conditional
CDF interval after subtracting the physical lower bound. Gumbel-max inversion
uses the equivalent conditional exponential race: draw the first arrival at
unit total rate, then independent residual clocks for the other categories.
This samples the conditional U distribution in bounded work rather than waiting
for an unlikely category in rejection sampling. A final forward check rejects
unrepresentable conditional draws explicitly.

## GP inference and scientific scope

`gaussian_process_prior_model(data)` is one stable v3 model callable. It realises
the five existing hyperparameters and returns a scalar log likelihood. Each
run supplies its data through `args`, avoiding a fresh closure over observations.
The native runner owns compilation. Bojaxns uses `run()` to reach its configured
depth, with a finite sample cap; nonzero termination status raises rather than
presenting a truncated fit as successful. Root allocation and the native depth
condition are configurable. This is a depth/budget contract, not a guarantee of
posterior ESS or an evidence-uncertainty target. The migration does not invent
an equivalence between v2 `parameter_estimation` and v3 posterior allocation.

Results use `X_samples` and `log_dp`. Weighted resampling delegates to JAXNS;
equally weighted predictive moments then use the law of total covariance.
Acquisitions still average conditional acquisition values, which is a different
operation from moment-matching a mixture.

An observed coordinate with zero range uses its full unit-domain width for
the length-scale prior bound. Fewer than two usable observations or a flat
objective cannot determine empirical objective scales; the service continues
random exploration and direct GP prior construction fails clearly. Overflowed
empirical scales fail too. Masked observations do not determine prior scales
or the acquisition incumbent.

The existing observation-noise formula and variance-prior upper bound are
retained. In particular, the code uses replicate variance and a correction
proportional to inverse square-root measurement count; this migration does not
assert that it is a variance-of-the-mean model. The legacy averaged conditional
log-likelihood API is an expected log-likelihood, not Bayesian evidence.

Lookahead remains a deterministic mean-fantasy approximation. Its terminal
state now has no continuation value, and no transition writes past its fixed
storage. Nonterminal bootstrap semantics are retained. Revising the planning
approximation or noise model requires separate scientific work.

## Runtime and diagnostics

Trial creation does not plot or write files. Posterior result plotting, service
visualisation, and explicit `tree_output` remain available to callers.
Acquisition search handles a partial last batch and can discard history while
returning the same best candidate. Public search methods retain history by
default; ordinary trial creation requests only the optimum. Benchmarks call
production implementations and separate compilation, warmed device execution,
and compiler-reported memory. No timing thresholds run in PR CI.
