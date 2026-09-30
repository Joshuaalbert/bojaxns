"""Measure production random search with real marginalised GP expected improvement.

Run ``python -m cicd.benchmarks.acquisition_search``. The default is the moderate
N=32,D=3,S=16,batch=16,num_search=1024 case; --larger also runs 64,6,32,32,4096.
Supplied deterministic hyperparameters exercise prediction, not posterior fitting.
"""

import argparse

import jax
import numpy as np
from jax import numpy as jnp
from jaxns.mixed_precision import mp_policy

from bojaxns.base import MarginalisationData, MarginalisedAcquisitionFunction
from bojaxns.gaussian_process_formulation.bayesian_optimiser import BayesianOptimiser
from bojaxns.gaussian_process_formulation.distribution_math import (
    ExpectedImprovementAcquisitionFactory,
    GaussianProcessConditionalPredictiveFactory,
    GaussianProcessData,
)
from cicd.benchmarks.gaussian_conditional_predictive import configuration, emit, measure, parser


def run_case(n: int, d: int, s: int, batch: int, num_search: int, args: argparse.Namespace) -> dict:
    """Compare history specializations with identical dynamic inputs and keys."""
    dtype = mp_policy.measure_dtype
    coordinates = jax.random.uniform(jax.random.PRNGKey(11), (n, d), dtype=dtype)
    data = GaussianProcessData(
        U=coordinates, Y=jnp.sum(jnp.sin(6. * coordinates), axis=-1),  # [N, D], [N]
        Y_var=jnp.full((n,), 0.05, dtype), sample_size=jnp.full((n,), 4., dtype),
    )
    samples = MarginalisationData(samples={
        "amplitude": jnp.linspace(0.8, 1.2, s, dtype=dtype),  # [S]
        "length_scale": jnp.linspace(0.2, 0.8, s * d, dtype=dtype).reshape(s, d),  # [S, D]
        "kernel_select": jnp.arange(s) % 3,  # [S], exercise all production kernels
        "variance": jnp.linspace(0.02, 0.08, s, dtype=dtype),  # [S]
        "mean": jnp.linspace(-0.1, 0.1, s, dtype=dtype),  # [S]
    }, log_dp_mean=jnp.full((s,), -jnp.log(jnp.asarray(s, dtype)), dtype))
    search_key, marginalise_key = jax.random.split(jax.random.PRNGKey(42))
    variants, outputs = {}, {}
    for return_history in (True, False):
        def search(data, samples, search_key, marginalise_key):
            # Factory objects are trace-time plumbing; arrays remain dynamic operands.
            factory = GaussianProcessConditionalPredictiveFactory(data)
            acquisition = MarginalisedAcquisitionFunction(
                key=marginalise_key, ns_results=samples,
                acquisition_factory=ExpectedImprovementAcquisitionFactory(factory), S=s,
            )
            return BayesianOptimiser._random_search(
                search_key, acquisition, d, batch, num_search, return_history=return_history,
            )

        name = "history" if return_history else "no_history"
        variants[name], outputs[name] = measure(
            search, (data, samples, search_key, marginalise_key), args,
            f"search_N{n}_D{d}_S{s}_B{batch}_K{num_search}_{name}",
        )
    tolerance = {"rtol": 1e-6, "atol": 1e-7}
    for reference, actual in zip(outputs["history"][0], outputs["no_history"][0]):
        assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
        np.testing.assert_allclose(actual, reference, **tolerance)
    values, candidates = outputs["history"][1]
    assert values.shape == (num_search,) and candidates.shape == (num_search, d)
    assert np.all(np.isfinite(values)) and np.all(np.isfinite(candidates))
    assert outputs["no_history"][1] is None
    best_index = int(np.argmax(values))
    np.testing.assert_allclose(outputs["history"][0][0], values[best_index], **tolerance)
    np.testing.assert_allclose(outputs["history"][0][1], candidates[best_index], **tolerance)
    return {"shapes": {"N": n, "D": d, "S": s, "batch": batch, "num_search": num_search},
            "dtype": str(data.U.dtype), "seeds": {"data": 11, "search_and_marginalisation": 42},
            "sample_shapes": {key: list(value.shape) for key, value in samples.samples.items()},
            "comparison": {"passed": True, **tolerance, "history_argmax_verified": True,
                           "best_value": float(outputs["history"][0][0]),
                           "best_U": np.asarray(outputs["history"][0][1]).tolist()},
            "analytic_history_bytes": num_search * (d + 1) * np.dtype(dtype).itemsize,
            "variants": variants}


def main() -> None:
    """Run explicit representative sizes without a CI timing gate."""
    options = parser(__doc__)
    options.add_argument("--larger", action="store_true")
    args = options.parse_args()
    cases = [(32, 3, 16, 16, 1024)]
    if args.larger:
        cases.append((64, 6, 32, 32, 4096))
    emit({"benchmark": "acquisition_search", "configuration": configuration(),
          "scope": "Production EI and resampling with supplied hyperparameters; excludes posterior inference.",
          "cases": [run_case(*case, args) for case in cases]}, args)


if __name__ == "__main__":
    main()
