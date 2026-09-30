"""Explicit GP measurements; run with ``python -m cicd.benchmarks.gaussian_conditional_predictive``.

Artifacts default to /tmp. Compiler memory is a static plan, not measured peak RSS
or device memory; runtime samples include dispatch and synchronization.
"""

import argparse
import json
import os
import platform
import re
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import jax
import jaxlib
import numpy as np
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from bojaxns.gaussian_process_formulation.distribution_math import (
    GaussianProcessConditionalPredictive,
    GaussianProcessData,
)


def positive_int(value: str) -> int:
    """Parse a positive benchmark size."""
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parser(description: str) -> argparse.ArgumentParser:
    """Build the common explicit-measurement options."""
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--repeats", type=positive_int, default=9)
    result.add_argument("--warmup", type=positive_int, default=2)
    result.add_argument("--output", type=Path, help="JSON file under /tmp; JSON also goes to stdout")
    result.add_argument("--artifacts", type=Path, default=Path("/tmp/bojaxns-benchmarks"))
    return result


def artifact_path(path: Path) -> Path:
    """Keep generated evidence out of the checkout and under /tmp."""
    path = path.resolve()
    checkout = Path(__file__).resolve().parents[2]
    if Path("/tmp") not in path.parents or path == checkout or checkout in path.parents:
        raise ValueError("Artifacts must be under /tmp and outside the checkout")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def configuration() -> dict[str, Any]:
    """Describe the runtime and precision used by the compiled executables."""
    return {
        "python": platform.python_version(), "jax": jax.__version__, "jaxlib": jaxlib.__version__,
        "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()],
        "device_kinds": [device.device_kind for device in jax.devices()],
        "jax_enable_x64": jax.config.jax_enable_x64,
        "jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision),
        "jax_enable_compilation_cache": jax.config.jax_enable_compilation_cache,
        "jax_compilation_cache_dir": jax.config.jax_compilation_cache_dir,
        "environment": {name: os.environ.get(name) for name in
                        ("XLA_FLAGS", "JAX_ENABLE_X64", "OMP_NUM_THREADS", "JAX_PLATFORMS",
                         "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "memory_note": "Compiler byte estimates exclude unreported library/runtime workspaces; not measured peak memory.",
        "timing_note": "Host wall time includes dispatch and full output synchronization; no timing thresholds.",
    }


def measure(function: Callable, arguments: tuple, args: argparse.Namespace, name: str) -> tuple[dict, Any]:
    """Lower, compile, warm up, and measure one exact static specialization."""
    jax.block_until_ready(arguments)
    start = perf_counter()
    lowered = jax.jit(function).lower(*arguments)
    lowering_seconds = perf_counter() - start
    hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text()
    start = perf_counter()
    compiled = lowered.compile()
    compile_seconds = perf_counter() - start
    optimized_hlo = compiled.as_text()
    paths = {}
    for suffix, contents in (("lowered", hlo), ("optimized", optimized_hlo)):
        path = artifact_path(args.artifacts / f"{name}.{suffix}.hlo.txt")
        path.write_text(contents)
        paths[suffix] = str(path)
    memory = compiled.memory_analysis()
    memory_bytes = None if memory is None else {
        field: int(getattr(memory, field)) for field in dir(memory) if field.endswith("_size_in_bytes")
    }
    for _ in range(args.warmup):
        output = jax.block_until_ready(compiled(*arguments))
    durations = []
    for _ in range(args.repeats):
        start = perf_counter()
        output = jax.block_until_ready(compiled(*arguments))
        durations.append(perf_counter() - start)
    leaves = jax.tree.leaves(output)
    return {
        "lowering_seconds": lowering_seconds, "compile_seconds": compile_seconds,
        "warmup_calls": args.warmup, "runtime_seconds": {
            "median": float(np.median(durations)), "min": min(durations), "max": max(durations),
            "samples": durations,
        },
        "compiler_memory_bytes": memory_bytes,
        "output_bytes": sum(leaf.size * leaf.dtype.itemsize for leaf in leaves),
        "outputs": [{"shape": list(leaf.shape), "dtype": str(leaf.dtype)} for leaf in leaves],
        "hlo": {"paths": paths, "lowered_text_bytes": len(hlo.encode()),
                "optimized_text_bytes": len(optimized_hlo.encode()),
                "optimized_operation_counts": {
                    operation: len(re.findall(r"\b" + operation + r"\(", optimized_hlo))
                    for operation in ("while", "fusion", "copy", "transpose", "broadcast", "constant", "custom-call")
                }},
    }, output


def emit(report: dict, args: argparse.Namespace) -> None:
    """Write strict JSON to stdout and optionally to a supplied file."""
    contents = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        artifact_path(args.output).write_text(contents + "\n")
    print(contents)


def main() -> None:
    """Compare production posterior paths with finite observation variances."""
    options = parser(__doc__)
    options.add_argument("--observations", type=positive_int, default=100)
    options.add_argument("--candidates", type=positive_int, default=100)
    args = options.parse_args()
    coordinates = jnp.linspace(0., 10., args.observations)
    data = GaussianProcessData(
        U=coordinates[:, None], Y=jnp.sin(coordinates),  # [N, D=1], [N]
        Y_var=jnp.ones_like(coordinates), sample_size=jnp.ones_like(coordinates),
    )
    candidates = jnp.linspace(-0.1, 1.1, args.candidates)[:, None]  # [M, D=1]

    def predict(data: GaussianProcessData, candidates: jax.Array, masked: bool):
        # Explicit dynamic data avoids precomputing the training solve as a closure constant.
        predictive = GaussianProcessConditionalPredictive(
            data=data,
            kernel=tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=1., length_scale=jnp.pi),
            variance=jnp.asarray(1.), mean=jnp.asarray(1.),
        )
        method = predictive._posterior_with_mask if masked else predictive._posterior
        return method(candidates, cov=False)

    results, outputs = {}, {}
    for masked in (False, True):
        name = "masked" if masked else "unmasked"
        results[name], outputs[name] = measure(
            lambda data, candidates: predict(data, candidates, masked), (data, candidates), args,
            f"gp_N{args.observations}_M{args.candidates}_{name}",
        )
    tolerance = {"rtol": 1e-6, "atol": 1e-7}
    errors = []
    for reference, actual in zip(outputs["unmasked"], outputs["masked"]):
        assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
        np.testing.assert_allclose(actual, reference, **tolerance)
        errors.append(float(np.max(np.abs(np.asarray(actual) - np.asarray(reference)))))
    emit({"benchmark": "gaussian_conditional_predictive", "configuration": configuration(),
          "shapes": {"N": args.observations, "D": 1, "M": args.candidates},
          "dtype": str(data.U.dtype), "comparison": {"passed": True, **tolerance,
          "max_abs_errors_mean_variance": errors, "finite_observation_variances": True},
          "variants": results}, args)


if __name__ == "__main__":
    main()
