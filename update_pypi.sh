#!/usr/bin/env bash
set -euo pipefail

# Build only by default. Publishing still requires explicit intent and the
# operator's existing credentials; this script never copies or stores them.
if [[ "$#" -gt 1 || ("${1:-}" != "" && "${1:-}" != "--upload") ]]; then
    echo "Usage: ./update_pypi.sh [--upload]" >&2
    exit 2
fi
cd "$(dirname "$0")"
artifact_dir="$(mktemp -d /tmp/bojaxns-dist.XXXXXX)"
conda run -n bojaxns_py python -m build --outdir "$artifact_dir"
conda run -n bojaxns_py python -m twine check "$artifact_dir"/*
# Validate runtime imports from the wheel without modifying the active env.
wheel_target="$(mktemp -d /tmp/bojaxns-wheel.XXXXXX)"
conda run -n bojaxns_py python -m pip install --no-deps --target "$wheel_target" "$artifact_dir"/*.whl
(
    cd "$wheel_target"
    PYTHONPATH="$wheel_target" conda run -n bojaxns_py python -c 'import os; from pathlib import Path; import bojaxns; from bojaxns import BayesianOptimisation, ParameterSpace; assert Path(bojaxns.__file__).resolve().is_relative_to(Path(os.environ["PYTHONPATH"]).resolve())'
)
echo "Built distributions: $artifact_dir"
echo "Validated wheel target: $wheel_target"
if [[ "${1:-}" == "--upload" ]]; then
    conda run -n bojaxns_py python -m twine upload "$artifact_dir"/*
fi
