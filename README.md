[![Python](https://img.shields.io/pypi/pyversions/bojaxns.svg)](https://badge.fury.io/py/bojaxns)
[![PyPI](https://badge.fury.io/py/bojaxns.svg)](https://badge.fury.io/py/bojaxns)
[![Documentation Status](https://readthedocs.org/projects/bojaxns/badge/?version=latest)](https://bojaxns.readthedocs.io/en/latest/?badge=latest)

Main
Status: ![Workflow name](https://github.com/JoshuaAlbert/bojaxns/actions/workflows/unittests.yml/badge.svg?branch=main)

Develop
Status: ![Workflow name](https://github.com/JoshuaAlbert/bojaxns/actions/workflows/unittests.yml/badge.svg?branch=develop)

## Mission: _To make advanced Bayesian Optimisation easy._

# What is it?

Bojaxns is:

1) a Bayesian Optimisation package for easily performing advanced non-myopic Bayesian optimisation.
2) using [JAXNS](https://github.com/JoshuaAlbert/jaxns) under the hood to marginalise over multiple models.
3) using multi-step lookahead to plan out your next step.
4) licensed under the Apache License, Version 2.0; see [LICENSE](LICENSE).

# Documentation

For examples, check out the [documentation](https://bojaxns.readthedocs.io/) (still in progress).

# Install

**Notes:**

1. Bojaxns requires Python >=3.10 (required by JAXNS 3.0.0).
2. It is always highly recommended to use a unique virtual environment for each project.
   To use `miniconda`, have it installed, and run

```bash
# To create a new env, if necessary
conda create -n bojaxns_py python=3.14
conda activate bojaxns_py
```

## For end users

Install directly from PyPi,

```bash
pip install bojaxns
```

## For development

Clone repo `git clone https://www.github.com/JoshuaAlbert/bojaxns.git`, and install:

```bash
cd bojaxns
conda run -n bojaxns_py python -m pip install -e '.[tests]'
conda run -n bojaxns_py python -m pytest
conda run -n bojaxns_py python -m pytest cicd/reviewer_autochecks
```

# Walkthrough: a cookie recipe with many tasters

This example uses the JAXNS v3 API in this checkout; install from source as
shown above. Keep flour (150 g), oven temperature (180°C), and the rest of the
recipe fixed. Vary butter, sugar, baking time and chocolate type. Bojaxns
**maximises** the score, so ask each taster to rate a batch from 0 to 10, with
higher meaning better.

```python
import json
from pathlib import Path

from jax import random

from bojaxns import (
    BayesianOptimisation, CategoricalPrior, ContinuousPrior, IntegerPrior,
    NewExperimentRequest, OptimisationExperiment, Parameter, ParameterSpace,
    TrialUpdate,
)

chocolates = ("dark", "milk")  # Categorical values are indices into this tuple.
tasters = ("Alex", "Blair", "Casey", "Drew", "Ellis")

experiment = BayesianOptimisation.create_new_experiment(
    NewExperimentRequest(
        parameter_space=ParameterSpace(parameters=[
            Parameter("butter_g", ContinuousPrior(50., 120., 85., float("inf"))),
            Parameter("sugar_g", ContinuousPrior(40., 120., 80., float("inf"))),
            Parameter("bake_minutes", IntegerPrior(8, 16, 12., float("inf"))),
            Parameter("chocolate", CategoricalPrior([1., 1.])),
        ]),
        init_explore_size=3,
    )
)
key = random.PRNGKey(42)

for batch in range(6):
    key, proposal_key = random.split(key)
    trial_id = experiment.create_new_trial(proposal_key)
    values = experiment.get_trial(trial_id).param_values
    print(
        f"Batch {batch + 1}: {values['butter_g'].value:.1f} g butter, "
        f"{values['sugar_g'].value:.1f} g sugar, "
        f"{values['bake_minutes'].value} minutes, "
        f"{chocolates[values['chocolate'].value]} chocolate"
    )
    # Bake this recipe, then record every taster's score on the SAME trial.
    for taster in tasters:
        score = float(input(f"{taster}'s score (0–10): "))
        if not 0. <= score <= 10.:
            raise ValueError("Scores must be between 0 and 10.")
        experiment.post_measurement(
            trial_id,
            TrialUpdate(ref_id=taster, objective_measurement=score),
        )

# Save and resume the measured trials, parameter space, and pending recipes.
path = Path("cookies.json")
path.write_text(json.dumps(experiment.experiment.to_json(), allow_nan=False))
restored = OptimisationExperiment.from_json(json.loads(path.read_text()))
experiment = BayesianOptimisation(restored)
```

The first three recipes form the initial exploration design. Later proposals
use the accumulated scores to fit a Gaussian process and choose another recipe;
a flat set of scores continues exploration. Fitting may take time. A trial is
one recipe, and its measurements are individual tasters' scores. Bojaxns uses
both their average and variability; it does not model each taster's preferences
separately. Reusing a `ref_id` on the same trial replaces that taster's score.
Wait for all tasters before requesting the next recipe: a trial with no scores
is returned again, while a trial with one score is already eligible for fitting.

Repeat the proposal-and-rating loop for as many batches as your budget allows.
On resume, supply fresh random keys (or persist your key separately). These
snapshots use JAXNS's native pytree format, including pickled tree metadata;
only load trusted files in a compatible environment. `to_json()` returns a
JSON-compatible dictionary, and `from_json()` accepts that dictionary. Standalone
records can be checked with `.validate()`; the service does this before use.

# Change Log

15 Jan, 2024 -- Bojaxns 1.1.0/1 released. Bumped to jaxns 2.4.3/4.
20 July, 2023 -- Bojaxns 1.0.0 released

## Contributor workflow

Production code lives in `src/bojaxns/`; installing the package (normally
editable for development) is required before importing it from a checkout.
The existing `bojaxns` public imports are preserved. The runtime uses
`jaxns==3.0.0` and the tested `tfp-nightly==0.26.0.dev20260930` distribution.
Experiment and parameter records use JAXNS `PureDataclassPytree` with explicit
host validation and native `to_json()` / `from_json()` round trips; Pydantic and pyDOE2 are no longer runtime dependencies. See the
[dependency decisions](docs/design/DEPENDENCIES.md) for compatibility details.

See [AGENTS.md](AGENTS.md), [CI/CD checks](cicd/README.md), and the
[design requirements](docs/design/REQUIREMENTS.md). Dependencies and extras
are maintained in `pyproject.toml`; requirements files are compatibility
entry points, not independent dependency lists. Build distributions with
`conda run -n bojaxns_py python -m build`.
