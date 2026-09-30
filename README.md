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

# Change Log

15 Jan, 2024 -- Bojaxns 1.1.0/1 released. Bumped to jaxns 2.4.3/4.
20 July, 2023 -- Bojaxns 1.0.0 released

## Contributor workflow

Production code lives in `src/bojaxns/`; installing the package (normally
editable for development) is required before importing it from a checkout.
The existing `bojaxns` public imports are preserved. The runtime uses
`jaxns==3.0.0` and the tested `tfp-nightly==0.26.0.dev20260930` distribution.
Experiment and parameter schemas use validated dataclasses with JSON round
trips; Pydantic and pyDOE2 are no longer runtime dependencies. See the
[dependency decisions](docs/design/DEPENDENCIES.md) for compatibility details.

See [AGENTS.md](AGENTS.md), [CI/CD checks](cicd/README.md), and the
[design requirements](docs/design/REQUIREMENTS.md). Dependencies and extras
are maintained in `pyproject.toml`; requirements files are compatibility
entry points, not independent dependency lists. Build distributions with
`conda run -n bojaxns_py python -m build`.
