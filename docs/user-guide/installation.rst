Installation
============

Bojaxns requires Python >=3.10, following JAXNS 3.0.0. Runtime dependencies
are declared in ``pyproject.toml``, including ``jaxns==3.0.0`` and the tested
``tfp-nightly==0.26.0.dev20260930``. The latter provides the
``tensorflow_probability`` import; do not install both TFP distributions.

Install the published package with:

.. code-block:: bash

   pip install bojaxns

To work on this checkout, install the src-layout package and test extra:

.. code-block:: bash

   conda run -n bojaxns_py python -m pip install -e '.[tests]'
   conda run -n bojaxns_py python -m pytest cicd/tests
   conda run -n bojaxns_py python -m pytest cicd/reviewer_autochecks

Use the existing environment; check its Python version before changing it.
Documentation tools are available via the ``docs`` extra and optional
Graphviz integration via ``visualisation``.
