# Reviewer autochecks

Run `conda run -n bojaxns_py python -m pytest cicd/reviewer_autochecks`.

These checks do not import the numerical runtime. They enforce src-only
package discovery, required CI/design files, no tests in production, symbolic
array-shape comments on scientific NamedTuples/dataclasses, and exact ledger
ownership for every active unit test. Experiment dataclasses may remain mutable.
Checks enforce package contracts, not the presence of machine-specific agent
configuration or rules; those files are not packaging prerequisites.

Uncovered invariants are allowed during feature work; the release gate rejects
them. Runtime incompatibility must still be reported separately.
