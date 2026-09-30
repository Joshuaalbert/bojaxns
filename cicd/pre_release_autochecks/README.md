# Pre-release autochecks

Run `conda run -n bojaxns_py python cicd/pre_release_autochecks/check_all_invariants_covered.py`.
Every exact invariant in `docs/design/INVARIANTS.md` must have recorded unit
coverage before develop-to-main release. Run reviewer checks too: they verify
that referenced tests exist and that classifications do not overlap.
