# Bojaxns learnings

Keep verified, project-specific findings here; do not import another project's
scientific assumptions as Bojaxns requirements.

- The src layout requires an installation. Use the `bojaxns_py` environment
  and an editable install for local work; validate release wheels independently.
- Issue #22 combines the src/test relocation with an explicit JAXNS 3.0.0
  compatibility migration and validated dataclass schemas. Preserve public
  imports and JSON round trips; do not treat the historical migration report
  as evidence for the current runtime.
- JAXNS 3.0.0 metadata requires Python >=3.10. The existing environment
  reports Python 3.11.16; a Python 3.14 CI target is not a local test result.
- The tested TFP distribution is tfp-nightly==0.26.0.dev20260930. It imports
  as tensorflow_probability; do not install both distributions together.
- A fresh install of the original unbounded requirements on 2026-09-27 resolved
  JAX 0.10.2 and TFP 0.25.0; importing the unchanged package failed because TFP
  references the removed `jax.interpreters.xla.pytype_aval_mappings`.
  Record and resolve compatibility explicitly; do not hide it with skipped tests.
- Existing package re-exports are used by tests and users. Preserve them.
- Infinite observation variance represents masking in the GP formulation;
  numerical tests compare masked and unmasked paths and verify finite outputs.
- The legacy experiment test writes a schema artifact; keep it in `tmp_path`
  so validation never modifies a checkout or a tracked reference schema.
