# Source layout and ownership

| Path | Ownership |
| --- | --- |
| `src/bojaxns/parameter_space.py` | Parameter priors and U-space transforms |
| `src/bojaxns/experiment.py`, `common.py` | Experiment, trial, update and scalar schemas |
| `src/bojaxns/service.py` | Public experiment/trial orchestration |
| `src/bojaxns/base.py` | Predictive/acquisition interfaces and marginalisation |
| `src/bojaxns/gaussian_process_formulation/` | GP likelihood/prediction, optimiser and lookahead |
| `src/bojaxns/utils.py`, `basic.py` | Existing schema/time/sampling utilities |
| `cicd/` | Verification and examples, never installed as production |
| `docs/` | Existing user/API docs plus design and system specifications |

The production package uses the src layout; issue #22 also migrates the
JAXNS integration and schema implementation under their production owners. Public package
re-exports remain for compatibility. Do not empty existing `__init__.py`
files by mechanically following a different repository's API convention.
