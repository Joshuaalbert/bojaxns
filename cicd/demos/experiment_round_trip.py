"""Exercise the public experiment/measurement workflow without external IO."""

from jax import random

from bojaxns.experiment import OptimisationExperiment
from bojaxns.experiment import TrialUpdate
from bojaxns.service import BayesianOptimisation
from cicd.testing.example import example_request


def main() -> None:
    service = BayesianOptimisation.create_new_experiment(example_request())
    for seed in range(2):
        trial_id = service.create_new_trial(random.PRNGKey(seed), random_explore=True)
        trial = service.get_trial(trial_id)
        x = trial.param_values["x"].value
        assert -1.0 <= x <= 1.0
        service.post_measurement(
            trial_id,
            TrialUpdate(ref_id=f"demo-{seed}", objective_measurement=-(x * x)),
        )
        assert service.trial_size(trial_id) == 1
    restored = OptimisationExperiment.from_json(service.experiment.to_json())
    assert restored == service.experiment
    BayesianOptimisation(restored)  # Revalidate persisted host state before use.
    print(f"Round-tripped {len(restored.trials)} measured trials.")


if __name__ == "__main__":
    main()
