from jax import numpy as jnp

from bojaxns.common import FloatValue, IntValue
from bojaxns.experiment import Trial, OptimisationExperiment, \
    TrialUpdate
from bojaxns.parameter_space import Parameter, IntegerPrior, CategoricalPrior, ParameterSpace, \
    ContinuousPrior
from bojaxns.utils import current_utc


def test_optimisation_experiment(tmp_path):
    param1 = Parameter(
        name='continuous',
        prior=ContinuousPrior(
            lower=0,
            upper=5,
            mode=4,
            uncert=1
        )
    )

    param2 = Parameter(
        name='integers',
        prior=IntegerPrior(
            lower=0,
            upper=5,
            mode=4,
            uncert=jnp.inf
        )
    )

    param3 = Parameter(
        name='categorical',
        prior=CategoricalPrior(
            probs=[1., 1., 1.]
        )
    )

    parameter_space = ParameterSpace(parameters=[param1, param2, param3])

    trial = Trial(param_values={'continuous': FloatValue(value=1.),
                                'integers': IntValue(value=1),
                                'categorical': IntValue(value=2)
                                }, U_value=[0.5, 0.5, 0.5, 0.5, 0.5])
    trials = {trial.trial_id: trial}
    s = OptimisationExperiment(parameter_space=parameter_space, trials=trials)
    assert s == OptimisationExperiment.parse_raw(s.json())

    trial.trial_updates['1234'] = TrialUpdate(ref_id='1234',
                                              measurement_dt=current_utc(),
                                              objective_measurement=1.)

    assert s == OptimisationExperiment.parse_raw(s.json())

    # Validation errors

    try:
        trial = Trial(param_values={'continuous': FloatValue(value=1.),
                                    'integers': IntValue(value=1)
                                    }, U_value=[0.5, 0.5])
        trials = {trial.trial_id: trial}
        _ = OptimisationExperiment(parameter_space=parameter_space, trials=trials)
        assert False
    except ValueError as e:
        assert "don't match param space" in str(e)

    with (tmp_path / 'optimisation_experiment_schema.json').open('w') as f:
        f.write(OptimisationExperiment.schema_json(indent=2))
