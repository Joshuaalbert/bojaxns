import json

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
    s.validate()
    assert s == OptimisationExperiment.from_json(json.loads(json.dumps(s.to_json(), allow_nan=False)))

    trial.trial_updates['1234'] = TrialUpdate(ref_id='1234',
                                              measurement_dt=current_utc(),
                                              objective_measurement=1.)

    s.validate()
    assert s == OptimisationExperiment.from_json(json.loads(json.dumps(s.to_json(), allow_nan=False)))

    # Validation errors

    try:
        trial = Trial(param_values={'continuous': FloatValue(value=1.),
                                    'integers': IntValue(value=1)
                                    }, U_value=[0.5, 0.5])
        trials = {trial.trial_id: trial}
        OptimisationExperiment(parameter_space=parameter_space, trials=trials).validate()
        assert False
    except ValueError as e:
        assert "don't match param space" in str(e)

    path = tmp_path / 'experiment.json'
    path.write_text(json.dumps(s.to_json(), allow_nan=False))
    assert s == OptimisationExperiment.from_json(json.loads(path.read_text()))
