"""Compatibility and boundary validation for the dataclass JSON schemas."""
import json
from dataclasses import is_dataclass
from datetime import datetime, timezone

import numpy as np
import pytest

from bojaxns.common import FloatValue, IntValue
from bojaxns.experiment import NewExperimentRequest, OptimisationExperiment, Trial, TrialUpdate
from bojaxns.parameter_space import CategoricalPrior, ContinuousPrior, IntegerPrior, Parameter, ParameterSpace
from bojaxns.utils import build_example, latin_hypercube


def test_nested_json_constructor_schema_and_roundtrip():
    experiment = OptimisationExperiment(parameter_space={'parameters': [
        {'name': 'n', 'prior': {'type': 'integer_prior', 'lower': -3, 'upper': 2,
                               'mode': 0., 'uncert': float('inf')}}]}, trials={
        't': {'trial_id': 't', 'create_dt': '2026-09-30T12:00:00+00:00',
              'param_values': {'n': {'type': 'int', 'value': -1}}, 'U_value': [.4],
              'trial_updates': {'m': {'ref_id': 'm', 'objective_measurement': 1.,
                                       'measurement_dt': '2026-09-30T12:01:00+00:00'}}}})
    assert is_dataclass(experiment)
    assert experiment == OptimisationExperiment.parse_raw(experiment.json())
    assert experiment.trials['t'].create_dt == datetime(2026, 9, 30, 12, tzinfo=timezone.utc)
    assert isinstance(experiment.trials['t'].param_values['n'], IntValue)
    schema = json.loads(OptimisationExperiment.schema_json())
    assert schema['required'] == ['parameter_space']
    assert schema['properties']['parameter_space']['properties']['parameters']['type'] == 'array'
    assert len(Parameter.schema()['properties']['prior']['anyOf']) == 3
    assert Trial.schema()['properties']['U_value']['items']['maximum'] == 1
    for cls in (ContinuousPrior, IntegerPrior, CategoricalPrior, Parameter, ParameterSpace,
                Trial, TrialUpdate, OptimisationExperiment, NewExperimentRequest):
        example = build_example(cls)
        assert example == cls.parse_raw(example.json())


def test_schema_validation_rechecks_mutable_experiment():
    space = ParameterSpace([Parameter('n', IntegerPrior(2, 5, 3, 1))])
    trial = Trial({'n': IntValue(3)}, [.3])
    experiment = OptimisationExperiment(space, trials={trial.trial_id: trial})
    trial.param_values['n'].value = 8
    with pytest.raises(ValueError, match='outside'):
        experiment.validate()
    trial.param_values['n'] = IntValue(3)
    trial.U_value = [.1, .2]
    with pytest.raises(ValueError, match='coordinates'):
        experiment.validate()
    trial.U_value = [.1]
    trial.param_values = {'wrong': IntValue(3)}
    with pytest.raises(ValueError, match="don't match"):
        experiment.validate()


def test_prior_and_scalar_validation():
    for kwargs in ({'lower': 1, 'upper': 0}, {'lower': 1, 'upper': 1}, {'uncert': 0},
                   {'uncert': float('nan')}, {'lower': float('inf')}):
        args = dict(lower=0., upper=1., mode=.5, uncert=1.)
        args.update(kwargs)
        with pytest.raises(ValueError):
            ContinuousPrior(**args)
    for probs in ([], [0, 0], [-1, 2], [float('nan')], [float('inf')]):
        with pytest.raises(ValueError):
            CategoricalPrior(probs)
    for value in (1.1, True, '1'):
        with pytest.raises(ValueError):
            IntValue(value)
    for value in (float('nan'), float('inf'), True):
        with pytest.raises(ValueError):
            FloatValue(value)
    with pytest.raises(ValueError):
        IntegerPrior(1.2, 4, 2, 1)
    with pytest.raises(ValueError):
        Trial({'x': FloatValue(0)}, [float('nan')])
    with pytest.raises(ValueError):
        Parameter('x', {'type': 'unknown'})
    # A truncated normal's location can lie outside its truncated support.
    assert ContinuousPrior(0, 1, 2, 1).mode == 2
    assert IntegerPrior(0, 1, -2, 1).mode == -2


def test_nonfinite_measurements_preserve_optimizer_penalty_contract():
    for value in (float('nan'), float('inf'), -float('inf')):
        update = TrialUpdate('m', value)
        loaded = TrialUpdate.parse_raw(update.json())
        assert not np.isfinite(loaded.objective_measurement)
    for value in (True, 'bad'):
        with pytest.raises(ValueError):
            TrialUpdate('m', value)


def test_latin_hypercube_is_seeded_stratified_and_rng_isolated():
    np.random.seed(73)
    before = np.random.get_state()
    samples = latin_hypercube(42, 31, 4)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    np.testing.assert_array_equal(samples, latin_hypercube(42, 31, 4))
    assert not np.array_equal(samples, latin_hypercube(43, 31, 4))
    np.testing.assert_array_equal(np.sort((samples * 31).astype(int), axis=0),
                                  np.broadcast_to(np.arange(31)[:, None], (31, 4)))
    for n, d in ((0, 2), (2, 0), (True, 2), (2, 1.2)):
        with pytest.raises(ValueError):
            latin_hypercube(42, n, d)
