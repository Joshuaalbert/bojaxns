"""Compatibility and boundary validation for the dataclass JSON schemas."""
import json
from dataclasses import is_dataclass

import jax
import jax.numpy as jnp
from jaxns.pytree import PureDataclassPytree
from datetime import datetime, timezone

import numpy as np
import pytest

from bojaxns.common import FloatValue, IntValue
from bojaxns.experiment import NewExperimentRequest, OptimisationExperiment, Trial, TrialUpdate
from bojaxns.parameter_space import CategoricalPrior, ContinuousPrior, IntegerPrior, Parameter, ParameterSpace
from bojaxns.utils import latin_hypercube


def test_native_pytree_json_roundtrip_and_explicit_validation():
    experiment = OptimisationExperiment(parameter_space={'parameters': [
        {'name': 'n', 'prior': {'type': 'integer_prior', 'lower': -3, 'upper': 2,
                               'mode': 0., 'uncert': float('inf')}}]}, trials={
        't': {'trial_id': 't', 'create_dt': '2026-09-30T12:00:00+00:00',
              'param_values': {'n': {'type': 'int', 'value': -1}}, 'U_value': [.4],
              'trial_updates': {'m': {'ref_id': 'm', 'objective_measurement': 1.,
                                       'measurement_dt': '2026-09-30T12:01:00+00:00'}}}})
    # Nested mappings are resolved only at the explicit host boundary.
    assert isinstance(experiment.parameter_space, dict)
    experiment.validate()
    assert is_dataclass(experiment)
    encoded = json.loads(json.dumps(experiment.to_json()))
    restored = OptimisationExperiment.from_json(encoded)
    assert isinstance(restored.trials['t'].param_values['n'].value, np.ndarray)
    assert restored.trials['t'].param_values['n'].value.shape == ()
    assert restored.trials['t'].create_dt == datetime(2026, 9, 30, 12, tzinfo=timezone.utc)
    restored.validate()
    assert restored == experiment
    parameter = Parameter('x', ContinuousPrior(0., 1., .5, 1.))
    samples = [FloatValue(.2), IntValue(2), parameter.prior, IntegerPrior(0, 3, 1., 1.),
               CategoricalPrior([.2, .8]), parameter, ParameterSpace([parameter]),
               Trial({'x': FloatValue(.2)}, [.2]), TrialUpdate('m', 1.),
               experiment, NewExperimentRequest(ParameterSpace([parameter]), 2)]
    for sample in samples:
        cls = type(sample)
        assert cls.__bases__ == (PureDataclassPytree,)
        assert '__slots__' in cls.__dict__
        assert not any(hasattr(sample, old) for old in ('json', 'parse_raw', 'schema_json', 'dict'))
        loaded = cls.from_json(json.loads(json.dumps(sample.to_json())))
        loaded.validate()
        sample.validate()
        assert loaded == sample


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
            ContinuousPrior(**args).validate()
    for probs in ([], [0, 0], [-1, 2], [float('nan')], [float('inf')]):
        with pytest.raises(ValueError):
            CategoricalPrior(probs).validate()
    for value in (1.1, True, '1'):
        with pytest.raises(ValueError):
            IntValue(value).validate()
    for value in (float('nan'), float('inf'), True):
        with pytest.raises(ValueError):
            FloatValue(value).validate()
    with pytest.raises(ValueError):
        IntegerPrior(1.2, 4, 2, 1).validate()
    with pytest.raises(ValueError):
        Trial({'x': FloatValue(0)}, [float('nan')]).validate()
    with pytest.raises(ValueError):
        Parameter('x', {'type': 'unknown'}).validate()
    # A truncated normal's location can lie outside its truncated support.
    assert ContinuousPrior(0, 1, 2, 1).mode == 2
    assert IntegerPrior(0, 1, -2, 1).mode == -2


def test_nonfinite_measurements_preserve_optimizer_penalty_contract():
    for value in (float('nan'), float('inf'), -float('inf')):
        update = TrialUpdate('m', value)
        update.validate()
        loaded = TrialUpdate.from_json(update.to_json())
        loaded.validate()
        assert not np.isfinite(loaded.objective_measurement)
    for value in (True, 'bad'):
        with pytest.raises(ValueError):
            TrialUpdate('m', value).validate()


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


def test_numeric_leaves_survive_jit_tree_map_and_vmap():
    time = datetime(2026, 9, 30, tzinfo=timezone.utc)
    space = ParameterSpace([Parameter('x', ContinuousPrior(0., 1., .5, 1.))])
    trial = Trial({'x': FloatValue(.2)}, [.2], trial_id='t', create_dt=time,
                  trial_updates={'m': TrialUpdate('m', 1., time)})
    experiment = OptimisationExperiment(space, experiment_id='e', trials={'t': trial})
    samples = [FloatValue(.2), IntValue(2), space.parameters[0].prior,
               IntegerPrior(0, 3, 1., 1.), CategoricalPrior([.2, .8]),
               space.parameters[0], space, trial, trial.trial_updates['m'],
               experiment, NewExperimentRequest(space, 3)]
    for sample in samples:
        original_leaves, structure = jax.tree.flatten(sample)
        assert all(np.asarray(leaf).dtype.kind in 'fiu' for leaf in original_leaves)
        # No schema constructor may synchronise or validate these traced leaves.
        mapped = jax.jit(lambda tree: jax.tree.map(lambda leaf: leaf + 1, tree))(sample)
        assert jax.tree.structure(mapped) == structure
        for actual, expected in zip(jax.tree.leaves(mapped), original_leaves):
            np.testing.assert_allclose(actual, np.asarray(expected) + 1)
        batched = jax.tree.map(lambda leaf: jnp.stack([leaf, leaf]), sample)
        result = jax.jit(jax.vmap(lambda tree: jax.tree.map(lambda leaf: leaf * 2, tree)))(batched)
        for actual, expected in zip(jax.tree.leaves(result), original_leaves):
            np.testing.assert_allclose(actual, np.repeat(np.asarray(expected)[None], 2, axis=0) * 2)
        restored = type(sample).from_json(batched.to_json())
        assert all(leaf.shape == (2,) for leaf in jax.tree.leaves(restored))
    assert mapped.parameter_space.parameters[0].name == 'x'
    assert jax.jit(lambda value: value)(trial).create_dt == time


def test_host_validation_accepts_native_scalars_and_rejects_nonscalars():
    for value in (np.asarray(True), np.asarray([1]), np.asarray('1')):
        for cls in (FloatValue, IntValue):
            instance = cls(value)
            with pytest.raises(ValueError):
                instance.validate()
    FloatValue.from_json(FloatValue(.25).to_json()).validate()
    IntValue.from_json(IntValue(2).to_json()).validate()
    # Invalid construction is allowed so tree unflatten never performs host work.
    invalid = ContinuousPrior(2., 1., .5, 1.)
    with pytest.raises(ValueError):
        invalid.validate()
    from bojaxns.parameter_space import build_prior_model, inverse_transform_param, sample_U_value
    with pytest.raises(ValueError):
        build_prior_model(ParameterSpace([Parameter('x', invalid)]))
    parameter = Parameter('x', ContinuousPrior(0., 1., .5, 1.))
    with pytest.raises(ValueError):
        inverse_transform_param(jax.random.PRNGKey(0), parameter, FloatValue(np.asarray([.5])))
    with pytest.raises(ValueError):
        sample_U_value(jax.random.PRNGKey(0), ParameterSpace([parameter, parameter]), {'x': FloatValue(.5)})
