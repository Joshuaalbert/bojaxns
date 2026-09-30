"""Service warm-up and acquisition-search migration regressions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bojaxns.experiment import NewExperimentRequest, TrialUpdate
from bojaxns.gaussian_process_formulation.bayesian_optimiser import BayesianOptimiser
from bojaxns.parameter_space import ContinuousPrior, Parameter, ParameterSpace
from bojaxns.service import BayesianOptimisation


def test_search_remainder_and_optional_history():
    key = jax.random.PRNGKey(2)
    acquisition = lambda u: -jnp.sum((u - 0.3) ** 2)
    best, history = BayesianOptimiser._random_search(key, acquisition, 2, 4, 7)
    values, candidates = history
    assert values.shape == (7,)
    assert candidates.shape == (7, 2)
    np.testing.assert_allclose(best[0], values.max())
    np.testing.assert_array_equal(best[1], candidates[jnp.argmax(values)])
    compiled = jax.jit(lambda k: BayesianOptimiser._random_search(
        k, acquisition, 2, 4, 7, return_history=False))
    no_history_best, history = compiled(key)
    assert history is None
    np.testing.assert_allclose(no_history_best[0], best[0])
    np.testing.assert_array_equal(no_history_best[1], best[1])


@pytest.mark.parametrize('batch_size,num_search', [(0, 7), (4, 0), (4, -1)])
def test_search_rejects_invalid_budgets(batch_size, num_search):
    with pytest.raises(ValueError, match='positive integers'):
        BayesianOptimiser._random_search(
            jax.random.PRNGKey(0), lambda u: u.sum(), 2, batch_size, num_search)


def test_one_trial_and_flat_objective_continue_exploring(monkeypatch):
    space = ParameterSpace(parameters=[Parameter(
        name='x', prior=ContinuousPrior(lower=-1., upper=1., mode=0., uncert=np.inf))])
    service = BayesianOptimisation.create_new_experiment(
        NewExperimentRequest(parameter_space=space, init_explore_size=1))

    def unexpected_solve(*args, **kwargs):
        pytest.fail('GP empirical scale is undefined during warm-up')

    import bojaxns.gaussian_process_formulation.bayesian_optimiser as module
    monkeypatch.setattr(module.NestedSampler, 'run', unexpected_solve)
    for index in range(4):
        trial_id = service.create_new_trial(jax.random.PRNGKey(index))
        trial = service.get_trial(trial_id)
        assert -1. <= trial.param_values['x'].value <= 1.
        assert len(service.experiment.trials) == index + 1
        service.post_measurement(trial_id, TrialUpdate(ref_id='measurement', objective_measurement=1.))


def test_multistep_search_does_not_render_by_default(monkeypatch, tmp_path):
    import bojaxns.gaussian_process_formulation.bayesian_optimiser as module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, 'run_multi_lookahead', lambda **kwargs: (jnp.array([0.5]), None))
    monkeypatch.setattr(module, 'convert_tree_to_graph',
                        lambda *args: pytest.fail('default search must not render'))
    result = BayesianOptimiser._multistep_lookahead_search(
        jax.random.PRNGKey(0), None, None, 1, 2, 4, 2, 2)
    np.testing.assert_array_equal(result, [0.5])
    assert not list(tmp_path.iterdir())


def test_failed_measurement_penalties_use_same_warmup_and_fit_data(monkeypatch):
    space = ParameterSpace([Parameter('x', ContinuousPrior(0., 1., .5, 1.))])
    service = BayesianOptimisation.create_new_experiment(
        NewExperimentRequest(space, init_explore_size=2))
    first, second = list(service.experiment.trials)
    service.post_measurement(first, TrialUpdate('good', 0.))
    service.post_measurement(second, TrialUpdate('good', 10.))
    for index in range(10):
        service.post_measurement(second, TrialUpdate(f'failed-{index}', float('nan')))
    # Each failed measurement is penalised to -1: both prepared means are zero.
    np.testing.assert_array_equal(BayesianOptimiser(service.experiment)._data.Y, [0., 0.])
    import bojaxns.gaussian_process_formulation.bayesian_optimiser as module
    monkeypatch.setattr(module.NestedSampler, 'run',
                        lambda *args, **kwargs: pytest.fail('Flat prepared data must not be fitted'))
    proposal = service.create_new_trial(jax.random.PRNGKey(0))
    assert proposal not in (first, second)
    assert 0. <= service.get_trial(proposal).param_values['x'].value <= 1.
