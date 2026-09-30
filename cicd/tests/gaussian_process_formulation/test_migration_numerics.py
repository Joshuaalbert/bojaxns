"""Numerical contracts for the JAXNS 3 GP migration."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxns.model import Model
from jaxns.random_utils import resample

from bojaxns.base import MarginalisationData, MarginalisedConditionalPredictive
from bojaxns.gaussian_process_formulation.distribution_math import (
    GaussianProcessConditionalPredictiveFactory,
    GaussianProcessData,
    NotEnoughData,
)


class _Component:
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, U_star, cov=False):
        variance = jnp.array([0.2, 0.7])
        return self.mean, jnp.diag(variance) if cov else variance

    def posterior(self):
        return self(None)


def _mixture():
    key = jax.random.PRNGKey(17)
    data = MarginalisationData(
        samples={'mean': jnp.array([[1., 4.], [5., -2.]])},
        log_dp_mean=jnp.log(jnp.array([0.3, 0.7])),
    )
    sample_count = 128
    sampled_means = resample(key, data.samples, data.log_dp_mean, S=sample_count)['mean']
    mixture = MarginalisedConditionalPredictive(key, data, _Component, sample_count)
    return mixture, sampled_means


@pytest.mark.parametrize('cov', [False, True])
def test_mixture_total_covariance(cov):
    mixture, means = _mixture()
    expected_mean = np.mean(means, axis=0)
    between = np.cov(means, rowvar=False, bias=True)
    expected_cov = np.diag([0.2, 0.7]) + between
    mean, uncertainty = jax.jit(lambda u: mixture(u, cov=cov))(jnp.zeros((2, 1)))
    np.testing.assert_allclose(mean, expected_mean)
    np.testing.assert_allclose(uncertainty, expected_cov if cov else np.diag(expected_cov))


def test_mixture_training_posterior_total_variance():
    mixture, means = _mixture()
    mean, variance = jax.jit(mixture.posterior)()
    np.testing.assert_allclose(mean, np.mean(means, axis=0))
    np.testing.assert_allclose(variance, np.array([0.2, 0.7]) + np.var(means, axis=0))


def _data():
    return GaussianProcessData(
        U=jnp.array([[0.2, 0.4], [0.8, 0.4], [0.5, 0.4]]),
        Y=jnp.array([-1., 2., 0.]),
        Y_var=jnp.array([0.1, jnp.nan, 0.3]),
        sample_size=jnp.array([2., 1., 4.]),
    )


def test_gp_v3_model_uses_runtime_data_and_constant_coordinate_domain():
    data = _data()
    factory = GaussianProcessConditionalPredictiveFactory(data)
    model_fn = factory.build_prior_model()
    assert model_fn is GaussianProcessConditionalPredictiveFactory(data).build_prior_model()
    assert model_fn.__closure__ is None
    model = Model(prior_model=model_fn)
    U = model.sample_U(jax.random.PRNGKey(42), args=(data,))
    X = model.transform_to_X(U, args=(data,)).to_dict()
    assert set(X) == {'amplitude', 'length_scale', 'variance', 'mean', 'kernel_select'}
    assert X['length_scale'].shape == (2,)
    assert 0. < X['length_scale'][0] < 0.6
    assert 0. < X['length_scale'][1] < 1.
    likelihood = model.log_likelihood(U, args=(data,))
    assert likelihood.shape == ()
    assert np.isfinite(likelihood)
    np.testing.assert_allclose(likelihood, factory(**X).marginal_likelihood())
    changed = data._replace(Y=data.Y * 3.)
    assert not np.isclose(likelihood, model.log_likelihood(U, args=(changed,)))


@pytest.mark.parametrize('observations', [jnp.ones(3), jnp.zeros(3)])
def test_flat_objective_rejected_at_prior_boundary(observations):
    factory = GaussianProcessConditionalPredictiveFactory(_data()._replace(Y=observations))
    with pytest.raises(NotEnoughData, match='variation'):
        factory.build_prior_model()


def test_nonfinite_empirical_prior_scale_rejected():
    # Finite observations can still overflow the empirical variance calculation.
    factory = GaussianProcessConditionalPredictiveFactory(
        _data()._replace(Y=jnp.array([-1e200, 0., 1e200])))
    with pytest.raises(NotEnoughData, match='finite, positive'):
        factory.build_prior_model()


def test_masked_padding_does_not_change_expected_improvement_incumbent():
    from bojaxns.gaussian_process_formulation.distribution_math import ExpectedImprovementAcquisition

    data = GaussianProcessData(
        U=jnp.array([[0.8], [1.]]), Y=jnp.array([-5., -3.]),
        Y_var=jnp.array([0.01, 0.01]), sample_size=jnp.ones(2),
    )
    padded = GaussianProcessData(
        U=jnp.concatenate([data.U, jnp.zeros((1, 1))]),
        Y=jnp.append(data.Y, 0.), Y_var=jnp.append(data.Y_var, jnp.inf),
        sample_size=jnp.append(data.sample_size, jnp.inf),
    )
    sample = dict(amplitude=jnp.array(1.), length_scale=jnp.array([0.2]),
                  variance=jnp.array(0.1), mean=jnp.array(0.), kernel_select=jnp.array(1))
    actual = ExpectedImprovementAcquisition(GaussianProcessConditionalPredictiveFactory(padded)(**sample))
    expected = ExpectedImprovementAcquisition(GaussianProcessConditionalPredictiveFactory(data)(**sample))
    np.testing.assert_allclose(jax.jit(actual)(jnp.array([0.])), expected(jnp.array([0.])))


@pytest.mark.parametrize('max_depth', [1, 3])
def test_lookahead_terminal_has_no_bootstrap_or_out_of_bounds_write(max_depth):
    from bojaxns.gaussian_process_formulation.multi_step_lookahead import _make_batched_env_model

    data = _data()
    ns_results = MarginalisationData(
        samples=dict(amplitude=jnp.ones(1), length_scale=jnp.ones((1, 2)), variance=jnp.ones(1),
                     mean=jnp.zeros(1), kernel_select=jnp.zeros(1, dtype=jnp.int32)),
        log_dp_mean=jnp.zeros(1),
    )
    root, recurrent = _make_batched_env_model(
        key=jax.random.PRNGKey(1), batch_size=1, num_actions=1, S=1, max_depth=max_depth,
        data=data, ns_results=ns_results, U_options=jnp.array([[0.3, 0.5]]),
    )
    step = jax.jit(recurrent)
    state = root.embedding
    for depth in range(max_depth - 1):
        output, state = step((), jax.random.PRNGKey(2), jnp.zeros(1, dtype=jnp.int32), state)
        np.testing.assert_array_equal(output.discount, [1.])
        np.testing.assert_array_equal(state.depth, [depth + 1])
        assert np.count_nonzero(np.isinf(state.data.Y_var)) == max_depth - 2 - depth
    before_terminal = state
    output, state = step((), jax.random.PRNGKey(2), jnp.zeros(1, dtype=jnp.int32), state)
    assert np.all(np.isfinite(output.reward))
    np.testing.assert_array_equal(output.discount, [0.])
    np.testing.assert_array_equal(output.value, [0.])
    np.testing.assert_array_equal(state.depth, [max_depth])
    for before, after in zip(before_terminal.data, state.data):
        np.testing.assert_array_equal(before, after)
    output, terminal = step((), jax.random.PRNGKey(3), jnp.zeros(1, dtype=jnp.int32), state)
    np.testing.assert_array_equal(output.reward, [0.])
    np.testing.assert_array_equal(terminal.depth, [max_depth])


def test_observation_variance_semantics_match_dense_conditioning():
    data = _data()
    factory = GaussianProcessConditionalPredictiveFactory(data)
    predictive = factory(amplitude=jnp.array(1.2), length_scale=jnp.array([0.3, 0.7]),
                         variance=jnp.array(0.4), mean=jnp.array(0.2), kernel_select=jnp.array(1))
    points = jnp.array([[0.3, 0.2], [0.7, 0.5]])
    # Known variances are not divided by n. The unknown entry uses inferred
    # variance, and every observation gets the legacy extra variance/sqrt(n).
    noise = np.array([0.1, 0.4, 0.3]) + 0.4 / np.sqrt([2., 1., 4.])
    U = np.asarray(data.U) / [0.3, 0.7]
    V = np.asarray(points) / [0.3, 0.7]

    def kernel(x, y):
        return 1.2 ** 2 * np.exp(-0.5 * np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1))

    matrix = kernel(U, U) + np.diag(noise)
    cross = kernel(U, V)
    expected_mean = 0.2 + cross.T @ np.linalg.solve(matrix, np.asarray(data.Y) - 0.2)
    expected_cov = kernel(V, V) - cross.T @ np.linalg.solve(matrix, cross)
    mean, covariance = jax.jit(lambda u: predictive(u, cov=True))(points)
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(covariance, expected_cov, rtol=1e-10, atol=1e-12)


def test_masked_observation_does_not_set_empirical_prior_scales():
    data = _data()
    padded = GaussianProcessData(
        U=jnp.concatenate([data.U, jnp.array([[0., 0.]])]),
        Y=jnp.append(data.Y, 100.), Y_var=jnp.append(data.Y_var, jnp.inf),
        sample_size=jnp.append(data.sample_size, jnp.inf),
    )
    factory = GaussianProcessConditionalPredictiveFactory(data)
    model = Model(prior_model=factory.build_prior_model())
    U = model.sample_U(jax.random.PRNGKey(3), args=(data,))
    actual = model.transform_to_X(U, args=(padded,)).to_dict()
    expected = model.transform_to_X(U, args=(data,)).to_dict()
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name])
    np.testing.assert_allclose(model.log_likelihood(U, args=(padded,)),
                               model.log_likelihood(U, args=(data,)))


def test_gp_prior_requires_two_unmasked_observations():
    factory = GaussianProcessConditionalPredictiveFactory(
        _data()._replace(Y_var=jnp.array([0.1, jnp.inf, jnp.inf])))
    with pytest.raises(NotEnoughData):
        factory.build_prior_model()


@pytest.mark.parametrize('masked', [False, True])
def test_diagonal_prediction_matches_full_covariance_under_jit(masked):
    data = GaussianProcessData(
        U=jnp.array([[0.], [.2], [.7], [1.]]), Y=jnp.array([-.2, .1, 1., .3]),
        Y_var=jnp.array([.1, jnp.inf if masked else .2, .3, .1]),
        sample_size=jnp.ones(4))
    factory = GaussianProcessConditionalPredictiveFactory(data)
    predictive = factory(amplitude=jnp.array(1.), length_scale=jnp.array([.4]),
                         variance=jnp.array(.1), mean=jnp.array(.2), kernel_select=jnp.array(1))
    points = jnp.linspace(0., 1., 9)[:, None]
    for method in ([predictive._posterior_with_mask] if masked else
                   [predictive._posterior_with_mask, predictive._posterior]):
        full_mean, full_cov = jax.jit(lambda u: method(u, cov=True))(points)
        mean, variance = jax.jit(lambda u: method(u, cov=False))(points)
        np.testing.assert_allclose(mean, full_mean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(variance, jnp.diag(full_cov), rtol=1e-12, atol=1e-12)
