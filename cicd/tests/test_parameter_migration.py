"""Released JAXNS/JAXCTX parameter adapter and inverse regressions."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from jaxctx import CtxParams
from jaxns.model import Model

from bojaxns.common import FloatValue, IntValue
from bojaxns.parameter_space import (CategoricalPrior, ContinuousPrior, IntegerPrior, Parameter,
                                     ParameterSpace, build_parameter_model, build_prior_model,
                                     inverse_transform_param, sample_U_categorical, sample_U_value)


def test_flat_adapter_preserves_declared_order_and_native_physical_values():
    space = ParameterSpace([Parameter('z', IntegerPrior(-3, 2, 0, float('inf'))),
                            Parameter('a', CategoricalPrior([0., 1., 2.])),
                            Parameter('q', ContinuousPrior(1., 4., 2., float('inf')))])
    adapter = build_parameter_model(space)
    assert isinstance(adapter.model, Model)
    assert adapter.U_ndims == adapter.model.U_ndims() == 5
    u = jnp.asarray([.4, .3, .6, .7, .2], jnp.float32)
    native = adapter.model.transform_to_X(CtxParams({'z': u[0], 'a': u[1:4], 'q': u[4]})).to_dict()
    actual = jax.jit(adapter.transform)(u)
    assert int(actual['z']) == -1
    assert float(actual['q']) == pytest.approx(1.6)
    for name in native:
        np.testing.assert_allclose(actual[name], native[name])
    with pytest.raises(ValueError, match='shape'):
        jax.jit(adapter.transform)(jnp.ones((1, 5)))
    model = Model(build_prior_model(space))
    assert float(model.log_likelihood(model.sample_U(random.PRNGKey(0)))) == 0.


def test_nonzero_integer_and_mixed_inverse_roundtrip():
    space = ParameterSpace([Parameter('z', IntegerPrior(-3, 2, 0, 1.3)),
                            Parameter('a', CategoricalPrior([0., 1., 2.])),
                            Parameter('q', ContinuousPrior(1., 4., 2., 1.))])
    adapter = build_parameter_model(space)
    for integer in range(-3, 3):
        for category in (1, 2):
            values = {'z': IntValue(integer), 'a': IntValue(category), 'q': FloatValue(2.4)}
            u = sample_U_value(random.PRNGKey(42), space, values)
            actual = adapter.transform(jnp.asarray(u))
            assert len(u) == 5
            for name, value in values.items():
                assert float(actual[name]) == pytest.approx(value.value, abs=1e-6)
    for lower in (3, -8):
        p = Parameter('i', IntegerPrior(lower, lower + 2, lower + 1, float('inf')))
        adapter = build_parameter_model(ParameterSpace([p]))
        for value in range(lower, lower + 3):
            u = inverse_transform_param(random.PRNGKey(0), p, IntValue(value))
            assert int(adapter.transform(jnp.asarray(u))['i']) == value


def test_uniform_infinite_uncertainty_and_singleton_categories():
    space = ParameterSpace([Parameter('u', ContinuousPrior(-2., 4., 0., float('inf'))),
                            Parameter('i', IntegerPrior(7, 7, 7, float('inf'))),
                            Parameter('c', CategoricalPrior([1.]))])
    adapter = build_parameter_model(space)
    assert adapter.U_ndims == 3
    for endpoint in (0., 1.):
        actual = adapter.transform(jnp.asarray([endpoint, .5, .5]))
        assert float(actual['u']) == -2 + 6 * endpoint
        assert int(actual['i']) == 7
        assert int(actual['c']) == 0


def test_impossible_inverse_fails_and_rare_category_has_finite_direct_sample():
    for target in (0, -1, 3):
        with pytest.raises(ValueError):
            sample_U_categorical(random.PRNGKey(0), jnp.log(jnp.asarray([0., 1., 2.])), target, 'gumbel_max')
    parameter = Parameter('c', CategoricalPrior([0., 1e-4, 1.]))
    adapter = build_parameter_model(ParameterSpace([parameter]))
    for seed in range(12):
        u = inverse_transform_param(random.PRNGKey(seed), parameter, IntValue(1))
        assert np.isfinite(u).all()
        assert int(adapter.transform(jnp.asarray(u))['c']) == 1
    with pytest.raises(ValueError):
        inverse_transform_param(random.PRNGKey(0), Parameter('i', IntegerPrior(4, 8, 6, 1)), IntValue(3))
    with pytest.raises(ValueError):
        sample_U_value(random.PRNGKey(0), ParameterSpace([parameter]), {})


def test_flat_precision_and_native_model_dtype_are_separate_contracts():
    space = ParameterSpace([Parameter('u', ContinuousPrior(0., 1., .5, float('inf')))])
    adapter = build_parameter_model(space)
    u = jnp.asarray([0.5 + 1e-10], dtype=jnp.float64)
    actual = jax.jit(adapter.transform)(u)['u']
    assert actual.dtype == jnp.float64
    assert float(actual) == float(u[0])
    native_u = adapter.model.sample_U(random.PRNGKey(0))
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(native_u))
    native_x = adapter.model.transform_to_X(native_u).to_dict()
    assert 0 <= float(native_x['u']) <= 1
    # The adapter snapshots its schema so mutable service state cannot silently
    # change already compiled transforms. Build a new adapter after edits.
    space.parameters[0].prior.upper = 3.
    assert float(adapter.transform(u)['u']) == float(u[0])
