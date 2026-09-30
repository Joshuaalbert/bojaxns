from jax import vmap, random, numpy as jnp
from jaxns.mixed_precision import mp_policy

float_type = mp_policy.measure_dtype

from bojaxns.common import FloatValue, IntValue
from bojaxns.parameter_space import IntegerPrior, CategoricalPrior, ParameterSpace, \
    Parameter, build_parameter_model, ContinuousPrior, sample_U_value


def test_serialisation():
    for prior in (IntegerPrior(0, 5, 2.5, 2.), CategoricalPrior([.1, .3, .6]),
                  ContinuousPrior(.1, 5.5, 2.5, 2.)):
        prior.validate()
        assert prior == type(prior).from_json(prior.to_json())


def test_build_prior():
    def _sample_prior(key, prior_model):
        U_ndims = prior_model.U_ndims
        U = random.uniform(key=key, shape=(U_ndims,), dtype=float_type)
        X = prior_model.transform(U)
        return X

    parameter_space = ParameterSpace(parameters=[
        Parameter(
            name='continuous',
            prior=ContinuousPrior(
                lower=0.,
                upper=5.,
                mode=4.,
                uncert=1.3
            )
        ),
        Parameter(
            name='integers',
            prior=IntegerPrior(
                lower=0,
                upper=5,
                mode=4.,
                uncert=1.3
            )
        ),
        Parameter(
            name='categorical',
            prior=CategoricalPrior(
                probs=[1., 1., 1.]
            )
        )
    ])
    prior_model = build_parameter_model(parameter_space=parameter_space)

    X = vmap(lambda key: _sample_prior(key=key, prior_model=prior_model))(random.split(random.PRNGKey(42), 1000))

    x = X['continuous']
    assert jnp.all(x >= 0.)
    assert jnp.all(x <= 5.)
    assert jnp.any(x < 1.)
    assert jnp.any(x > 4.)

    x = X['integers']

    assert jnp.all(x >= 0)
    assert jnp.all(x <= 5)
    assert jnp.any(x == 0)
    assert jnp.any(x == 5)

    x = X['categorical']
    assert jnp.all(x >= 0)
    assert jnp.all(x <= 2)
    assert jnp.any(x == 0)
    assert jnp.any(x == 1)
    assert jnp.any(x == 2)


def test_decision_variable():
    def _sample_prior(key, prior_model):
        U_ndims = prior_model.U_ndims
        U = random.uniform(key=key, shape=(U_ndims,), dtype=float_type)
        X = prior_model.transform(U)
        return X

    parameter_space = ParameterSpace(parameters=[
        Parameter(
            name='categorical',
            prior=CategoricalPrior(
                probs=[1., 1.]
            )
        )
    ])
    prior_model = build_parameter_model(parameter_space=parameter_space)

    X = vmap(lambda key: _sample_prior(key=key, prior_model=prior_model))(random.split(random.PRNGKey(42), 1000))

    x = X['categorical']
    assert jnp.all(x >= 0)
    assert jnp.all(x <= 1)
    assert jnp.any(x == 0)
    assert jnp.any(x == 1)


def test_close_to_zero():
    def _sample_prior(key, prior_model):
        U_ndims = prior_model.U_ndims
        U = random.uniform(key=key, shape=(U_ndims,), dtype=float_type)
        X = prior_model.transform(U)
        return X

    parameter_space = ParameterSpace(parameters=[
        Parameter(
            name='churn_rate',
            prior=ContinuousPrior(
                lower=1 / 20.,
                upper=1 / 10.,
                mode=1 / 15.,
                uncert=1 / 15.
            )
        )
    ])

    prior_model = build_parameter_model(parameter_space=parameter_space)

    X = vmap(lambda key: _sample_prior(key=key, prior_model=prior_model))(random.split(random.PRNGKey(42), 1000))

    x = X['churn_rate']
    assert jnp.all(x >= 1 / 20)
    assert jnp.all(x <= 1 / 10)


def test_parameter_space():
    param1 = Parameter(
        name='continuous',
        prior=ContinuousPrior(.1, 5.5, 2.5, 2.)
    )

    param2 = Parameter(
        name='integers',
        prior=IntegerPrior(0, 5, 2.5, 2.)
    )

    param3 = Parameter(
        name='categorical',
        prior=CategoricalPrior([.1, .3, .6])
    )

    _ = ParameterSpace(parameters=[param1, param2, param3])
    try:
        ParameterSpace(parameters=[param1, param1]).validate()
        assert False
    except ValueError as e:
        assert 'parameter names must be unique' in str(e)


def test_sample_U_value():
    param1 = Parameter(
        name='continuous',
        prior=ContinuousPrior(.1, 5.5, 2.5, 2.)
    )

    param2 = Parameter(
        name='integers',
        prior=IntegerPrior(0, 5, 2.5, 2.)
    )

    param3 = Parameter(
        name='categorical',
        prior=CategoricalPrior([.1, .3, .6])
    )

    parameter_space = ParameterSpace(parameters=[param1, param2, param3])

    param_values = {
        'continuous': FloatValue(value=param1.prior.lower),
        'integers': IntValue(value=param2.prior.lower),
        'categorical': IntValue(value=0)
    }
    U_sample = sample_U_value(key=random.PRNGKey(42), param_space=parameter_space,
                              param_values=param_values)

    prior_model = build_parameter_model(parameter_space=parameter_space)
    X = prior_model.transform(jnp.asarray(U_sample))
    for key in X:
        assert jnp.allclose(X[key], param_values[key].value)

    param_values = {
        'continuous': FloatValue(value=param1.prior.upper),
        'integers': IntValue(value=param2.prior.upper),
        'categorical': IntValue(value=0)
    }
    U_sample = sample_U_value(key=random.PRNGKey(42), param_space=parameter_space,
                              param_values=param_values)

    prior_model = build_parameter_model(parameter_space=parameter_space)
    X = prior_model.transform(jnp.asarray(U_sample))
    for key in X:
        assert jnp.allclose(X[key], param_values[key].value)
