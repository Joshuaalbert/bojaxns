
import jax
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from bojaxns import GaussianProcessData
from bojaxns.gaussian_process_formulation.distribution_math import GaussianProcessConditionalPredictive


def test_broadcast_lengthscale():
    # Define your length scales.
    # For example, for a 3-dimensional feature vector:
    length_scale = jnp.asarray([1.0, 2.0, 3.0])

    # Create the kernel.
    base_kernel = tfp.math.psd_kernels.MaternOneHalf(amplitude=1.0, length_scale=None)
    kernel = tfp.math.psd_kernels.FeatureTransformed(base_kernel,
                                                     transformation_fn=lambda x, _1, _2: x / length_scale)
    X = jnp.ones((4, 3))
    K = kernel.matrix(X, X)
    print(K.shape)  # (4, 4)


def test_gaussian_conditional_predictive_no_infs():
    U = jnp.linspace(0., 10., 100)
    Y = jnp.sin(U)
    Y_var = jnp.ones_like(Y)
    sample_size = jnp.ones_like(Y)
    data = GaussianProcessData(
        U=U[:, None],
        Y=Y,
        Y_var=Y_var,
        sample_size=sample_size
    )
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=1., length_scale=jnp.pi)

    conditional_predictive = GaussianProcessConditionalPredictive(
        data=data,
        kernel=kernel,
        variance=jnp.asarray(1.),
        mean=jnp.asarray(1.)
    )
    ml_no_mask = conditional_predictive._marginal_likelihood()
    ml_mask = conditional_predictive._marginal_likelihood_with_mask()
    assert jnp.allclose(ml_no_mask, ml_mask)

    U_star = jnp.asarray([[0.52345],
                          [0.72347]])

    posterior_no_mask = conditional_predictive._posterior(U_star=U_star, cov=True)
    posterior_mask = conditional_predictive._posterior_with_mask(U_star=U_star, cov=True)
    print(posterior_mask)
    print(posterior_no_mask)
    for x, y in zip(posterior_mask, posterior_no_mask):
        assert jnp.allclose(x, y)


def test_gaussian_conditional_predictive_some_infs():
    U = jnp.linspace(0., 10., 100)
    Y = jnp.sin(U)
    Y_var = jnp.ones_like(Y)
    inf_mask = Y > jnp.percentile(Y, 95)
    Y_var = jnp.where(inf_mask, jnp.inf, Y_var)
    sample_size = jnp.ones_like(Y)
    data = GaussianProcessData(
        U=U[:, None],
        Y=Y,
        Y_var=Y_var,
        sample_size=sample_size
    )
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=1., length_scale=1)

    conditional_predictive = GaussianProcessConditionalPredictive(
        data=data,
        kernel=kernel,
        variance=jnp.asarray(1.),
        mean=jnp.asarray(1.)
    )
    ml_mask = conditional_predictive._marginal_likelihood_with_mask()
    assert jnp.isfinite(ml_mask)

    U_star = jnp.asarray([[0.52345],
                          [0.72347]])

    posterior_mask = conditional_predictive._posterior_with_mask(U_star=U_star, cov=True)
    for x in posterior_mask:
        assert jnp.all(jnp.isfinite(x))


def test_gaussian_conditional_predictive_lots_of_infs():
    U = jnp.linspace(0., 10., 100)
    Y = jnp.sin(U)
    Y_var = jnp.ones_like(Y)
    inf_mask = Y > jnp.percentile(Y, 50)
    Y_var = jnp.where(inf_mask, jnp.inf, Y_var)
    sample_size = jnp.ones_like(Y)
    data = GaussianProcessData(
        U=U[:, None],
        Y=Y,
        Y_var=Y_var,
        sample_size=sample_size
    )
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=1., length_scale=1)

    conditional_predictive = GaussianProcessConditionalPredictive(
        data=data,
        kernel=kernel,
        variance=jnp.asarray(1.),
        mean=jnp.asarray(1.)
    )
    ml_mask = conditional_predictive._marginal_likelihood_with_mask()
    assert jnp.isfinite(ml_mask)

    U_star = jnp.asarray([[0.52345],
                          [0.72347]])

    posterior_mask = conditional_predictive._posterior_with_mask(U_star=U_star, cov=True)
    for x in posterior_mask:
        assert jnp.all(jnp.isfinite(x))

    # Visually ensure it lines up. Comment out to skip.
    # mu, var = conditional_predictive._posterior_with_mask(U_star=data.U)
    #
    # import pylab as plt
    # plt.scatter(data.U, data.Y, c=inf_mask)
    # plt.plot(data.U[:, 0], mu)
    # plt.show()
