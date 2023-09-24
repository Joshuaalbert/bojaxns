from functools import cached_property
from typing import NamedTuple, List, Type

from jax import numpy as jnp, tree_map
from jax._src.scipy.linalg import solve_triangular
from jaxns import PriorModelGen, Prior, Categorical
from jaxns.prior import PriorModelType
from jaxns.types import float_type
from tensorflow_probability.substrates import jax as tfp

from bojaxns.base import _assert_rank, _assert_same_leading_dim, ConditionalPredictive, ConditionalPredictiveFactory, \
    AbstractAcquisition, AcquisitionFactory

tfpd = tfp.distributions


def log_normal(x, mean, cov):
    L = jnp.linalg.cholesky(cov)
    # U, S, Vh = jnp.linalg.svd(cov)
    log_det = jnp.sum(jnp.log(jnp.diag(L)))  # jnp.sum(jnp.log(S))
    dx = x - mean
    dx = solve_triangular(L, dx, lower=True)
    # U S Vh V 1/S Uh
    # pinv = (Vh.T.conj() * jnp.where(S!=0., jnp.reciprocal(S), 0.)) @ U.T.conj()
    maha = dx @ dx  # dx @ pinv @ dx#solve_triangular(L, dx, lower=True)
    log_likelihood = -0.5 * x.size * jnp.log(2. * jnp.pi) - log_det - 0.5 * maha
    return log_likelihood


def log_normal_with_mask(x, mean, cov, sigma):
    """
    Computes log-Normal density in a numerically stable way so that sigma can contain +inf for masked data.

    Args:
        x: RV value
        mean: mean of Gaussian
        cov: covariance of underlying, minus the obs. covariance
        sigma: stddev's of obs. error, inf encodes an outlier.

    Returns: a normal density for all points not of inf stddev obs. error.
    """
    C = cov / (sigma[:, None] * sigma[None, :]) + jnp.eye(cov.shape[0])
    L = jnp.linalg.cholesky(C)
    Ls = sigma[:, None] * L
    log_det = jnp.sum(jnp.where(jnp.isinf(sigma), 0., jnp.log(jnp.diag(Ls))))
    dx = (x - mean)
    dx = solve_triangular(L, dx / sigma, lower=True)
    maha = dx @ dx
    log_likelihood = -0.5 * jnp.sum(~jnp.isinf(sigma)) * jnp.log(2. * jnp.pi) \
                     - log_det \
                     - 0.5 * maha
    return log_likelihood


class GaussianProcessData(NamedTuple):
    U: jnp.ndarray
    Y: jnp.ndarray
    Y_var: jnp.ndarray
    sample_size: jnp.ndarray

class NotEnoughData(Exception):
    pass

def _ensure_gaussian_process_data(data: GaussianProcessData) -> GaussianProcessData:
    data = tree_map(lambda x: jnp.asarray(x, float_type), data)
    _assert_rank(2, U=data.U)
    _assert_rank(1, sample_size=data.sample_size, Y=data.Y, Y_var=data.Y_var)
    _assert_same_leading_dim(*data)
    if data.Y.size < 2:
        raise NotEnoughData('Need more samples to form mean and variance of data.')
    return data


class GaussianProcessConditionalPredictive(ConditionalPredictive):
    def __init__(self,
                 data: GaussianProcessData,
                 kernel: tfp.math.psd_kernels.PositiveSemidefiniteKernel,
                 variance: jnp.ndarray,
                 mean: jnp.ndarray):
        self._data = _ensure_gaussian_process_data(data)
        self._kernel = kernel
        self._variance = variance
        self._mean = mean

    def _ndims(self):
        return self._data.U.shape[-1]

    def posterior(self):
        return self.__call__(self._data.U)

    def _marginal_likelihood_with_mask(self):
        Kxx = self._kernel.matrix(self._data.U, self._data.U)

        no_uncert_data = jnp.isnan(self._data.Y_var)

        variance = jnp.where(no_uncert_data,
                             self._variance + self._variance / self._data.sample_size,
                             self._data.Y_var + self._variance / self._data.sample_size)

        sigma = jnp.sqrt(jnp.maximum(1e-6, variance))
        return log_normal_with_mask(x=self._data.Y, mean=self._mean, cov=Kxx, sigma=sigma)

    def _marginal_likelihood(self):
        Kxx = self._kernel.matrix(self._data.U, self._data.U)

        no_uncert_data = jnp.isnan(self._data.Y_var)

        variance = jnp.where(no_uncert_data,
                             self._variance + self._variance / self._data.sample_size,
                             self._data.Y_var + self._variance / self._data.sample_size)

        data_cov = jnp.diag(variance)
        return log_normal(self._data.Y, self._mean, Kxx + data_cov)

    def marginal_likelihood(self):
        return self._marginal_likelihood_with_mask()

    def _posterior_with_mask(self, U_star: jnp.ndarray, cov: bool = False):
        Kxx = self._kernel.matrix(self._data.U, self._data.U)
        Kxs = self._kernel.matrix(self._data.U, U_star)
        Kss = self._kernel.matrix(U_star, U_star)

        no_uncert_data = jnp.isnan(self._data.Y_var)

        variance = jnp.where(no_uncert_data,
                             self._variance + self._variance / self._data.sample_size,
                             self._data.Y_var + self._variance / self._data.sample_size)
        std_dev = jnp.sqrt(jnp.maximum(1e-6, variance))

        L = jnp.linalg.cholesky(Kxx / (std_dev[:, None] * std_dev[None, :]) + jnp.eye(std_dev.size))
        # L = jnp.where(jnp.isnan(L), jnp.eye(L.shape[0])/sigma, L)

        J = solve_triangular(L, Kxs / std_dev[:, None],
                             lower=True)  # same J as below, but safely taking into account inf mask.

        post_cov_s = Kss - J.T @ J

        dY = self._data.Y - self._mean
        dX = solve_triangular(L, dY / std_dev, lower=True)
        post_mu_s = self._mean + J.T @ dX  # mu - J^T L^-1 dY = mu - J^T dX

        if cov:
            return post_mu_s, post_cov_s
        return post_mu_s, jnp.diag(post_cov_s)

    def _posterior(self, U_star: jnp.ndarray, cov: bool = False):
        Kxx = self._kernel.matrix(self._data.U, self._data.U)
        no_uncert_data = jnp.isnan(self._data.Y_var)

        variance = jnp.where(no_uncert_data,
                             self._variance + self._variance / self._data.sample_size,
                             self._data.Y_var + self._variance / self._data.sample_size)

        data_cov = jnp.diag(variance)

        Kxs = self._kernel.matrix(self._data.U, U_star)
        Kss = self._kernel.matrix(U_star, U_star)
        L = jnp.linalg.cholesky(Kxx + data_cov)
        # inv(LL^T) = L^-T L^-1
        J = solve_triangular(L, Kxs, lower=True)

        post_cov_s = Kss - J.T @ J

        H = solve_triangular(L.T, J, lower=False)
        dY = self._data.Y - self._mean
        post_mu_s = self._mean + H.T @ dY
        if cov:
            return post_mu_s, post_cov_s
        return post_mu_s, jnp.diag(post_cov_s)

    def __call__(self, U_star: jnp.ndarray, cov: bool = False):
        return self._posterior_with_mask(U_star=U_star, cov=cov)


class GaussianProcessConditionalPredictiveFactory(ConditionalPredictiveFactory):
    def __init__(self, data: GaussianProcessData):
        self._data = _ensure_gaussian_process_data(data)

    def ndims(self):
        return self._data.U.shape[-1]

    def build_prior_model(self) -> PriorModelType:
        amplitude_scale = 2 * jnp.std(self._data.Y)
        length_scale_scale = jnp.max(self._data.U, axis=0) - jnp.min(self._data.U, axis=0)
        variance_scale = jnp.std(self._data.Y)
        mean_loc = jnp.mean(self._data.Y)
        mean_scale = jnp.std(self._data.Y)

        def prior_model() -> PriorModelGen:
            amplitude = yield Prior(tfpd.Uniform(high=amplitude_scale), name='amplitude')
            length_scale = yield Prior(tfpd.Uniform(high=length_scale_scale), name='length_scale')
            variance = yield Prior(tfpd.Uniform(high=variance_scale), name='variance')  # measurement variance
            mean = yield Prior(tfpd.Normal(loc=mean_loc, scale=mean_scale), name='mean')
            kernel_select = yield Categorical(parametrisation='gumbel_max', logits=jnp.zeros(len(self.psd_kernels)),
                                              name='kernel_select')
            return amplitude, length_scale, variance, mean, kernel_select

        return prior_model

    @cached_property
    def psd_kernels(self) -> List[Type[tfp.math.psd_kernels.PositiveSemidefiniteKernel]]:
        return [
            tfp.math.psd_kernels.MaternThreeHalves,
            tfp.math.psd_kernels.ExponentiatedQuadratic,
            tfp.math.psd_kernels.MaternOneHalf
        ]

    def __call__(self, **samples) -> GaussianProcessConditionalPredictive:
        amplitude = samples.get('amplitude')
        length_scale = samples.get('length_scale')
        kernel_select = samples.get('kernel_select')
        mask = jnp.where(jnp.arange(len(self.psd_kernels)) == kernel_select, 1., 0.)

        kernels = []
        for i, psd_kernel in enumerate(self.psd_kernels):
            base_kernel = psd_kernel(amplitude=amplitude, length_scale=None)
            _kernel = tfp.math.psd_kernels.FeatureTransformed(base_kernel,
                                                              transformation_fn=lambda x, _1, _2: x / length_scale)
            _kernel = tfp.math.psd_kernels.Constant(mask[i]) * _kernel
            kernels.append(_kernel)
        kernel = sum(kernels[1:], kernels[0])

        variance = samples.get('variance')
        mean = samples.get('mean')
        return GaussianProcessConditionalPredictive(
            data=self._data,
            kernel=kernel,
            variance=variance,
            mean=mean
        )


class ExpectedImprovementAcquisition(AbstractAcquisition):
    """
    A class that represents the heteroscedastic expected improvement acquisition function.
    """

    def __init__(self, conditional_predictive: GaussianProcessConditionalPredictive):
        self._conditional_predictive = conditional_predictive

    @staticmethod
    def _expected_improvement(post_mu_x_max: jnp.ndarray, post_mu_s: jnp.ndarray,
                              post_var_s: jnp.ndarray) -> jnp.ndarray:
        post_stddev_s = jnp.sqrt(jnp.maximum(1e-6, post_var_s))
        posterior_pdf = tfpd.Normal(loc=0., scale=1.)
        u = (post_mu_s - post_mu_x_max) / post_stddev_s
        return post_stddev_s * (posterior_pdf.prob(u) + u * posterior_pdf.cdf(u))

    def __call__(self, u_star: jnp.ndarray):
        post_mu_x, post_var_x = self._conditional_predictive.posterior()
        post_mu_x_max = jnp.max(post_mu_x)
        post_mu_s, post_var_s = self._conditional_predictive(u_star[None, :])
        ei = ExpectedImprovementAcquisition._expected_improvement(
            post_mu_x_max=post_mu_x_max,
            post_mu_s=post_mu_s,
            post_var_s=post_var_s
        )
        return jnp.reshape(ei, ())


class ScaledExpectedImprovementAcquisition(AbstractAcquisition):
    """
    A class that represents the heteroscedastic expected improvement acquisition function.
    """

    def __init__(self, condition_predictive: GaussianProcessConditionalPredictive):
        self._condition_predictive = condition_predictive

    @staticmethod
    def _expected_squared_improvement(post_mu_x_max: jnp.ndarray, post_mu_s: jnp.ndarray, post_var_s: jnp.ndarray):
        post_stddev_s = jnp.sqrt(jnp.maximum(1e-6, post_var_s))
        posterior_pdf = tfpd.Normal(loc=0., scale=1.)
        u = (post_mu_s - post_mu_x_max) / post_stddev_s
        return post_var_s * (u * posterior_pdf.prob(u) + (u ** 2 + 1.) * posterior_pdf.cdf(u))

    def __call__(self, u_star: jnp.ndarray):
        post_mu_x, post_var_x = self._condition_predictive.posterior()
        post_mu_x_max = jnp.max(post_mu_x)

        post_mu_s, post_var_s = self._condition_predictive(u_star[None, :])
        ei2 = ScaledExpectedImprovementAcquisition._expected_squared_improvement(
            post_mu_x_max=post_mu_x_max,
            post_mu_s=post_mu_s,
            post_var_s=post_var_s
        )
        ei = ExpectedImprovementAcquisition._expected_improvement(
            post_mu_x_max=post_mu_x_max,
            post_mu_s=post_mu_s,
            post_var_s=post_var_s
        )
        scaled_ei = ei / jnp.sqrt(jnp.maximum(1e-6, ei2 - ei ** 2))
        return jnp.reshape(scaled_ei, ())


class TopTwoAcquisition(AbstractAcquisition):
    def __init__(self, condition_predictive: GaussianProcessConditionalPredictive, u1: jnp.ndarray):
        self._condition_predictive = condition_predictive
        u1 = jnp.asarray(u1, float_type)
        self._u1 = u1

    def __call__(self, u_star: jnp.ndarray):
        S = jnp.stack([u_star, self._u1], axis=0)
        post_mu_s, post_K_s = self._condition_predictive(S, cov=True)
        sigma2 = post_K_s[0, 0] + post_K_s[1, 1] - 2. * post_K_s[0, 1]

        ei = ExpectedImprovementAcquisition._expected_improvement(
            post_mu_x_max=post_mu_s[1],
            post_mu_s=post_mu_s[0],
            post_var_s=sigma2
        )
        return jnp.reshape(ei, ())


class ScaledTopTwoAcquisition(AbstractAcquisition):
    def __init__(self, condition_predictive: GaussianProcessConditionalPredictive, u1: jnp.ndarray):
        self._condition_predictive = condition_predictive
        u1 = jnp.asarray(u1, float_type)
        self._u1 = u1

    def __call__(self, u_star: jnp.ndarray):
        S = jnp.stack([u_star, self._u1], axis=0)
        post_mu_s, post_K_s = self._condition_predictive(S, cov=True)
        sigma2 = post_K_s[0, 0] + post_K_s[1, 1] - 2. * post_K_s[0, 1]

        ei2 = ScaledExpectedImprovementAcquisition._expected_squared_improvement(
            post_mu_x_max=post_mu_s[1],
            post_mu_s=post_mu_s[0],
            post_var_s=sigma2
        )
        ei = ExpectedImprovementAcquisition._expected_improvement(
            post_mu_x_max=post_mu_s[1],
            post_mu_s=post_mu_s[0],
            post_var_s=sigma2
        )
        scaled_ei = ei / jnp.sqrt(jnp.maximum(1e-6, ei2 - ei ** 2))
        return jnp.reshape(scaled_ei, ())


class ExpectedImprovementAcquisitionFactory(AcquisitionFactory):
    def __init__(self, conditional_predictive_factory: GaussianProcessConditionalPredictiveFactory):
        self._conditional_predictive_factory = conditional_predictive_factory

    def __call__(self, **sample) -> AbstractAcquisition:
        conditional_predictive = self._conditional_predictive_factory(**sample)
        return ExpectedImprovementAcquisition(conditional_predictive=conditional_predictive)


class ScaledExpectedImprovementAcquisitionFactory(AcquisitionFactory):
    def __init__(self, conditional_predictive_factory: GaussianProcessConditionalPredictiveFactory):
        self._conditional_predictive_factory = conditional_predictive_factory

    def __call__(self, **sample) -> AbstractAcquisition:
        conditional_predictive = self._conditional_predictive_factory(**sample)
        return ScaledExpectedImprovementAcquisition(condition_predictive=conditional_predictive)


class TopTwoAcquisitionFactory(AcquisitionFactory):
    def __init__(self, conditional_predictive_factory: GaussianProcessConditionalPredictiveFactory, u1: jnp.ndarray):
        self._conditional_predictive_factory = conditional_predictive_factory
        self._u1 = u1

    def __call__(self, **sample) -> AbstractAcquisition:
        conditional_predictive = self._conditional_predictive_factory(**sample)
        return TopTwoAcquisition(condition_predictive=conditional_predictive,
                                 u1=self._u1)


class ScaledTopTwoAcquisitionFactory(AcquisitionFactory):
    def __init__(self, conditional_predictive_factory: GaussianProcessConditionalPredictiveFactory, u1: jnp.ndarray):
        self._conditional_predictive_factory = conditional_predictive_factory
        self._u1 = u1

    def __call__(self, **sample) -> AbstractAcquisition:
        conditional_predictive = self._conditional_predictive_factory(**sample)
        return ScaledTopTwoAcquisition(condition_predictive=conditional_predictive,
                                       u1=self._u1)
