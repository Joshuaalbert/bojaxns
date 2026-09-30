from abc import abstractmethod
from typing import NamedTuple, Dict, Callable

import chex
from jax import numpy as jnp, vmap
from jax.tree_util import tree_map
from jax.random import PRNGKey
from jaxns.random_utils import resample


class AbstractAcquisition:
    """
    A class that represents any acquisition function. All acquisition functions take a point in the U-domain
    and returns a metric that gives a proxy as to how valuable it would be to try that point.
    All acquisition values only make sense relatively.
    """

    @abstractmethod
    def __call__(self, u_star: jnp.ndarray):
        ...


def _assert_rank(rank: int, **kwargs):
    for name, t in kwargs.items():
        if len(t.shape) != rank:
            raise ValueError(f"{name} shoue be rank {rank} got {t.shape}.")


def _assert_same_leading_dim(*args):
    n = set()
    for arg in args:
        n.add(arg.shape[0])
    if len(n) > 1:
        raise ValueError(f"Got mismatched leading dimensions: {n}")


class ConditionalPredictive:

    @abstractmethod
    def _ndims(self):
        ...

    @property
    def ndims(self):
        return self._ndims()

    @abstractmethod
    def posterior(self):
        ...

    @abstractmethod
    def marginal_likelihood(self):
        ...

    @abstractmethod
    def __call__(self, U_star: jnp.ndarray, cov: bool = False):
        ...


class MarginalisationData(NamedTuple):
    samples: Dict[str, chex.Array]  # [S, ...] per posterior parameter
    log_dp_mean: chex.Array  # [S] log posterior weights


class ConditionalPredictiveFactory:

    @abstractmethod
    def ndims(self):
        ...

    @abstractmethod
    def build_prior_model(self) -> Callable:
        ...

    @abstractmethod
    def __call__(self, **samples) -> ConditionalPredictive:
        ...


class AcquisitionFactory:

    @abstractmethod
    def __call__(self, **sample) -> AbstractAcquisition:
        ...


class MarginalisedAcquisitionFunction(AbstractAcquisition):
    """
    Class that represents a marginalisation of an acquisition function over samples.
    """

    def __init__(self, key: PRNGKey, ns_results: MarginalisationData, acquisition_factory: AcquisitionFactory, S: int):
        self._acquisition_factory = acquisition_factory
        self._key = key
        self._ns_results = ns_results
        self._S = int(S)

    def __call__(self, u_star: jnp.ndarray):
        def _eval(**sample):
            acquisition = self._acquisition_factory(**sample)
            return acquisition(u_star=u_star)

        samples = resample(self._key, self._ns_results.samples, self._ns_results.log_dp_mean,
                           S=self._S, replace=True)
        marginalised = tree_map(lambda marg: jnp.nanmean(marg, axis=0), vmap(_eval)(**samples))
        return marginalised


def _mixture_moments(means: chex.Array, uncertainties: chex.Array, cov: bool = False):
    """Combine equally weighted resampled components by total covariance."""
    mean = jnp.mean(means, axis=0)  # [N], from component means [S, N]
    centred = means - mean
    if cov:
        between = centred.T @ centred / means.shape[0]  # [N, N], population covariance
    else:
        between = jnp.mean(jnp.square(centred), axis=0)  # [N]
    # Invalid components propagate: dropping different samples for each moment
    # would no longer describe one posterior mixture.
    return mean, jnp.mean(uncertainties, axis=0) + between


class MarginalisedConditionalPredictive(ConditionalPredictive):
    """
    Predictive mixture including uncertainty between posterior parameter samples.
    """

    def __init__(self, key: PRNGKey, ns_results: MarginalisationData,
                 conditional_predictive_factory: ConditionalPredictiveFactory,
                 S: int):
        self._conditional_predictive_factory = conditional_predictive_factory
        self._key = key
        self._ns_results = ns_results
        self._S = int(S)

    def _ndims(self):
        return self._conditional_predictive_factory.ndims()

    def posterior(self):
        def _eval(**sample):
            conditional_predictive = self._conditional_predictive_factory(**sample)
            return conditional_predictive.posterior()

        samples = resample(self._key, self._ns_results.samples, self._ns_results.log_dp_mean,
                           S=self._S, replace=True)
        return _mixture_moments(*vmap(_eval)(**samples))

    def marginal_likelihood(self):
        """Return the posterior-resampled mean log likelihood, not evidence.

        This averages log p(Y | theta), not likelihoods before taking a log.
        NaN evaluations are omitted, preserving the existing averaging policy.
        """
        def _eval(**sample):
            conditional_predictive = self._conditional_predictive_factory(**sample)
            return conditional_predictive.marginal_likelihood()

        samples = resample(self._key, self._ns_results.samples, self._ns_results.log_dp_mean,
                           S=self._S, replace=True)
        marginalised = tree_map(lambda marg: jnp.nanmean(marg, axis=0), vmap(_eval)(**samples))
        return marginalised

    def __call__(self, U_star: jnp.ndarray, cov: bool = False):
        def _eval(**sample):
            conditional_predictive = self._conditional_predictive_factory(**sample)
            return conditional_predictive(U_star=U_star, cov=cov)

        samples = resample(self._key, self._ns_results.samples, self._ns_results.log_dp_mean,
                           S=self._S, replace=True)
        return _mixture_moments(*vmap(_eval)(**samples), cov=cov)
