"""Parameter schemas and their released JAXNS 3 prior realisation."""
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from math import isfinite, isinf
from typing import Callable, List, Literal, Union

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import random
from jaxctx import CtxParams, transform
from jaxctx.context import set_state
from jaxctx.priors.special_priors import Categorical
from jaxns.model import Model
from jaxns.mixed_precision import mp_policy
from jaxns.priors import Prior
from jaxns.pytree import PureDataclassPytree

from bojaxns.common import FloatValue, IntValue, ParamValues, UValue, finite_float, integer, real_scalar

__all__ = ['ContinuousPrior', 'IntegerPrior', 'CategoricalPrior', 'Parameter',
           'ParameterSpace', 'build_prior_model', 'build_parameter_model', 'ParameterModel']

tfpd = tfp.distributions


def _validate_interval(prior, tag: str) -> None:
    if prior.type != tag:
        raise ValueError(f'Expected type {tag}.')
    prior.mode = finite_float(prior.mode, 'mode')
    prior.uncert = real_scalar(prior.uncert, 'uncert')
    if not prior.uncert > 0:
        raise ValueError('uncert must be positive (positive infinity means uniform).')
    if prior.lower > prior.upper or (tag == 'continuous_prior' and prior.lower == prior.upper):
        raise ValueError('Prior bounds must be ordered and continuous intervals must have positive width.')


@dataclass(slots=True)
class ContinuousPrior(PureDataclassPytree):
    lower: float  # [] lower bound.
    upper: float  # [] upper bound.
    mode: float  # [] normal location.
    uncert: float  # [] scale; positive infinity selects a uniform prior.
    type: Literal['continuous_prior'] = 'continuous_prior'

    @classmethod
    def flatten(cls, this):
        return cls.build_flatten(this, ['type'])

    def validate(self) -> None:
        self.lower = finite_float(self.lower, 'lower')
        self.upper = finite_float(self.upper, 'upper')
        _validate_interval(self, 'continuous_prior')


@dataclass(slots=True)
class IntegerPrior(PureDataclassPytree):
    lower: int  # [] inclusive lower bound.
    upper: int  # [] inclusive upper bound.
    mode: float  # [] normal location.
    uncert: float  # [] scale; positive infinity selects a uniform prior.
    type: Literal['integer_prior'] = 'integer_prior'

    @classmethod
    def flatten(cls, this):
        return cls.build_flatten(this, ['type'])

    def validate(self) -> None:
        self.lower = integer(self.lower, 'lower')
        self.upper = integer(self.upper, 'upper')
        _validate_interval(self, 'integer_prior')


@dataclass(slots=True)
class CategoricalPrior(PureDataclassPytree):
    probs: List[float]  # [K] unnormalised category weights.
    type: Literal['categorical_prior'] = 'categorical_prior'

    @classmethod
    def flatten(cls, this):
        return cls.build_flatten(this, ['type'])

    def validate(self) -> None:
        if self.type != 'categorical_prior':
            raise ValueError('Expected type categorical_prior.')
        self.probs = [finite_float(p, 'probability') for p in self.probs]
        if not self.probs or any(p < 0 for p in self.probs) or not any(p > 0 for p in self.probs):
            raise ValueError('Categorical probabilities must be nonnegative with positive total mass.')


ParamPrior = Union[ContinuousPrior, IntegerPrior, CategoricalPrior]


@dataclass(slots=True)
class Parameter(PureDataclassPytree):
    name: str
    prior: ParamPrior

    @classmethod
    def flatten(cls, this):
        return cls.build_flatten(this, ['name'])

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('Parameter name must be a nonempty string.')
        if isinstance(self.prior, dict):
            cls = {'continuous_prior': ContinuousPrior, 'integer_prior': IntegerPrior,
                   'categorical_prior': CategoricalPrior}.get(self.prior.get('type'))
            if cls is None:
                raise ValueError('Unknown prior type.')
            self.prior = cls(**self.prior)
        if not isinstance(self.prior, (ContinuousPrior, IntegerPrior, CategoricalPrior)):
            raise ValueError('Unsupported parameter prior.')
        self.prior.validate()


@dataclass(slots=True)
class ParameterSpace(PureDataclassPytree):
    parameters: List[Parameter]

    def validate(self) -> None:
        self.parameters = [Parameter(**p) if isinstance(p, dict) else p for p in self.parameters]
        if not self.parameters or not all(isinstance(p, Parameter) for p in self.parameters):
            raise ValueError('Parameter space must contain parameters.')
        for parameter in self.parameters:
            parameter.validate()
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError(f'parameter names must be unique. Got {names}.')


ContinuousPrior.register_pytree()
IntegerPrior.register_pytree()
CategoricalPrior.register_pytree()
Parameter.register_pytree()
ParameterSpace.register_pytree()


def _continuous_distribution(prior: ContinuousPrior):
    lower = jnp.asarray(prior.lower, dtype=mp_policy.measure_dtype)
    upper = jnp.asarray(prior.upper, dtype=mp_policy.measure_dtype)
    if isinf(prior.uncert):
        return tfpd.Uniform(low=lower, high=upper)
    return tfpd.TruncatedNormal(loc=jnp.asarray(prior.mode, dtype=mp_policy.measure_dtype),
                                scale=jnp.asarray(prior.uncert, dtype=mp_policy.measure_dtype),
                                low=lower, high=upper)


def _integer_logits(prior: IntegerPrior) -> jax.Array:
    options = jnp.arange(prior.lower, prior.upper + 1, dtype=mp_policy.measure_dtype)
    if isinf(prior.uncert):
        return jnp.zeros_like(options)
    # The shared normalisation constant cancels in the categorical distribution.
    return -0.5 * jnp.square((options - prior.mode) / prior.uncert)


def translate_parameter(param: Parameter) -> jax.Array:
    """Realise a parameter inside an active JAXCTX model context."""
    prior = param.prior
    if isinstance(prior, ContinuousPrior):
        return Prior(_continuous_distribution(prior), name=param.name).realise()
    if isinstance(prior, IntegerPrior):
        index = Categorical('cdf', logits=_integer_logits(prior), name=param.name).realise()
        value = index + prior.lower
        # Keep the categorical U and log mass, but expose the physical integer.
        set_state(name=param.name, collection='X', value=value)
        return value
    return Categorical('gumbel_max', logits=jnp.log(jnp.asarray(prior.probs, dtype=mp_policy.measure_dtype)), name=param.name).realise()


def build_prior_model(parameter_space: ParameterSpace) -> Callable:
    """Build a JAXNS 3 callable whose named priors have zero log likelihood."""
    # Snapshot host schemas so subsequent editing cannot invalidate JIT constants.
    parameter_space.validate()
    parameters = deepcopy(parameter_space.parameters)

    def prior_model():
        for parameter in parameters:
            translate_parameter(parameter)
        return jnp.asarray(0.)

    return prior_model


@dataclass(frozen=True)
class ParameterModel:
    """Flat-U adapter; coordinates follow parameter declaration order.

    Each continuous/integer parameter occupies one coordinate. Each categorical
    parameter occupies K coordinates, in category order. JAXCTX's sorted pytree
    order is deliberately not part of the persisted experiment contract.
    Flat transforms use measure precision (float64), while ``model.sample_U``
    and ``model.transform_to_X`` use JAXNS 3's native float32 base measure.
    """
    model: Model
    U_ndims: int
    names: tuple
    base_shapes: tuple

    def transform(self, U: jax.Array) -> dict:
        U = jnp.asarray(U, dtype=mp_policy.measure_dtype)  # [D] flat unit coordinates
        if U.shape != (self.U_ndims,):
            raise ValueError(f'Expected U shape {(self.U_ndims,)}, got {U.shape}.')
        values, offset = {}, 0
        for name, shape in zip(self.names, self.base_shapes):
            size = int(np.prod(shape))
            values[name] = U[offset:offset + size].reshape(shape)
            offset += size
        # JAXNS Model's native U is float32. Persisted optimisation coordinates
        # retain measure precision through the public JAXCTX transform instead.
        applied = transform(self.model.prior_model, base_dtype=mp_policy.measure_dtype).apply(
            None, {'U': CtxParams(values), 'params': CtxParams(),
                   'X': CtxParams(), 'log_prob': CtxParams()})
        return applied.collections['X'].to_dict()



def build_parameter_model(parameter_space: ParameterSpace) -> ParameterModel:
    """Construct the model and stable flat-U adapter once at the host boundary."""
    parameter_space.validate()
    shapes = tuple((len(p.prior.probs),) if isinstance(p.prior, CategoricalPrior) else ()
                   for p in parameter_space.parameters)
    dimension = sum(int(np.prod(shape)) for shape in shapes)
    return ParameterModel(Model(build_prior_model(parameter_space)), dimension,
                          tuple(p.name for p in parameter_space.parameters), shapes)


@partial(jax.jit, static_argnames=['parametrisation'])
def _sample_U_categorical(key, logits, target_cat, parametrisation):
    probabilities = jax.nn.softmax(logits)
    if parametrisation == 'cdf':
        # JAXCTX uses a left-continuous search: sample strictly above the lower
        # boundary so the preceding category cannot be returned on a PRNG zero.
        cumulative = jnp.cumsum(probabilities)
        low = jnp.where(target_cat == 0, 0., cumulative[jnp.maximum(target_cat - 1, 0)])
        high = cumulative[target_cat]
        u = low + (high - low) * random.uniform(key, dtype=logits.dtype)
        return jnp.minimum(high, jnp.maximum(jnp.nextafter(low, high), u)).reshape((1,))
    # Independent exponential clocks are equivalent to Gumbel-max. Conditional
    # on the winner, the first arrival has rate sum(p)=1; the remaining clocks
    # have independent exponential residuals. This is bounded work even for
    # arbitrarily unlikely categories (unlike rejection sampling).
    time_key, residual_key = random.split(key)
    arrival = random.exponential(time_key, dtype=logits.dtype)
    residual = random.exponential(residual_key, shape=logits.shape, dtype=logits.dtype)
    residual = residual.at[target_cat].set(0.)
    u = jnp.exp(-probabilities * arrival - residual)
    return jnp.clip(u, jnp.finfo(logits.dtype).tiny, jnp.nextafter(jnp.asarray(1., logits.dtype), 0.))


def sample_U_categorical(key, logits, target_cat,
                         parametrisation: Literal['cdf', 'gumbel_max']) -> jax.Array:
    """Sample conditional U on the host; reject impossible/unrepresentable targets."""
    logits = jnp.asarray(logits, dtype=mp_policy.measure_dtype)
    host_logits = np.asarray(logits)
    target = integer(np.asarray(target_cat).item(), 'target category')
    if parametrisation not in ('cdf', 'gumbel_max'):
        raise ValueError('Unknown categorical parametrisation.')
    if (host_logits.ndim != 1 or len(host_logits) == 0 or np.isnan(host_logits).any()
            or np.isposinf(host_logits).any() or not 0 <= target < len(host_logits)
            or not isfinite(host_logits[target])):
        raise ValueError('Target category must have positive finite probability and be in range.')
    u = _sample_U_categorical(key, logits, target, parametrisation)
    # Extremely small masses may have no representable U in the active dtype.
    # Fail explicitly rather than recording coordinates for a different value.
    realised = Categorical(parametrisation, logits=logits).forward(u.reshape(()) if parametrisation == 'cdf' else u)
    if int(realised) != target:
        raise ValueError('Target category has no sampled representable U in the active dtype.')
    return u


def inverse_transform_param(key, param: Parameter, param_value: Union[FloatValue, IntValue]) -> List[float]:
    param.validate()
    if not isinstance(param_value, (FloatValue, IntValue)):
        raise ValueError('Expected FloatValue or IntValue.')
    param_value.validate()
    prior = param.prior
    value = param_value.value
    if isinstance(prior, ContinuousPrior):
        if not isinstance(param_value, FloatValue) or not prior.lower <= value <= prior.upper:
            raise ValueError('Continuous value must be a FloatValue within its bounds.')
        return _continuous_distribution(prior).cdf(value).reshape((-1,)).tolist()
    if not isinstance(param_value, IntValue):
        raise ValueError('Discrete value must be an IntValue.')
    if isinstance(prior, IntegerPrior):
        if not prior.lower <= value <= prior.upper:
            raise ValueError('Integer value outside its bounds.')
        return sample_U_categorical(key, _integer_logits(prior), value - prior.lower, 'cdf').tolist()
    return sample_U_categorical(key, jnp.log(jnp.asarray(prior.probs, dtype=mp_policy.measure_dtype)), value, 'gumbel_max').tolist()


def sample_U_value(key, param_space: ParameterSpace, param_values: ParamValues) -> UValue:
    """Invert validated physical values in the same order as ParameterModel."""
    param_space.validate()
    if set(param_values) != {p.name for p in param_space.parameters}:
        raise ValueError('Parameter values must match parameter space names.')
    result = []
    for param in param_space.parameters:
        key, sample_key = random.split(key)
        result.extend(inverse_transform_param(sample_key, param, param_values[param.name]))
    return result
