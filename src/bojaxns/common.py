"""Scalar pytrees with explicit host-side validation."""
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Dict, List, Literal, Union

import numpy as np
from jaxns.pytree import PureDataclassPytree


def real_scalar(value: Real, name: str) -> float:
    """Read a host scalar, including native JSON's zero-dimensional arrays."""
    value = np.asarray(value)
    if value.shape != ():
        raise ValueError(f'{name} must be a scalar number.')
    value = value.item()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be numeric.')
    return float(value)


def finite_float(value: Real, name: str) -> float:
    value = real_scalar(value, name)
    if not isfinite(value):
        raise ValueError(f'{name} must be a finite number.')
    return value


def integer(value: Integral, name: str) -> int:
    value = np.asarray(value)
    if value.shape != ():
        raise ValueError(f'{name} must be a scalar integer.')
    value = value.item()
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer.')
    return int(value)


@dataclass(slots=True)
class FloatValue(PureDataclassPytree):
    value: float  # [] scalar numeric leaf; tree transforms may add batch axes.
    type: Literal['float'] = 'float'

    @classmethod
    def flatten(cls, this):
        return cls.build_flatten(this, ['type'])

    def validate(self) -> None:
        if self.type != 'float':
            raise ValueError('FloatValue type must be float.')
        self.value = finite_float(self.value, 'value')


FloatValue.register_pytree()


@dataclass(slots=True)
class IntValue(PureDataclassPytree):
    value: int  # [] scalar numeric leaf; tree transforms may add batch axes.
    type: Literal['int'] = 'int'

    @classmethod
    def flatten(cls, this):
        return cls.build_flatten(this, ['type'])

    def validate(self) -> None:
        if self.type != 'int':
            raise ValueError('IntValue type must be int.')
        self.value = integer(self.value, 'value')


IntValue.register_pytree()


def parse_param_value(value):
    if isinstance(value, (FloatValue, IntValue)):
        return value
    if isinstance(value, dict):
        cls = {'float': FloatValue, 'int': IntValue}.get(value.get('type'))
        if cls is not None:
            return cls(**value)
    raise ValueError('Parameter value must have a float or int type tag.')


ParamValues = Dict[str, Union[FloatValue, IntValue]]
UValue = List[float]
