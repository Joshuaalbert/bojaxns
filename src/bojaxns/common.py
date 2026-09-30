"""Validated scalar values at the experiment's host-side boundary."""
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Dict, List, Literal, Union

from bojaxns.basic import SerialisableBaseModel


def finite_float(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f'{name} must be a finite number.')
    return float(value)


def integer(value: Integral, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer.')
    return int(value)


@dataclass
class FloatValue(SerialisableBaseModel):
    value: float
    type: Literal['float'] = 'float'

    def __post_init__(self):
        if self.type != 'float':
            raise ValueError('FloatValue type must be float.')
        self.value = finite_float(self.value, 'value')


@dataclass
class IntValue(SerialisableBaseModel):
    value: int
    type: Literal['int'] = 'int'

    def __post_init__(self):
        if self.type != 'int':
            raise ValueError('IntValue type must be int.')
        self.value = integer(self.value, 'value')


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
