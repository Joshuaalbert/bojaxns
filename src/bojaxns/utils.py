"""Host-side design sampling and time utilities."""
from datetime import datetime, timezone, tzinfo
from typing import Union

import numpy as np
from scipy.stats import qmc

from bojaxns.basic import build_example, example_from_schema as example_from_schema

__all__ = ['latin_hypercube', 'build_example', 'current_utc']


def latin_hypercube(seed: int, num_samples: int, num_dim: int) -> np.ndarray:
    """Return [num_samples, num_dim] stratified samples using a local RNG."""
    if (isinstance(num_samples, bool) or not isinstance(num_samples, (int, np.integer))
            or num_samples < 1):
        raise ValueError('num_samples must be a positive integer.')
    if isinstance(num_dim, bool) or not isinstance(num_dim, (int, np.integer)) or num_dim < 1:
        raise ValueError('num_dim must be a positive integer.')
    return qmc.LatinHypercube(d=num_dim, seed=seed).random(n=num_samples)


def set_datetime_timezone(dt: datetime, offset: Union[str, tzinfo]) -> datetime:
    """Replace a datetime's timezone with a tzinfo or ISO offset."""
    if isinstance(offset, str):
        return datetime.fromisoformat(f'{dt.replace(tzinfo=None).isoformat()}{offset}')
    if isinstance(offset, tzinfo):
        return dt.replace(tzinfo=offset)
    raise ValueError(f'offset {offset} not understood.')


def set_utc_timezone(dt: datetime) -> datetime:
    return set_datetime_timezone(dt, '+00:00')


def current_utc() -> datetime:
    return datetime.now(timezone.utc)
