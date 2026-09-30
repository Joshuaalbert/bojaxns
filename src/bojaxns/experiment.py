"""Explicit host-side schemas for persisted experiment state."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict
from uuid import uuid4

from jaxns.pytree import PureDataclassPytree

from bojaxns.common import FloatValue, IntValue, ParamValues, UValue, finite_float, integer, parse_param_value, real_scalar
from bojaxns.parameter_space import CategoricalPrior, ContinuousPrior, ParameterSpace
from bojaxns.utils import current_utc

__all__ = ['Trial', 'TrialUpdate', 'OptimisationExperiment', 'NewExperimentRequest']


def _datetime(value) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if not isinstance(value, datetime):
        raise ValueError('Expected a datetime or ISO datetime string.')
    return value


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f'{name} must be a nonempty string.')


@dataclass(slots=True)
class TrialUpdate(PureDataclassPytree):
    ref_id: str
    objective_measurement: float  # [] objective observation.
    measurement_dt: datetime = field(default_factory=current_utc)

    @classmethod
    def flatten(cls, this):
        return cls.build_flatten(this, ['ref_id', 'measurement_dt'])

    def validate(self) -> None:
        _identifier(self.ref_id, 'ref_id')
        self.measurement_dt = _datetime(self.measurement_dt)
        # The optimiser owns the established penalty for nonfinite objectives.
        self.objective_measurement = real_scalar(self.objective_measurement, 'objective_measurement')


@dataclass(slots=True)
class Trial(PureDataclassPytree):
    param_values: ParamValues
    U_value: UValue  # [D] flat unit coordinates in declaration order.
    trial_id: str = field(default_factory=lambda: str(uuid4()))
    create_dt: datetime = field(default_factory=current_utc)
    trial_updates: Dict[str, TrialUpdate] = field(default_factory=dict)

    @classmethod
    def flatten(cls, this):
        return cls.build_flatten(this, ['trial_id', 'create_dt'])

    def validate(self) -> None:
        _identifier(self.trial_id, 'trial_id')
        self.create_dt = _datetime(self.create_dt)
        self.param_values = {name: parse_param_value(value) for name, value in self.param_values.items()}
        for name, value in self.param_values.items():
            _identifier(name, 'parameter name')
            value.validate()
        self.U_value = [finite_float(u, 'U coordinate') for u in self.U_value]
        if any(not 0 <= u <= 1 for u in self.U_value):
            raise ValueError('U coordinates must lie in [0, 1].')
        self.trial_updates = {key: TrialUpdate(**value) if isinstance(value, dict) else value
                              for key, value in self.trial_updates.items()}
        for key, update in self.trial_updates.items():
            _identifier(key, 'measurement key')
            if not isinstance(update, TrialUpdate):
                raise ValueError('Expected TrialUpdate.')
            update.validate()


@dataclass(slots=True)
class OptimisationExperiment(PureDataclassPytree):
    parameter_space: ParameterSpace
    experiment_id: str = field(default_factory=lambda: str(uuid4()))
    trials: Dict[str, Trial] = field(default_factory=dict)

    @classmethod
    def flatten(cls, this):
        return cls.build_flatten(this, ['experiment_id'])

    def validate(self) -> None:
        if isinstance(self.parameter_space, dict):
            self.parameter_space = ParameterSpace(**self.parameter_space)
        self.trials = {key: Trial(**value) if isinstance(value, dict) else value
                       for key, value in self.trials.items()}
        _identifier(self.experiment_id, 'experiment_id')
        if not isinstance(self.parameter_space, ParameterSpace):
            raise ValueError('Expected ParameterSpace.')
        self.parameter_space.validate()
        parameters = self.parameter_space.parameters
        names = {p.name for p in parameters}
        dimension = sum(len(p.prior.probs) if isinstance(p.prior, CategoricalPrior) else 1 for p in parameters)
        for key, trial in self.trials.items():
            _identifier(key, 'trial key')
            if not isinstance(trial, Trial):
                raise ValueError('Expected Trial.')
            trial.validate()
            if set(trial.param_values) != names:
                raise ValueError(f"trial param_values {list(trial.param_values)} don't match param space {sorted(names)}.")
            if len(trial.U_value) != dimension:
                raise ValueError(f'Trial U_value must have {dimension} coordinates.')
            for parameter in parameters:
                prior, value = parameter.prior, trial.param_values[parameter.name]
                if isinstance(prior, ContinuousPrior):
                    valid = isinstance(value, FloatValue) and prior.lower <= value.value <= prior.upper
                elif isinstance(prior, CategoricalPrior):
                    valid = (isinstance(value, IntValue) and 0 <= value.value < len(prior.probs)
                             and prior.probs[value.value] > 0)
                else:
                    valid = isinstance(value, IntValue) and prior.lower <= value.value <= prior.upper
                if not valid:
                    raise ValueError(f'Trial value for {parameter.name} is outside its prior domain or has the wrong type.')


@dataclass(slots=True)
class NewExperimentRequest(PureDataclassPytree):
    parameter_space: ParameterSpace
    init_explore_size: int  # [] initial design size.

    def validate(self) -> None:
        if isinstance(self.parameter_space, dict):
            self.parameter_space = ParameterSpace(**self.parameter_space)
        if not isinstance(self.parameter_space, ParameterSpace):
            raise ValueError('Expected ParameterSpace.')
        self.parameter_space.validate()
        self.init_explore_size = integer(self.init_explore_size, 'init_explore_size')
        if self.init_explore_size < 1:
            raise ValueError('init_explore_size must be positive.')


TrialUpdate.register_pytree()
Trial.register_pytree()
OptimisationExperiment.register_pytree()
NewExperimentRequest.register_pytree()
