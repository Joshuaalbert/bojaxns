"""Explicit host-side schemas for persisted experiment state."""
from dataclasses import dataclass, field
from datetime import datetime
from numbers import Real
from typing import Dict
from uuid import uuid4

from bojaxns.basic import SerialisableBaseModel
from bojaxns.common import FloatValue, IntValue, ParamValues, UValue, finite_float, integer, parse_param_value
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


@dataclass
class TrialUpdate(SerialisableBaseModel):
    ref_id: str = field(metadata={'example': 'measurement-1'})
    objective_measurement: float = field(metadata={'example': 1.})
    measurement_dt: datetime = field(default_factory=current_utc)

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        _identifier(self.ref_id, 'ref_id')
        self.measurement_dt = _datetime(self.measurement_dt)
        if isinstance(self.objective_measurement, bool) or not isinstance(self.objective_measurement, Real):
            raise ValueError('objective_measurement must be numeric.')
        # The optimiser owns the established penalty for nonfinite objectives.
        self.objective_measurement = float(self.objective_measurement)


@dataclass
class Trial(SerialisableBaseModel):
    param_values: ParamValues = field(metadata={'example': {'price': {'type': 'float', 'value': 1.}}})
    U_value: UValue = field(metadata={'example': [0.2], 'items': {'type': 'number', 'minimum': 0, 'maximum': 1}})
    trial_id: str = field(default_factory=lambda: str(uuid4()))
    create_dt: datetime = field(default_factory=current_utc)
    trial_updates: Dict[str, TrialUpdate] = field(default_factory=dict)

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        _identifier(self.trial_id, 'trial_id')
        self.create_dt = _datetime(self.create_dt)
        self.param_values = {name: parse_param_value(value) for name, value in self.param_values.items()}
        for name, value in self.param_values.items():
            _identifier(name, 'parameter name')
            value.__post_init__()
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


@dataclass
class OptimisationExperiment(SerialisableBaseModel):
    parameter_space: ParameterSpace = field(metadata={'example': {'parameters': [
        {'name': 'price', 'prior': {'type': 'continuous_prior', 'lower': 0.1,
                                  'upper': 5.5, 'mode': 2.5, 'uncert': 2.}}]}})
    experiment_id: str = field(default_factory=lambda: str(uuid4()))
    trials: Dict[str, Trial] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.parameter_space, dict):
            self.parameter_space = ParameterSpace(**self.parameter_space)
        self.trials = {key: Trial(**value) if isinstance(value, dict) else value
                       for key, value in self.trials.items()}
        self.validate()

    def validate(self) -> None:
        """Recheck mutable state at a service boundary, without device work."""
        _identifier(self.experiment_id, 'experiment_id')
        if not isinstance(self.parameter_space, ParameterSpace):
            raise ValueError('Expected ParameterSpace.')
        self.parameter_space.__post_init__()
        parameters = self.parameter_space.parameters
        for parameter in parameters:
            parameter.__post_init__()
            parameter.prior.__post_init__()
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


@dataclass
class NewExperimentRequest(SerialisableBaseModel):
    parameter_space: ParameterSpace = field(metadata={'example': {'parameters': [
        {'name': 'price', 'prior': {'type': 'continuous_prior', 'lower': 0.1,
                                  'upper': 5.5, 'mode': 2.5, 'uncert': 2.}}]}})
    init_explore_size: int = field(metadata={'minimum': 1, 'example': 10})

    def __post_init__(self):
        if isinstance(self.parameter_space, dict):
            self.parameter_space = ParameterSpace(**self.parameter_space)
        if not isinstance(self.parameter_space, ParameterSpace):
            raise ValueError('Expected ParameterSpace.')
        self.init_explore_size = integer(self.init_explore_size, 'init_explore_size')
        if self.init_explore_size < 1:
            raise ValueError('init_explore_size must be positive.')
