"""Small public experiment configuration for demos and composed tests."""

from bojaxns.experiment import NewExperimentRequest
from bojaxns.parameter_space import ContinuousPrior
from bojaxns.parameter_space import Parameter
from bojaxns.parameter_space import ParameterSpace


def example_request() -> NewExperimentRequest:
    """Create two initial trials for one bounded continuous parameter."""
    return NewExperimentRequest(
        parameter_space=ParameterSpace(parameters=[
            Parameter(
                name="x",
                prior=ContinuousPrior(lower=-1.0, upper=1.0, mode=0.0, uncert=1.0),
            ),
        ]),
        init_explore_size=2,
    )
