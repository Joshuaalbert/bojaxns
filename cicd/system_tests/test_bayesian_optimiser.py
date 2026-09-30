from dataclasses import dataclass, field
import os
from typing import List, Union
from uuid import uuid4

import pylab as plt
from jax import random, numpy as jnp

from bojaxns.experiment import NewExperimentRequest, Trial, TrialUpdate
from bojaxns.parameter_space import ParameterSpace, Parameter, ContinuousPrior, IntegerPrior, CategoricalPrior
from bojaxns.service import BayesianOptimisation


def test_bayesian_optimisation(tmp_path, monkeypatch):
    # Keep sampler diagnostics out of the checkout. Budgets tune real work,
    # never replace posterior sampling or acquisition evaluation with a fake.
    monkeypatch.chdir(tmp_path)
    num_steps = int(os.environ.get("BOJAXNS_SYSTEM_STEPS", "5"))
    assert num_steps >= 3
    sampler_options = {
        "num_search": int(os.environ.get("BOJAXNS_SYSTEM_NUM_SEARCH", "128")),
        "batch_size": int(os.environ.get("BOJAXNS_SYSTEM_BATCH_SIZE", "8")),
        "S": int(os.environ.get("BOJAXNS_SYSTEM_POSTERIOR_SAMPLES", "8")),
        "root_allocation_degree": int(os.environ.get("BOJAXNS_SYSTEM_ROOTS", "16")),
        "max_samples": int(os.environ.get("BOJAXNS_SYSTEM_MAX_SAMPLES", "4096")),
    }
    assert all(value > 0 for value in sampler_options.values())
    assert sampler_options["num_search"] % sampler_options["batch_size"] == 0

    def objective(x):
        return -0.5 * jnp.sum(x ** 4 - 16 * x ** 2 + 5 * x)

    def test_example(ndim):

        lower_bound = 39.16616 * ndim
        upper_bound = 39.16617 * ndim
        print(f"Optimal value in ({lower_bound}, {upper_bound}).")

        x_max = -2.903534

        print(f"Global optimum at {jnp.ones(ndim) * x_max}")

        parameter_space = ParameterSpace(
            parameters=[
                Parameter(
                    name=f'x{i}',
                    prior=ContinuousPrior(
                        lower=-5,
                        upper=5.,
                        mode=0.,
                        uncert=10.
                    )
                )
                for i in range(ndim)
            ]
        )
        new_experiment_request = NewExperimentRequest(
            parameter_space=parameter_space,
            init_explore_size=num_steps - 1
        )
        bo_experiment = BayesianOptimisation.create_new_experiment(new_experiment=new_experiment_request)

        seed_trial_ids = set(bo_experiment.experiment.trials)
        for i in range(num_steps):
            trial_id = bo_experiment.create_new_trial(
                key=random.PRNGKey(i),
                random_explore=False,
                **sampler_options
            )
            trial = bo_experiment.get_trial(trial_id=trial_id)
            params = []
            for param_name in sorted(trial.param_values.keys()):
                param = trial.param_values[param_name]
                params.append(param.value)
            params = jnp.asarray(params)
            assert params.shape == (ndim,)
            assert bool(jnp.all(jnp.isfinite(params)))
            assert bool(jnp.all((params >= -5) & (params <= 5)))
            if i == num_steps - 1:
                assert trial_id not in seed_trial_ids

            obj_val = float(objective(params))
            assert bool(jnp.isfinite(obj_val))
            bo_experiment.post_measurement(
                trial_id=trial_id,
                trial_update=TrialUpdate(ref_id='a', objective_measurement=obj_val)
            )
            assert bo_experiment.trial_size(trial_id) == 1
            # fig = bo_experiment.visualise()
            # plt.show()
            # plt.close('all')

        trial_id = bo_experiment.create_new_trial(
            key=random.PRNGKey(42),
            random_explore=True
        )
        _ = bo_experiment.get_trial(trial_id=trial_id)

        obj_val = float('nan')
        bo_experiment.post_measurement(
            trial_id=trial_id,
            trial_update=TrialUpdate(ref_id='illegal', objective_measurement=obj_val)
        )

    try:
        test_example(2)
    finally:
        plt.close('all')


def test_bayesian_optimiser():
    parameter_space = ParameterSpace(parameters=[
        Parameter(
            name='churn_rate',
            prior=ContinuousPrior(
                lower=1 / 20.,
                upper=1 / 10.,
                mode=1 / 15.,
                uncert=1 / 15.
            )
        ),
        Parameter(
            name='integers',
            prior=IntegerPrior(
                lower=0,
                upper=5,
                mode=4.,
                uncert=1.3
            )
        ),
        Parameter(
            name='categorical',
            prior=CategoricalPrior(
                probs=[1., 1., 1.]
            )
        )
    ])
    bo = BayesianOptimisation.create_new_experiment(
        new_experiment=NewExperimentRequest(parameter_space=parameter_space, init_explore_size=10)
    )

    for trial in bo.experiment.trials.values():
        assert (1 / 20 <= trial.param_values['churn_rate'].value <= 1 / 10)

    @dataclass
    class User:
        trial_id: str
        join_dt: float
        churn_dt: float
        user_id: str = field(default_factory=lambda: str(uuid4()))
        observable: Union[float, None] = None

    users: List[User] = []

    t = 0.
    T = 200.
    n_per_trial = 10
    trial: Union[Trial, None] = None
    new_user_rate = 2
    key = random.PRNGKey(42)
    while t < T:
        # Create a new trial
        if trial is None:
            key, sample_key = random.split(key)
            trial_id = bo.create_new_trial(key=sample_key, random_explore=True)
            trial = bo.get_trial(trial_id)
        # Update time to next user join
        key, sample_key = random.split(key)
        diff_dt = float(jnp.abs(random.laplace(sample_key)) / new_user_rate)
        t += diff_dt
        key, sample_key = random.split(key)
        churn_dt = t + float(jnp.abs(random.laplace(sample_key)) / trial.param_values['churn_rate'].value)
        users.append(User(trial_id=trial.trial_id, join_dt=t, churn_dt=churn_dt))
        # Handle trial
        trial_count = bo.trial_size(trial_id=trial.trial_id)
        if trial_count >= n_per_trial:
            trial = None
        # Report data
        for user in users:
            if t > user.churn_dt:
                user.observable = user.churn_dt - user.join_dt
            if user.observable is None:
                continue
            trial_update = TrialUpdate(ref_id=user.user_id,
                                       objective_measurement=user.observable)
            bo.post_measurement(trial_id=user.trial_id, trial_update=trial_update)
    assert any(bo.trial_size(trial_id) > 0 for trial_id in bo.experiment.trials)
    fig = bo.visualise()
    assert fig.axes
    plt.close('all')
