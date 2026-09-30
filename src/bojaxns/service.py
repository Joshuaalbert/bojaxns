import logging

import numpy as np
import pylab as plt
from jax import random, numpy as jnp
from jaxns.types import PRNGKey
from jaxns.depth_condition import DepthCondition
from matplotlib import dates as mdates

from bojaxns.common import FloatValue, IntValue, ParamValues
from bojaxns.experiment import OptimisationExperiment, NewExperimentRequest, Trial, TrialUpdate
from bojaxns.gaussian_process_formulation.bayesian_optimiser import BayesianOptimiser
from bojaxns.gaussian_process_formulation.distribution_math import NotEnoughData
from bojaxns.parameter_space import build_parameter_model, ContinuousPrior, IntegerPrior, CategoricalPrior, sample_U_value
from bojaxns.utils import latin_hypercube

logger = logging.getLogger(__name__)

__all__ = [
    'InvalidTrial',
    'BayesianOptimisation'
]


class InvalidTrial(Exception):
    pass


class BayesianOptimisation:
    def __init__(self, experiment: OptimisationExperiment):
        experiment.validate()
        self._experiment = experiment

    @property
    def experiment(self):
        return self._experiment

    @classmethod
    def create_new_experiment(cls, new_experiment: NewExperimentRequest) -> 'BayesianOptimisation':
        new_experiment.validate()
        experiment = OptimisationExperiment(parameter_space=new_experiment.parameter_space)
        prior_model = build_parameter_model(experiment.parameter_space)
        U_dims = prior_model.U_ndims

        U_seeds = latin_hypercube(
            seed=42,
            num_samples=new_experiment.init_explore_size,
            num_dim=U_dims
        )

        # Add trials for seeds.
        for U_seed in U_seeds:
            trial = BayesianOptimisation._create_trial(
                experiment=experiment,
                U=U_seed,
                prior_model=prior_model
            )
            experiment.trials[trial.trial_id] = trial

        return cls(experiment=experiment)

    @staticmethod
    def _create_trial(experiment: OptimisationExperiment, U: jnp.ndarray, prior_model) -> Trial:
        prior_sample = prior_model.transform(U)
        param_values = {}
        for param in experiment.parameter_space.parameters:
            val = prior_sample[param.name]
            if isinstance(param.prior, ContinuousPrior):
                param_values[param.name] = FloatValue(value=float(val))
                continue
            if isinstance(param.prior, IntegerPrior):
                param_values[param.name] = IntValue(value=int(val))
                continue
            if isinstance(param.prior, CategoricalPrior):
                param_values[param.name] = IntValue(value=int(val))
                continue
        trial = Trial(param_values=param_values, U_value=U.tolist())
        return trial

    def add_trial_from_data(self, key: PRNGKey, param_values: ParamValues) -> str:
        self.experiment.validate()
        U = sample_U_value(key=key, param_space=self._experiment.parameter_space, param_values=param_values)
        trial = Trial(param_values=param_values, U_value=U)
        self._experiment.trials[trial.trial_id] = trial
        return trial.trial_id

    def create_new_trial(
            self, key: PRNGKey, random_explore: bool = False, beta: float = 0.5,
            *, num_search: int = 100000, batch_size: int = 10, S: int = 128,
            max_samples: int = 100000, root_allocation_degree: int | None = None,
            depth_condition: DepthCondition | None = None) -> str:
        """Return an unmeasured trial or propose a new point.

        Random exploration continues until at least two usable trial means
        differ. A flat objective supplies no scale for the empirical GP priors.
        Optional budgets control the existing JAXNS solver and random search.
        """
        self.experiment.validate()
        for trial in sorted(self._experiment.trials.values(), key=lambda t: t.create_dt):
            if not trial.trial_updates:
                return trial.trial_id

        prior_model = build_parameter_model(self.experiment.parameter_space)
        U = None
        if not random_explore and len(self.experiment.trials) >= 2:
            optimiser = BayesianOptimiser(
                experiment=self.experiment, beta=beta, S=S,
                max_samples=max_samples, root_allocation_degree=root_allocation_degree,
                depth_condition=depth_condition,
            )
            try:
                U = optimiser.choose_next_U_toptwo(
                    key=key, batch_size=batch_size, num_search=num_search,
                )
            except NotEnoughData:
                # GP validation sees the same penalised observations as the fit.
                # Recomputing a separate warm-up statistic can disagree with it.
                pass
        if U is None:
            U = random.uniform(key, shape=(prior_model.U_ndims,))
        trial = self._create_trial(self.experiment, U, prior_model)
        self.experiment.trials[trial.trial_id] = trial
        return trial.trial_id

    def get_trial(self, trial_id: str) -> Trial:
        if trial_id not in self._experiment.trials:
            raise InvalidTrial(trial_id)
        return self._experiment.trials[trial_id]

    def delete_trial(self, trial_id: str):
        if trial_id not in self._experiment.trials:
            raise InvalidTrial(trial_id)
        del self._experiment.trials[trial_id]

    def post_measurement(self, trial_id: str, trial_update: TrialUpdate):
        trial_update.validate()
        trial = self.get_trial(trial_id=trial_id)
        if (trial_update.ref_id in trial.trial_updates) and (
                trial_update.objective_measurement == trial.trial_updates[trial_update.ref_id].objective_measurement):
            return
        trial.trial_updates[trial_update.ref_id] = trial_update

    def trial_size(self, trial_id: str):
        trial = self.get_trial(trial_id=trial_id)
        return len(trial.trial_updates)

    def visualise(self,
                  main_color="#7e97bf",
                  grid_color="#969396",
                  ) -> plt.Figure:
        """
        Constructs a visual breakdown of condition

        Args:
            main_color: color of main axes
            grid_color: color of grid

        Returns:
            a pylab Figure

        Raises:
            NotEnoughData if not enough to compute a breakdown
        """
        # Plots scatter of trial outcomes over time.
        # Highlight, best.
        # For each trial use a colored line, with error bars that scale 1/sqrt(S)
        series = []
        for trial in self._experiment.trials.values():
            if len(trial.trial_updates) == 0:
                continue
            x, y, n = [], [], []
            for trial_update in sorted(trial.trial_updates.values(), key=lambda tu: tu.measurement_dt):
                x.append(trial_update.measurement_dt)
                if len(y) == 0:
                    y.append(trial_update.objective_measurement)
                    n.append(1)
                else:
                    y.append(y[-1] + trial_update.objective_measurement)
                    n.append(n[-1] + 1)
            y = list(map(lambda _y, _n: _y / _n, y, n))
            if len(y) < 3:
                y_std = [0.] * len(y)
            else:
                _mu = jnp.mean(jnp.asarray(y))
                _y_std = jnp.abs(y[0] - _mu)
                y_std = _y_std / jnp.sqrt(jnp.asarray(n))
                y_std = y_std.tolist()
            series.append((x, y, y_std))

        if len(series) == 0:
            raise RuntimeError("Nothing to visualise. Provide data to a trial first.")

        min_dt = min(min(x) for (x, y, y_std) in series)
        max_dt = max(max(x) for (x, y, y_std) in series)

        fig_width = 6
        fig_height = fig_width
        fig, ax = plt.subplots(figsize=(fig_height, fig_width), facecolor='None')
        ax.set_facecolor('none')
        colour_rng = np.random.default_rng(42)  # deterministic without global RNG mutation
        for x, y, y_std in series:
            color = [colour_rng.uniform(), colour_rng.uniform(), colour_rng.uniform(), 1.]
            ax.plot(x, y, c=color)
            color[-1] = 0.2
            ax.errorbar(x,
                        y,
                        y_std,
                        fmt='-o',
                        color=color
                        )

        agg_series = [(x[-1], y[-1], y_std[-1]) for (x, y, y_std) in series]
        agg_series = np.asarray(agg_series)

        idx_max = np.argmax(agg_series[:, 1])
        ax.scatter(agg_series[idx_max, 0], agg_series[idx_max, 1], s=100, fc='none', ec='black', marker='o',
                   label=f"Best {agg_series[idx_max, 1]}")

        # min_date_num = mdates.date2num(min_dt)
        # max_date_num = mdates.date2num(max_dt)
        #
        # interval = max((max_date_num - min_date_num) // 6, 1)
        #
        # ax.set_xticks(np.arange(min_date_num, max_date_num, interval))

        # xticks = []
        # ax.set_xticks(mdates.date2num(datetime.combine(today, time())))
        # restrict x lim
        ax.set_xlim(
            mdates.date2num(min_dt),
            mdates.date2num(max_dt)
        )
        # ax.set_ylim(0, 100)

        date_formatter = mdates.DateFormatter('%a, %-d %b')  # Customize the format as per your preference
        ax.xaxis.set_major_formatter(date_formatter)

        ax.set_title(f"Trial progression", color=main_color, fontsize=8)

        # Rotate value labels
        ax.tick_params(axis='x', rotation=45, labelsize=6)

        # Add legend
        ax.legend(loc='best', prop={'size': 6}, framealpha=0.1)
        # ax.legend(loc='upper center', prop={'size': 6}, framealpha=0.25, bbox_to_anchor=(0.5, -0.05))

        # Set xlim and ylim tight to data points

        ax.margins(0.01)

        # Invisible top, right
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set axis-colors
        ax.spines['left'].set_color(main_color)
        ax.spines['bottom'].set_color(main_color)
        ax.tick_params(axis='x', colors=main_color)
        ax.tick_params(axis='y', colors=main_color)

        # grid
        ax.grid(axis='y', linestyle='dashed', color=grid_color)

        # Make tight
        fig.tight_layout()

        return fig
