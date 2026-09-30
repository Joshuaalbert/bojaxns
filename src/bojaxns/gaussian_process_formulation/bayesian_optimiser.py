import numpy as np
from chex import PRNGKey
from jax import random, numpy as jnp, vmap
from jax.lax import scan
from jaxns.model import Model
from jaxns.core import NestedSampler
from jaxns.depth_condition import DepthCondition
from jaxns.mixed_precision import mp_policy
from jaxns.results import NestedSamplerResults

from bojaxns.base import AbstractAcquisition, MarginalisedAcquisitionFunction, MarginalisationData
from bojaxns.experiment import OptimisationExperiment
from bojaxns.gaussian_process_formulation.distribution_math import GaussianProcessData, \
    GaussianProcessConditionalPredictiveFactory, ExpectedImprovementAcquisitionFactory, TopTwoAcquisitionFactory
from bojaxns.gaussian_process_formulation.multi_step_lookahead import run_multi_lookahead, convert_tree_to_graph

float_type = mp_policy.measure_dtype


class BayesianOptimiser:
    def __init__(self, experiment: OptimisationExperiment, num_parallel_solvers: int = 1, beta: float = 0.5,
                 S: int = 512, *, max_samples: int = 100000,
                 root_allocation_degree: int | None = None,
                 depth_condition: DepthCondition | None = None):
        experiment.validate()
        if not 0. <= beta <= 1.:
            raise ValueError("beta must be between zero and one.")
        if S < 1 or int(S) != S:
            raise ValueError("S must be a positive integer.")
        self._max_samples = max_samples
        self._root_allocation_degree = root_allocation_degree
        self._depth_condition = depth_condition
        self._experiment = experiment
        self._num_parallel_solvers = num_parallel_solvers
        self._beta = beta
        self._S = int(S)
        self._data = self._prepare_data()

    def _prepare_data(self) -> GaussianProcessData:
        U = []
        Y = []
        Y_var = []
        sample_size = []

        # handle nans ==> illegal value
        min_val, max_val = np.inf, -np.inf
        for trial_id, trial in self._experiment.trials.items():
            for ref_id, trial_update in trial.trial_updates.items():
                if not np.isfinite(trial_update.objective_measurement):
                    continue
                min_val = min(trial_update.objective_measurement, min_val)
                max_val = max(trial_update.objective_measurement, max_val)

        illegal_value = min_val - 0.1*(max_val - min_val)
        if not np.isfinite(illegal_value):
            illegal_value = 0.

        for trial_id, trial in self._experiment.trials.items():
            if len(trial.trial_updates) == 0:
                continue
            samples = []
            for ref_id, trial_update in trial.trial_updates.items():
                if not np.isfinite(trial_update.objective_measurement):
                    samples.append(illegal_value)
                else:
                    samples.append(trial_update.objective_measurement)
            U.append(trial.U_value)
            Y.append(np.mean(samples))
            if len(samples) < 2:
                Y_var.append(np.nan)
            else:
                Y_var.append(np.var(samples))
            sample_size.append(len(samples))
        U = jnp.asarray(U, float_type)
        Y = jnp.asarray(Y, float_type)
        Y_var = jnp.asarray(Y_var, float_type)
        sample_sizes = jnp.asarray(sample_size, float_type)
        data = GaussianProcessData(U=U, Y=Y, Y_var=Y_var, sample_size=sample_sizes)
        return data

    def posterior_solve(self, key: PRNGKey) -> NestedSamplerResults:
        """Infer GP hyperparameters with the v3 depth-controlled runner.

        The returned weights describe the classic posterior measure. A hard
        resource stop is an error, not a silently accepted posterior fit.
        Diagnostic plotting is available on the returned result explicitly.
        """
        factory = GaussianProcessConditionalPredictiveFactory(data=self._data)
        # The callable is stable; changing observations are dynamic run inputs.
        model = Model(prior_model=factory.build_prior_model())
        sampler = NestedSampler(
            model=model,
            max_samples=self._max_samples,
            root_allocation_degree=self._root_allocation_degree,
            depth_condition=self._depth_condition,
        )
        state = sampler.run(key=key, args=(self._data,))
        if int(state.termination_reason) != 0:
            raise RuntimeError(
                f"GP posterior solve stopped before its depth goal "
                f"(JAXNS termination reason {int(state.termination_reason)}). "
                "Increase max_samples or review the model and depth condition."
            )
        return state.to_result().trim()

    @staticmethod
    def _random_search(search_key: PRNGKey, acquisition_function: AbstractAcquisition,
                       ndims: int, batch_size: int, num_search: int,
                       *, return_history: bool = True):
        """Search exactly num_search candidates, optionally retaining diagnostics."""
        if any(type(value) is not int or value < 1
               for value in (ndims, batch_size, num_search)):
            raise ValueError("ndims, batch_size and num_search must be positive integers.")
        evaluate = vmap(acquisition_function)
        num_batches = (num_search + batch_size - 1) // batch_size

        def body(carry, inputs):
            best_value, best_U = carry
            batch_index, key = inputs
            candidates = random.uniform(key, (batch_size, ndims), dtype=float_type)
            values = evaluate(candidates)
            # Padding keeps the last device batch static without adding candidates.
            valid = batch_index * batch_size + jnp.arange(batch_size) < num_search
            scores = jnp.where(valid & jnp.isfinite(values), values, -jnp.inf)
            index = jnp.argmax(scores)
            better = scores[index] > best_value
            best = (jnp.where(better, scores[index], best_value),
                    jnp.where(better, candidates[index], best_U))
            history = (values, candidates) if return_history else None
            return best, history

        best, history = scan(
            body,
            (jnp.asarray(-jnp.inf, float_type), jnp.zeros(ndims, float_type)),
            (jnp.arange(num_batches), random.split(search_key, num_batches)),
        )
        if return_history:
            values, candidates = history
            history = (values.reshape(-1)[:num_search],
                       candidates.reshape(-1, ndims)[:num_search])
        return best, history

    @staticmethod
    def _multistep_lookahead_search(key: PRNGKey, data: GaussianProcessData, ns_results: MarginalisationData,
                                    batch_size: int, max_depth: int, num_simulations: int, branch_factor: int,
                                    S: int, *, tree_output: str | None = None):
        u_best, policy_output = run_multi_lookahead(
            rng_key=key,
            data=data,
            ns_results=ns_results,
            batch_size=batch_size,
            max_depth=max_depth,
            num_actions=branch_factor,
            num_simulations=num_simulations,
            S=S
        )
        if tree_output is not None:
            graph = convert_tree_to_graph(policy_output.search_tree)
            graph.draw(tree_output, prog="dot")
        return u_best

    def search_U_top1(self, key: PRNGKey, ns_results: MarginalisationData, batch_size: int, num_search: int,
                      *, return_history: bool = True):
        conditional_predictive_factory = GaussianProcessConditionalPredictiveFactory(data=self._data)
        acquisition_factory = ExpectedImprovementAcquisitionFactory(
            conditional_predictive_factory=conditional_predictive_factory
        )
        search_key, marginalise_key = random.split(key)
        marginalised_acquisition = MarginalisedAcquisitionFunction(
            key=marginalise_key,
            ns_results=ns_results,
            acquisition_factory=acquisition_factory,
            S=self._S
        )

        return BayesianOptimiser._random_search(
            search_key=search_key,
            acquisition_function=marginalised_acquisition,
            ndims=conditional_predictive_factory.ndims(),
            batch_size=batch_size,
            num_search=num_search,
            return_history=return_history
        )

    def search_U_top2(self, key: PRNGKey, ns_results: MarginalisationData, u1: jnp.ndarray, batch_size: int,
                      num_search: int, *, return_history: bool = True):
        conditional_predictive_factory = GaussianProcessConditionalPredictiveFactory(data=self._data)
        acquisition_factory = TopTwoAcquisitionFactory(
            conditional_predictive_factory=conditional_predictive_factory,
            u1=u1
        )
        search_key, marginalise_key = random.split(key)
        marginalised_acquisition = MarginalisedAcquisitionFunction(
            key=marginalise_key,
            ns_results=ns_results,
            acquisition_factory=acquisition_factory,
            S=self._S
        )

        return BayesianOptimiser._random_search(
            search_key=search_key,
            acquisition_function=marginalised_acquisition,
            ndims=conditional_predictive_factory.ndims(),
            batch_size=batch_size,
            num_search=num_search,
            return_history=return_history
        )

    def choose_next_U_toptwo(self, key: PRNGKey, batch_size: int, num_search: int):
        ns_key, search_top1_key, search_top2_key, do_top2_key = random.split(key, 4)

        do_top2 = random.uniform(do_top2_key) < self._beta

        ns_results = self.posterior_solve(key=ns_key)

        ns_results = MarginalisationData(
            samples=ns_results.X_samples,
            log_dp_mean=ns_results.log_dp
        )

        (best_value, next_u), _ = self.search_U_top1(
            key=search_top1_key, ns_results=ns_results, batch_size=batch_size,
            num_search=num_search, return_history=False,
        )
        if not np.isfinite(best_value):
            raise RuntimeError("No finite acquisition value was found.")
        if do_top2:
            (best_value, next_u), _ = self.search_U_top2(
                key=search_top2_key, ns_results=ns_results, u1=next_u,
                batch_size=batch_size, num_search=num_search, return_history=False,
            )
            if not np.isfinite(best_value):
                raise RuntimeError("No finite acquisition value was found.")

        return next_u

    def choose_next_U_multistep(self, key: PRNGKey, batch_size: int, max_depth: int, num_simulations: int,
                                branch_factor: int, *, tree_output: str | None = None):
        ns_key, search_key = random.split(key, 2)

        ns_results = self.posterior_solve(key=ns_key)
        ns_results = MarginalisationData(
            samples=ns_results.X_samples,
            log_dp_mean=ns_results.log_dp
        )

        next_u = BayesianOptimiser._multistep_lookahead_search(
            key=search_key,
            data=self._data,
            ns_results=ns_results,
            batch_size=batch_size,
            max_depth=max_depth,
            num_simulations=num_simulations,
            branch_factor=branch_factor,
            S=self._S,
            tree_output=tree_output
        )

        return next_u
