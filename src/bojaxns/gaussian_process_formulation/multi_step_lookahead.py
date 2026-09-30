from functools import partial
from typing import Optional, Sequence, Tuple, Callable, NamedTuple

import chex
import jax
import jax.numpy as jnp
import mctx
from jax.tree_util import tree_map
from mctx import PolicyOutput, GumbelMuZeroExtraData, RootFnOutput, RecurrentFn

from bojaxns.base import MarginalisedConditionalPredictive, MarginalisationData, MarginalisedAcquisitionFunction
from bojaxns.gaussian_process_formulation.distribution_math import GaussianProcessData, \
    GaussianProcessConditionalPredictiveFactory, ExpectedImprovementAcquisitionFactory


def convert_tree_to_graph(
        tree: mctx.Tree,
        action_labels: Optional[Sequence[str]] = None,
        batch_index: int = 0
):
    """Converts a search tree into a Graphviz graph.

    Args:
      tree: A `Tree` containing a batch of search data.
      action_labels: Optional labels for edges, defaults to the action index.
      batch_index: Index of the batch element to plot.

    Returns:
      A Graphviz graph representation of `tree`.
    """
    import pygraphviz

    chex.assert_rank(tree.node_values, 2)
    batch_size = tree.node_values.shape[0]
    if action_labels is None:
        action_labels = range(tree.num_actions)
    elif len(action_labels) != tree.num_actions:
        raise ValueError(
            f"action_labels {action_labels} has the wrong number of actions "
            f"({len(action_labels)}). "
            f"Expecting {tree.num_actions}.")

    def node_to_str(node_i, reward=0, discount=1):
        return (f"{node_i}\n"
                f"Reward: {reward:.2f}\n"
                f"Discount: {discount:.2f}\n"
                f"Value: {tree.node_values[batch_index, node_i]:.2f}\n"
                f"Visits: {tree.node_visits[batch_index, node_i]}\n")

    def edge_to_str(node_i, a_i):
        node_index = jnp.full([batch_size], node_i)
        probs = jax.nn.softmax(tree.children_prior_logits[batch_index, node_i])
        return (f"{action_labels[a_i]}\n"
                f"Q: {tree.qvalues(node_index)[batch_index, a_i]:.2f}\n"  # pytype: disable=unsupported-operands  # always-use-return-annotations
                f"p: {probs[a_i]:.2f}\n")

    graph = pygraphviz.AGraph(directed=True)

    # Add root
    graph.add_node(0, label=node_to_str(node_i=0), color="green")
    # Add all other nodes and connect them up.
    for node_i in range(tree.num_simulations):
        for a_i in range(tree.num_actions):
            # Index of children, or -1 if not expanded
            children_i = tree.children_index[batch_index, node_i, a_i]
            if children_i >= 0:
                graph.add_node(
                    children_i,
                    label=node_to_str(
                        node_i=children_i,
                        reward=tree.children_rewards[batch_index, node_i, a_i],
                        discount=tree.children_discounts[batch_index, node_i, a_i]),
                    color="red")
                graph.add_edge(node_i, children_i, label=edge_to_str(node_i, a_i))

    return graph


class MultiLookAheadState(NamedTuple):
    data: GaussianProcessData
    depth: chex.Array  # [...] scalar per state, with optional vmap batch axes


RewardFnType = Callable[[MultiLookAheadState], chex.Array]


@partial(jax.jit, static_argnames=['batch_size', 'max_depth', 'num_actions', 'num_simulations', 'S'])
def run_multi_lookahead(rng_key: chex.PRNGKey, data: GaussianProcessData, ns_results: MarginalisationData,
                        batch_size: int, max_depth: int, num_actions: int, num_simulations: int, S: int) -> \
        Tuple[chex.Array, PolicyOutput[GumbelMuZeroExtraData]]:
    recurrent_key, search_key, action_key = jax.random.split(rng_key, 3)

    U_ndims = data.U.shape[-1]
    U_options = jax.random.uniform(key=action_key, shape=(num_actions, U_ndims))

    root, recurrent_fn = _make_batched_env_model(
        # Using batch_size=2 to test the batched search.
        key=recurrent_key,
        batch_size=batch_size,
        max_depth=max_depth,
        num_actions=num_actions,
        S=S,
        data=data,
        ns_results=ns_results,
        U_options=U_options,
    )

    # Running the search.
    policy_output = mctx.gumbel_muzero_policy(
        params=(),
        rng_key=search_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        max_depth=max_depth,
        max_num_considered_actions=num_actions,
    )

    @jax.vmap
    def _extract_value(batch_index):
        selected_action = policy_output.action[batch_index]
        q_value = policy_output.search_tree.summary().qvalues[
            batch_index, selected_action]
        return selected_action, q_value

    selected_actions, q_values = _extract_value(jnp.arange(batch_size, dtype=jnp.int32))
    best_idx = jnp.argmax(q_values)
    best_action = selected_actions[best_idx]
    u_best = U_options[best_action]

    return u_best, policy_output


def _make_batched_env_model(
        key: chex.PRNGKey,
        batch_size: int,
        num_actions: int,
        S: int,
        max_depth: int,
        data: GaussianProcessData,
        ns_results: MarginalisationData,
        U_options: chex.Array) -> Tuple[RootFnOutput, RecurrentFn]:
    """Returns a batched `(root, recurrent_fn)`."""

    if max_depth < 1:
        raise ValueError('max_depth must be at least one.')

    reward_key, marginalise_key = jax.random.split(key, 2)

    def _duplicate_batch(x):
        return jnp.repeat(x[None, ...], batch_size, axis=0)

    def _extend(x, value, extra_len):
        extra = jnp.full((extra_len,) + x.shape[1:], value)
        return jnp.concatenate([x, extra], axis=0)

    init_data_size = data.Y.size
    # Need up to depth-1
    init_data = GaussianProcessData(
        U=_extend(data.U, 0., max_depth - 1),
        Y=_extend(data.Y, 0., max_depth - 1),
        Y_var=_extend(data.Y_var, jnp.inf, max_depth - 1),
        sample_size=_extend(data.sample_size, jnp.inf, max_depth - 1),
    )

    init_state = MultiLookAheadState(
        data=init_data,
        depth=jnp.asarray(0)
    )
    init_state = tree_map(_duplicate_batch, init_state)

    prior_logits = jnp.full([batch_size, num_actions], 0.)
    # The approximate expected value of sum_i discount^i * R_i
    # Zero if more expected improvement.
    approximate_value = jnp.full([batch_size], 0.)

    root = mctx.RootFnOutput(
        prior_logits=prior_logits,
        value=approximate_value,
        embedding=init_state
    )

    def recurrent_fn(params, rng_key, action: chex.Array, embedding: MultiLookAheadState) -> Tuple[
        mctx.RecurrentFnOutput, MultiLookAheadState]:
        del params, rng_key
        chex.assert_shape(action, [batch_size])
        chex.assert_shape(embedding.depth, [batch_size])

        def _single_batch(_action: chex.Array, _state: MultiLookAheadState) -> Tuple[
            mctx.RecurrentFnOutput, MultiLookAheadState]:
            conditional_predictive_factory = GaussianProcessConditionalPredictiveFactory(data=_state.data)
            conditional_predictive = MarginalisedConditionalPredictive(
                key=marginalise_key,
                ns_results=ns_results,
                conditional_predictive_factory=conditional_predictive_factory,
                S=S
            )
            acquisition_factory = ExpectedImprovementAcquisitionFactory(
                conditional_predictive_factory=conditional_predictive_factory
            )
            acquisition = MarginalisedAcquisitionFunction(
                key=reward_key,
                ns_results=ns_results,
                acquisition_factory=acquisition_factory,
                S=S
            )
            # Compute EI(f_i, f_i-1)
            u_star = U_options[_action]
            reward = jnp.where(_state.depth < max_depth, acquisition(u_star), 0.)
            post_mu_s, post_var_s = conditional_predictive(U_star=u_star[None, :])  # should be optimised away
            next_Y = jnp.reshape(post_mu_s, ())
            next_Y_var = jnp.reshape(post_var_s, ())
            def append_fantasy(current_data):
                # vmap may evaluate both cond branches, so even a discarded
                # terminal update must use an in-bounds scatter index.
                replace_idx = jnp.minimum(init_data_size + _state.depth, current_data.Y.size - 1)
                # Fantasy variance is the latent mixture variance. Infinite
                # sample_size suppresses the extra observation-noise term; this
                # retains the existing deterministic-mean planning approximation.
                return GaussianProcessData(
                    U=current_data.U.at[replace_idx, :].set(u_star),
                    Y=current_data.Y.at[replace_idx].set(next_Y),
                    Y_var=current_data.Y_var.at[replace_idx].set(next_Y_var),
                    sample_size=current_data.sample_size,
                )

            # Only nonterminal transitions need a stored fantasy. There are
            # max_depth-1 slots; the final reward needs no subsequent GP update.
            next_data = jax.lax.cond(_state.depth < max_depth - 1, append_fantasy,
                                     lambda current_data: current_data, _state.data)
            next_depth = jnp.minimum(_state.depth + 1, max_depth)
            terminal = next_depth >= max_depth

            next_state = MultiLookAheadState(
                data=next_data,
                depth=next_depth
            )

            discount = jnp.where(terminal, 0., 1.)
            prior_logits = jnp.full([num_actions], 0.)
            approximate_value = jnp.where(terminal, 0., reward)

            recurrent_fn_output = mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=prior_logits,
                value=approximate_value
            )

            return recurrent_fn_output, next_state

        return jax.vmap(_single_batch)(action, embedding)

    return root, recurrent_fn


def static_fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val
