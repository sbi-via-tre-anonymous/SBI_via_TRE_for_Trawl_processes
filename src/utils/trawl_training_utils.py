# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from functools import partial
from src.utils.acf_functions import get_acf
from src.utils.KL_divergence import vec_monte_carlo_kl_3_param_nig


if True:
    from path_setup import setup_sys_path
    setup_sys_path()


def apply_transformation_to_trawl():
    pass


def loss_functions_wrapper(state, config):

    learn_config = config['learn_config']
    learn_acf = learn_config['learn_acf']
    learn_marginal = learn_config['learn_marginal']

    ####
    trawl_config = config['trawl_config']
    batch_size = trawl_config['batch_size']

    ####
    loss_config = config['loss_config']
    p = loss_config['p']

    # acf hyperparams
    trawl_func_name = trawl_config['acf']
    use_acf_directly = loss_config['use_acf_directly']
    nr_acf_lags = loss_config['nr_acf_lags']
    acf_func = jax.jit(
        jax.vmap(get_acf(trawl_config['acf']), in_axes=(None, 0)))

    # marginal hyperparams
    use_kl_div = loss_config['use_kl_div']
    if learn_marginal:
        kl_type = loss_config['kl_type']
    marginal_distr = trawl_config['marginal_distr']
    trawl_process_type = trawl_config['trawl_process_type']
    assert isinstance(use_kl_div, bool)

    ###########################################################################
    #  Helper functions to predict theta, for when we predict on log scale    #
    ###########################################################################

    def get_pred_theta_acf_from_nn_output(pred_theta, trawl_func_name):

        if trawl_func_name in ('sup_IG', 'exponential'):

            return jnp.exp(pred_theta)  # work on the log scale

        else:
            raise ValueError

    def get_pred_theta_marginal_from_nn_output(pred_theta, marginal_distr, trawl_process_type):

        if marginal_distr == 'NIG' and trawl_process_type == 'sup_ig_nig_5p':

            return pred_theta.at[:, 1].set(jnp.exp(pred_theta[:, 1]))
            # learn sigma on the log scale
        else:

            raise ValueError('not yet implemented')

    ###########################################################################
    #                          Neural network output                          #
    ###########################################################################

    @partial(jax.jit, static_argnames=('train',))
    def predict_theta(params, trawl, dropout_rng, train):

        if learn_acf:

            trawl = (trawl - jnp.mean(trawl, axis=1, keepdims=True)) / \
                jnp.std(trawl, axis=1, keepdims=True)

        # can add another elif learn_marginal and do the transformation

        if train:

            nn_output = state.apply_fn(
                params,
                trawl,
                train=True,
                rngs={'dropout': dropout_rng}
            )

        else:

            nn_output = state.apply_fn(
                params,
                trawl,
                train=False
            )

        if learn_acf:
            pred_theta = get_pred_theta_acf_from_nn_output(
                nn_output, trawl_func_name)

        else:
            pred_theta = get_pred_theta_marginal_from_nn_output(
                nn_output, marginal_distr, trawl_process_type)

        return pred_theta

    ###########################################################################
    #                      Loss function helpers                              #
    ###########################################################################

    @jax.jit
    def _acf_loss(true_theta, pred_theta):
        """Compute ACF-based loss."""
        H = jnp.arange(1, nr_acf_lags + 1)
        pred_acf = acf_func(H, pred_theta)
        theoretical_acf = acf_func(H, true_theta)
        # return jnp.mean(jnp.abs((pred_acf - theoretical_acf))**p)**(1 / p)
        l_p_norms = jnp.mean(
            jnp.abs((pred_acf - theoretical_acf))**p, axis=1)**(1/p)
        return jnp.mean(l_p_norms)

    @jax.jit
    def _direct_params_loss(true_theta, pred_theta):
        """Compute direct parameter-based loss."""
        l_p_norms = jnp.mean(
            jnp.abs(true_theta - pred_theta)**p, axis=1)**(1/p)
        return jnp.mean(l_p_norms)

    ###########################################################################
    # Allow for different loss functions for learning acf and marginal params #
    ###########################################################################

    if learn_acf or (learn_marginal and (not use_kl_div)):

        # subcase: when learning acf params, can either compare acfs or params

        @partial(jax.jit, static_argnames=('train'))
        def compute_loss(params, trawl, theta_acf, dropout_rng, train, num_KL_samples):
            """num_KL_samples is not used here. it's just added for simplicity, so
            that i don't have to rewrite the validation function"""

            pred_theta = predict_theta(params, trawl, dropout_rng, train)

            if learn_acf and use_acf_directly:

                return _acf_loss(theta_acf, pred_theta)

            else:

                return _direct_params_loss(theta_acf, pred_theta)

        compute_loss_and_grad = jax.jit(jax.value_and_grad(
            compute_loss), static_argnames=('train',))

    elif learn_marginal and use_kl_div:

        @partial(jax.jit, static_argnames=('train', 'num_KL_samples', 'compute_grads'))
        def _compute_loss(params, trawl, theta_marginal, dropout_rng, train, num_KL_samples, compute_grads):
            """
            Compute KL divergence and its gradient using Jacobian forward-mode differentiation.

            The key insight is using jax.jacfwd to compute the full Jacobian of the KL objective 
            with respect to the predicted parameters.
            """
            # Step 1: Compute pred_theta with potential gradient tracking
            def compute_pred_theta(params):
                pred_theta = predict_theta(params, trawl, dropout_rng, train)
                return pred_theta

            if compute_grads:
                # Use reverse-mode differentiation to compute pred_theta and vjp_fn
                pred_theta, vjp_fn = jax.vjp(compute_pred_theta, params)
            else:
                # No gradients required, just compute pred_theta
                pred_theta = compute_pred_theta(params)

            # Step 2: KL divergence objective function
            if marginal_distr == 'NIG' and trawl_process_type == 'sup_ig_nig_5p':

                split_dropout_rng = jax.random.split(dropout_rng, batch_size)

                def kl_objective(theta_pred):
                    # Compute KL divergence samples
                    kl_samples, _ = vec_monte_carlo_kl_3_param_nig(
                        theta_marginal,
                        theta_pred,
                        split_dropout_rng,
                        num_KL_samples
                    )
                    # Return mean KL divergence
                    return jnp.mean(kl_samples)

                def rev_kl(theta_pred):
                    # Compute KL divergence samples
                    kl_samples, _ = vec_monte_carlo_kl_3_param_nig(
                        theta_pred,
                        theta_marginal,
                        jax.random.split(split_dropout_rng[-1], batch_size),
                        num_KL_samples
                    )
                    # Return mean KL divergence
                    return jnp.mean(kl_samples)
            else:
                raise ValueError

            # Compute KL divergence loss
            if kl_type == 'kl':
                kl_loss = kl_objective(pred_theta)

                if compute_grads:
                    kl_fwd_jac = jax.jacfwd(kl_objective)(pred_theta)

            elif kl_type == 'rev':
                kl_loss = rev_kl(pred_theta)

                if compute_grads:
                    kl_fwd_jac = jax.jacfwd(rev_kl)(pred_theta)

            elif kl_type == 'sym':
                kl_loss = (kl_objective(pred_theta) + rev_kl(pred_theta))/2

                if compute_grads:
                    kl_fwd_jac = (jax.jacfwd(kl_objective)(
                        pred_theta) + jax.jacfwd(rev_kl)(pred_theta)) / 2

            else:
                raise ValueError

            # If no gradients are needed, return only the KL divergence loss
            if not compute_grads:
                return kl_loss

            # Step 3: Compute full Jacobian using jax.jacfwd
            # This computes the gradient of the KL objective with respect to pred_theta

            # Step 4: Backpropagate Jacobian through vjp_fn
            combined_grad = vjp_fn(kl_fwd_jac)[0]

            return kl_loss, combined_grad

        compute_loss = partial(_compute_loss, compute_grads=False)
        compute_loss_and_grad = partial(_compute_loss, compute_grads=True)

    ###########################################################################
    #                    Validation function                                  #
    ###########################################################################

    @partial(jax.jit, static_argnames=('num_KL_samples'))
    def compute_validation_stats(params, val_trawls, val_thetas_marginal, dropout_rng, num_KL_samples):

        def body_fun(i, carry):

            dropout_rng, acc = carry
            # Split RNG for this iteration
            dropout_rng, subkey = jax.random.split(dropout_rng)

            theta_val = jax.lax.dynamic_slice_in_dim(
                val_thetas_marginal, i, 1)[0]
            trawl_val = jax.lax.dynamic_slice_in_dim(
                val_trawls, i, 1)[0]

            loss = compute_loss(params, trawl_val, theta_val,
                                subkey, False, num_KL_samples)

            return (dropout_rng, acc + jnp.array([loss, loss**2]))

        # Initialize carry with both RNG and accumulator
        init_carry = (dropout_rng, jnp.zeros(2))

        # Run the loop, keeping track of both RNG and accumulated stats
        _, total = jax.lax.fori_loop(
            0, val_trawls.shape[0], body_fun, init_carry
        )

        n = val_trawls.shape[0]
        mean = total[0] / n
        variance = (total[1] / n) - (mean**2)
        std = jnp.sqrt(jnp.maximum(variance, 0.0))
        return mean, std, dropout_rng

    return predict_theta, compute_loss, compute_loss_and_grad, compute_validation_stats

  