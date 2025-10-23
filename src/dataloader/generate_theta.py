# -*- coding: utf-8 -*-

# set path, othterwise import such as
# from src.module_name import X  won't work
if True:
    from path_setup import setup_sys_path
    setup_sys_path()

# module imports
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from src.utils.get_transformed_distr import get_transformed_beta_distr
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

# to add hyperparameters for the distributions
# @partial(jax.jit, static_argnums=0)


def generate_nig_marginal_params(mu_hyperparams,
                                 scale_hyperparams,
                                 beta_hyperparams,
                                 marginal_distr_name, key
                                 ):

    key, subkey_mu, subkey_scale, subkey_beta = jax.random.split(key, 4)

    if marginal_distr_name == 'beta':
        # sample the loc parameter mu
        jax_mu = get_transformed_beta_distr(
            mu_hyperparams).sample(seed=subkey_mu)

        # sample the std dev parameter scale
        jax_scale = get_transformed_beta_distr(
            scale_hyperparams).sample(seed=subkey_scale)

        # beta_sampler refers to a sampler for the beta (tilt) param
        # which also happes   to be a beta distribution
        jax_beta = get_transformed_beta_distr(
            beta_hyperparams).sample(seed=subkey_beta)

    elif marginal_distr_name == 'uniform':

        jax_mu = tfd.Uniform(
            low=mu_hyperparams[0], high=mu_hyperparams[1]).sample(seed=subkey_mu)
        jax_scale = tfd.Uniform(
            low=scale_hyperparams[0], high=scale_hyperparams[1]).sample(seed=subkey_scale)
        jax_beta = tfd.Uniform(
            low=beta_hyperparams[0], high=beta_hyperparams[1]).sample(seed=subkey_beta)

    else:
        raise ValueError

    jax_gamma = 1 + jnp.abs(jax_beta)/5
    jax_alpha = jnp.sqrt(jax_beta**2+jax_gamma**2)

    # convert to 4 param tf format
    tf_delta = jax_scale**2 * jax_gamma**3 / jax_alpha**2
    tf_mu = jax_mu - jax_beta * tf_delta / jax_gamma
    tf_alpha = jax_alpha
    tf_beta = jax_beta

    theta_jax = jnp.vstack([jax_mu, jax_scale, jax_beta])
    theta_tf = jnp.vstack([tf_mu, tf_delta, tf_alpha, tf_beta])

    return jnp.squeeze(theta_jax, axis=-1), jnp.squeeze(theta_tf, axis=-1), key


# @partial(jax.jit, static_argnames='distr_name')
def generate_sup_ig_acf_params_jax(gamma_hyperparams, eta_hyperparams, acf_distr_name, key):

    key, subkey_gamma, subkey_eta = jax.random.split(key, 3)

    if acf_distr_name == 'uniform':
        gamma = tfd.Uniform(
            low=gamma_hyperparams[0], high=gamma_hyperparams[1]).sample(seed=subkey_gamma)
        eta = tfd.Uniform(
            low=eta_hyperparams[0], high=eta_hyperparams[1]).sample(seed=subkey_eta)

    elif acf_distr_name == 'beta':
        raise ValueError('Not implemented yet')

    return jnp.array((gamma, eta)), key

# Exampls Usage
# f= jax.jit(jax.vmap(generate_sup_ig_acf_params_jax,in_axes=(0,0,0,None)),static_argnames = 'distr_name')


# @partial(jax.jit, static_argnums=0)
def generate_exp_param_jax(key):
    pass


if __name__ == "__main__":

    # Example usage: nig params

    batch_size = 10
    key = jax.random.PRNGKey(3)
    mu_hyperparams = (-1., 1., 1.25, 1.25)
    scale_hyperparams = (0.5, 1.5, 1.25, 1.25)
    beta_hyperparams = (-5., 5., 1.25, 1.25)
    marginal_distr_name = 'uniform'

    theta_marginal_jax, theta_marginal_tf, key = generate_nig_marginal_params(mu_hyperparams,
                                                                              scale_hyperparams,
                                                                              beta_hyperparams,
                                                                              marginal_distr_name,
                                                                              key
                                                                              )
    theta_marginal_tf.shape

    partial_marginal_generator = partial(generate_nig_marginal_params, mu_hyperparams, scale_hyperparams,                                                        beta_hyperparams,
                                         marginal_distr_name)

    vec_partial_marginal_generator = jax.jit(
        jax.vmap(partial_marginal_generator))
    vec_partial_marginal_generator(jax.random.split(key, 5))

    vec_marginal_generator_2 = jax.jit(
        jax.vmap(generate_nig_marginal_params, in_axes=(None, None, None, None, 0)), static_argnames=('marginal_distr_name',))

    vec_marginal_generator_2(mu_hyperparams,
                             scale_hyperparams,
                             beta_hyperparams,
                             marginal_distr_name,
                             jax.random.split(key, 5))
    vec_partial_marginal_generator_2 = partial(vec_marginal_generator_2, mu_hyperparams, scale_hyperparams,                                                        beta_hyperparams,
                                               marginal_distr_name)

    vec_partial_marginal_generator_2(jax.random.split(key, 5))

    # partial_marginal_generator = jax.jit(jax.vmap(generate_nig_marginal_params,in_axes=(None,None,None,None,0)))

    # Example usage: acf params
