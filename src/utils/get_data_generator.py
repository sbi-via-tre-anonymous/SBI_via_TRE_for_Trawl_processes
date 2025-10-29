# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:54:35 2024

@author: dleon
"""
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from functools import partial
from copy import deepcopy

if True:
    from path_setup import setup_sys_path
    setup_sys_path()

# data generation imports: fiirst marginal params, then acf params, then trawl generation
from src.dataloader.generate_theta import generate_nig_marginal_params
from src.dataloader.generate_theta import generate_sup_ig_acf_params_jax, generate_exp_param_jax
from src.dataloader.generate_sup_ig_nig_5params import slice_sample_sup_ig_nig_trawl


def get_theta_and_trawl_generator(config):

    trawl_config = config['trawl_config']
    batch_size = trawl_config['batch_size']

    if trawl_config['acf'] == 'sup_IG' and trawl_config['marginal_distr'] == 'NIG' and \
            trawl_config['trawl_process_type'] == 'sup_ig_nig_5p':

        # get simulators
        theta_acf_simulator = jax.jit(jax.vmap(generate_sup_ig_acf_params_jax,
                                               in_axes=(None, None, None, 0)),
                                      static_argnames='acf_distr_name')

        theta_marginal_simulator = jax.jit(jax.vmap(generate_nig_marginal_params,
                                                    in_axes=(None, None, None, None, 0)),
                                           static_argnames='marginal_distr_name')

        trawl_simulator = jax.jit(jax.vmap(slice_sample_sup_ig_nig_trawl,
                                           in_axes=(None, None, 0, 0, 0)),
                                  static_argnames=('nr_trawls', 'tau'))

        # get params and hyperparams
        trawl_config = config['trawl_config']
        nr_trawls = trawl_config['seq_len']
        tau = trawl_config['tau']

        # acf hyperparams
        acf_hyperparams = trawl_config['acf_prior_hyperparams']
        gamma_hyperparams = acf_hyperparams['gamma_prior_hyperparams']
        eta_hyperparams = acf_hyperparams['eta_prior_hyperparams']
        acf_distr_name = acf_hyperparams['distr_name']

        # marginal params
        marginal_hyperparams = trawl_config['marginal_distr_hyperparams']
        loc_prior_hyperparams = marginal_hyperparams['loc_prior_hyperparams']
        scale_prior_hyperparams = marginal_hyperparams['scale_prior_hyperparams']
        beta_prior_hyperparams = marginal_hyperparams['beta_prior_hyperparams']
        marginal_distr_name = marginal_hyperparams['distr_name']

        # seeds
        test_key = jax.random.split(
            PRNGKey(config['prng_key'])+140, 3 * batch_size)
        test_key_acf, test_key_marginal, test_key_trawl = test_key[:batch_size],\
            test_key[batch_size:2*batch_size], test_key[2*batch_size:]

        # run simulators: acf, marginal and then trawl
        theta_acf_test, _ = theta_acf_simulator(gamma_hyperparams,
                                                eta_hyperparams, acf_distr_name, test_key_acf)

        theta_marginal_jax_test, theta_marginal_tf_test, _ = theta_marginal_simulator(loc_prior_hyperparams,
                                                                                      scale_prior_hyperparams,
                                                                                      beta_prior_hyperparams,
                                                                                      marginal_distr_name,
                                                                                      test_key_marginal)

        trawl_test, _ = trawl_simulator(
            nr_trawls, tau, theta_acf_test, theta_marginal_tf_test, test_key_trawl)

        # once the simulatior functions have been compiled, we can apply the
        # partial function; not sure jax.jit(jax.vmap()) and partial are commutative operations
        # i think  first applying the partial and then jax.jit ( jax.vmap())
        # should work too

        theta_acf_simulator = partial(theta_acf_simulator,
                                      gamma_hyperparams, eta_hyperparams,
                                      acf_distr_name)

        theta_marginal_simulator = partial(theta_marginal_simulator,
                                           loc_prior_hyperparams,
                                           scale_prior_hyperparams,
                                           beta_prior_hyperparams,
                                           marginal_distr_name)

        trawl_simulator = partial(
            trawl_simulator, nr_trawls, tau)

        return theta_acf_simulator, theta_marginal_simulator, trawl_simulator

    else:
        raise ValueError('not yet implemented')


def get_variable_size_theta_and_trawl_generator(config):

    config = deepcopy(config)
    config['trawl_config']['seq_len'] = None

    trawl_config = config['trawl_config']
    batch_size = trawl_config['batch_size']

    if trawl_config['acf'] == 'sup_IG' and trawl_config['marginal_distr'] == 'NIG' and \
            trawl_config['trawl_process_type'] == 'sup_ig_nig_5p':

        # get simulators
        theta_acf_simulator = jax.jit(jax.vmap(generate_sup_ig_acf_params_jax,
                                               in_axes=(None, None, None, 0)),
                                      static_argnames='acf_distr_name')

        theta_marginal_simulator = jax.jit(jax.vmap(generate_nig_marginal_params,
                                                    in_axes=(None, None, None, None, 0)),
                                           static_argnames='marginal_distr_name')

        # trawl_simulator = jax.jit(jax.vmap(slice_sample_sup_ig_nig_trawl,
        #                                   in_axes=(None, None, 0, 0, 0)),
        #                          static_argnames=('nr_trawls', 'tau'))\

        # get params and hyperparams
        trawl_config = config['trawl_config']
        # nr_trawls = trawl_config['seq_len']
        tau = trawl_config['tau']

        # partially_applied_slice_sample = partial(
        #    slice_sample_sup_ig_nig_trawl, tau=tau)

        trawl_simulator = jax.jit(jax.vmap(slice_sample_sup_ig_nig_trawl,
                                           in_axes=(None, None, 0, 0, 0)),
                                  static_argnames=('nr_trawls', 'tau'))

        # acf hyperparams
        acf_hyperparams = trawl_config['acf_prior_hyperparams']
        gamma_hyperparams = acf_hyperparams['gamma_prior_hyperparams']
        eta_hyperparams = acf_hyperparams['eta_prior_hyperparams']
        acf_distr_name = acf_hyperparams['distr_name']

        # marginal params
        marginal_hyperparams = trawl_config['marginal_distr_hyperparams']
        loc_prior_hyperparams = marginal_hyperparams['loc_prior_hyperparams']
        scale_prior_hyperparams = marginal_hyperparams['scale_prior_hyperparams']
        beta_prior_hyperparams = marginal_hyperparams['beta_prior_hyperparams']
        marginal_distr_name = marginal_hyperparams['distr_name']

        # seeds
        test_key = jax.random.split(
            PRNGKey(config['prng_key'])+140, 3 * batch_size)
        test_key_acf, test_key_marginal, test_key_trawl = test_key[:batch_size],\
            test_key[batch_size:2*batch_size], test_key[2*batch_size:]

        # run simulators: acf, marginal and then trawl
        theta_acf_test, _ = theta_acf_simulator(gamma_hyperparams,
                                                eta_hyperparams, acf_distr_name, test_key_acf)

        theta_marginal_jax_test, theta_marginal_tf_test, _ = theta_marginal_simulator(loc_prior_hyperparams,
                                                                                      scale_prior_hyperparams,
                                                                                      beta_prior_hyperparams,
                                                                                      marginal_distr_name,
                                                                                      test_key_marginal)

        trawl_test, _ = trawl_simulator(
            1500, tau, theta_acf_test, theta_marginal_tf_test, test_key_trawl)

        # def partial_trawl_simulator(nr_trawls,  theta_acf_test, theta_marginak_test, test_key_trawl): return trawl_simulator(
        #    nr_trawls, 1.0, theta_acf_test, theta_marginal_tf_test, test_key_trawl)

        # once the simulatior functions have been compiled, we can apply the
        # partial function; not sure jax.jit(jax.vmap()) and partial are commutative operations
        # i think  first applying the partial and then jax.jit ( jax.vmap())
        # should work too

        theta_acf_simulator = partial(theta_acf_simulator,
                                      gamma_hyperparams, eta_hyperparams,
                                      acf_distr_name)

        theta_marginal_simulator = partial(theta_marginal_simulator,
                                           loc_prior_hyperparams,
                                           scale_prior_hyperparams,
                                           beta_prior_hyperparams,
                                           marginal_distr_name)

        # trawl_simulator = jax.vmap(partially_applied_slice_sample,
        #                          in_axes=(None, 0, 0, 0))

        # trawl_simulator = partial(
        #    trawl_simulator, nr_trawls, tau)

        # trawl_simulator = partial(
        #    trawl_simulator,  tau=tau)

        # partial_trawl_simulator
        return theta_acf_simulator, theta_marginal_simulator, trawl_simulator

    else:
        raise ValueError('not yet implemented')
