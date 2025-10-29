from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.classifier_utils import get_projection_function
import numpy as np
import datetime
import pickle
import optax
import wandb
import yaml
import os
import time


import jax
import jax.numpy as jnp
from jax.random import PRNGKey

import tensorflow_probability.substrates.jax as tfp
import corner

if True:
    from path_setup import setup_sys_path
    setup_sys_path()
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt


import numpyro
from numpyro.infer import MCMC, NUTS, HMC, BarkerMH
import numpyro.distributions as dist
from numpyro.diagnostics import effective_sample_size as ess
import arviz as az

# mcmc_functions.py - Core functions
# from jax import debug


def run_mcmc_for_trawl(trawl_idx, approximate_log_likelihood_to_evidence_just_theta,
                       true_thetas, seed, num_samples=25000,
                       num_warmup=10000, num_burnin=10000,
                       num_chains=5):
    """
    Run MCMC for a specific trawl index

    Parameters:
    -----------
    trawl_idx : int
        Index of the trawl to process
    true_trawls : array
        Array of all trawls
    true_thetas : array
        Array of all thetas
    model_config : dict
        Configuration parameters

    Returns:
    --------
    dict
        Results dictionary with samples, diagnostics, etc.
    """
    test_theta = true_thetas[trawl_idx, :]



    # Use in model function
    def model_vec():
        gamma = numpyro.sample("gamma", dist.Uniform(10, 20))
        eta = numpyro.sample("eta", dist.Uniform(10, 20))
        mu = numpyro.sample("mu", dist.Uniform(-1, 1))
        sigma = numpyro.sample("sigma", dist.Uniform(0.5, 1.5))
        beta = numpyro.sample("beta", dist.Uniform(-5, 5))

        theta_vec = jnp.array([gamma, eta, mu, sigma, beta])

        # Use the pre-compiled function
        log_likelihood = approximate_log_likelihood_to_evidence_just_theta(
            theta_vec)

        numpyro.deterministic("log_likelihood", log_likelihood)
        numpyro.factor("likelihood_factor", log_likelihood)

    rng_key = jax.random.PRNGKey(trawl_idx)
    chain_keys = jax.random.split(rng_key, num_chains)

    # We need to run for (num_samples + num_burnin) steps after warmup
    total_post_warmup = num_samples + num_burnin

    # hmc_kernel = HMC(model_vec, step_size=0.075, find_heuristic_step_size=False,
    #                 adapt_step_size=True, adapt_mass_matrix=True, dense_mass=True, num_steps=5)
    nuts_kernel = NUTS(model_vec, step_size=0.075, adapt_step_size=True,
                       adapt_mass_matrix=True, dense_mass=True,  # num_steps = 5)
                       max_tree_depth=10)  # Adaptive HMC

    mcmc = MCMC(nuts_kernel,  # hmc_kernel,
                num_warmup=num_warmup, num_samples=total_post_warmup,
                num_chains=num_chains, chain_method='sequential', progress_bar=False)  # True defeats the jit speedups

    start_time = time.time()
    mcmc.run(chain_keys)
    end_time = time.time()

    # Get all samples including burn-in
    all_samples = mcmc.get_samples(group_by_chain=True)

    # Discard burn-in samples for each chain
    posterior_samples = {}
    for param, chain_samples in all_samples.items():
        # chain_samples shape: (num_chains, total_post_warmup)
        # Keep only the samples after burn-in
        if param != 'log_likelihood':
            posterior_samples[param] = chain_samples[:, num_burnin:]

    # Convert to arviz format for diagnostics
    az_data = az.convert_to_dataset(posterior_samples)

    log_liked_at_true_params = approximate_log_likelihood_to_evidence_just_theta(
        test_theta[jnp.newaxis, :])[0].item()

    log_likelihood_at_samples = all_samples["log_likelihood"][:, num_burnin:]

    # Gather results
    results = {
        # 'posterior_samples': posterior_samples,
        'runtime': end_time - start_time,
        # az.ess(az.from_numpyro(mcmc), method="bulk"),
        'ess_bulk': az.ess(posterior_samples, method="bulk"),
        # az.ess(az.from_numpyro(mcmc), method="tail"),
        'ess_tail': az.ess(posterior_samples, method="tail"),
        'rhat': az.rhat(posterior_samples),  # az.rhat(az.from_numpyro(mcmc)),
        'log_likelihood_samples': log_likelihood_at_samples,
        'true_log_like': log_liked_at_true_params,
        'coverage': np.sum(log_liked_at_true_params < log_likelihood_at_samples) / np.prod(log_likelihood_at_samples.shape)
        # 'covariance_matrix': mcmc.last_state.adapt_state.inverse_mass_matrix,
        # 'true_theta': test_theta,
    }
    return results, posterior_samples

# Save/load helpers


def save_results(results, filename):
    with open(filename, 'wb') as f:
        pickle.dump(results, f)


def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def create_and_save_plots(posterior_samples, true_theta, save_dir):
    """Create and save diagnostic plots from MCMC results"""
    # Convert to arviz format
    az_data = az.convert_to_dataset(posterior_samples)

    # Create pair plot
    grid = az.plot_pair(
        az_data,
        var_names=["gamma", "eta",  "mu", "sigma", "beta"],
        marginals=True,
        kind='kde',
        figsize=(12, 12),
        reference_values={"gamma": true_theta[0],
                          "eta": true_theta[1],
                          "mu": true_theta[2],
                          "sigma": true_theta[3],
                          "beta": true_theta[4]},
        reference_values_kwargs={"color": "r", "marker": "o"}
        # point_estimate='mode'
    )

    plt.tight_layout()

    # Loop through axes to format
    for ax in grid.flat:
        if ax is not None:
            # Rotate x-axis labels
            ax.tick_params(axis='x', labelrotation=45)

            # Format tick labels to use fewer decimal places
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

            # Increase font size
            ax.tick_params(axis='both', labelsize=10)


    # Get the figure from one of the axes
    fig = plt.gcf()  # Get current figure
    plt.savefig(os.path.join(save_dir, "pair_plot.png"),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Create trace plot
    fig = plt.figure(figsize=(12, 8))
    az.plot_trace(az_data)
    plt.savefig(os.path.join(save_dir, "trace_plot.png"),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')


if __name__ == '__main__':
    pass
    # test_index = -10
    # test_trawl = true_trawls[test_index, :]
    # test_theta = true_thetas[test_index, :]
    #
    # def model_vec():
    #    eta = numpyro.sample("eta", dist.Uniform(10, 20))
    #    gamma = numpyro.sample("gamma", dist.Uniform(10, 20))
    #    mu = numpyro.sample("mu", dist.Uniform(-1, 1))
    #    sigma = numpyro.sample("sigma", dist.Uniform(0.5, 1.5))
    #    beta = numpyro.sample("beta", dist.Uniform(-5, 5))
    #
    #    params = jnp.array([eta, gamma, mu, sigma, beta])[jnp.newaxis, :]
    #    batch_size = params.shape[0]  # Should be `num_chains`
    #    x_tiled = jnp.tile(test_trawl, (batch_size, 1))
    #    numpyro.factor("likelihood", jnp.squeeze(approximate_log_likelihood_to_evidence(x_tiled,
    #                                                                                    params)))  # Include log-likelihood in inference

    # posterior_samples = mcmc.get_samples(group_by_chain=True)
    # az_data = az.from_numpyro(mcmc)
    # az.plot_trace(az_data)
    # ess = az.ess(az_data)
    # print(ess)
    # az.plot_pair(
    #    az_data,
    #    var_names=["eta", "gamma", "mu", "sigma", "beta"],
    #    marginals=True,
    #    kind='kde',
    #    figsize=(10, 10),
    #    reference_values={"eta": test_theta[0],
    #                      "gamma": test_theta[1],
    #                      "mu": test_theta[2],
    #                      "sigma": test_theta[3],
    #                      "beta": test_theta[4]},  # Add true values
    #    reference_values_kwargs={"color": "r", "marker": "o"}
    # )
    # plt.savefig('pair_plot.png', dpi=300, bbox_inches='tight')

    # ess_bulk = az.ess(az_data, method="bulk")
    # ess_tail = az.ess(az_data, method="tail")
    # print("Bulk ESS:", ess_bulk)
    # print("Tail ESS:", ess_tail)
    # rhat = az.rhat(az_data)
    # print("Rhat values:", rhat)
    # az.plot_autocorr(az_data)
    # plt.savefig('autocorr_plot.png')
    # mcmc.last_state.adapt_state #to get the adapted covariance matrix
