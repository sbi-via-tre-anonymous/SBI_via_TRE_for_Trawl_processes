# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
from statsmodels.tsa.stattools import acf as compute_empirical_acf
import matplotlib.pyplot as plt
import numpy as np


@jax.jit
def corr_exponential_envelope(h, params):
    u = params[0]
    return jnp.exp(-u * h)


@jax.jit
def corr_gamma_envelope(h, params):
    H, delta = params
    return (1+h/delta)**(-H)


@jax.jit
def corr_sup_ig_envelope(h, params):
    gamma, eta = params
    return jnp.exp(eta * (1-jnp.sqrt(2*h/gamma**2+1)))

    # previous implementation had gamma, delta
    # where gamma is the same and eta = delta * gamma, delta = eta / gamma
    # gamma, delta = params
    # return jnp.exp(delta * gamma * (1-jnp.sqrt(2*h/gamma**2+1)))


def get_acf(acf_type):

    if acf_type == 'exponential':

        return corr_exponential_envelope

    elif acf_type == 'gamma':

        return corr_gamma_envelope

    elif acf_type == 'sup_IG':

        return corr_sup_ig_envelope

    else:

        raise ValueError(f'acf_type {acf_type} not implemented yet')


def _old_plot_theoretical_empirical_inferred_acf(trawl, theoretical_theta, inferred_theta, trawl_type, nlags):
    """ Plots the empirical, theoretical, infered and GMM acf functions"""

    empirical_acf = compute_empirical_acf(np.array(trawl), nlags=nlags)[1:]

    acf_func = get_acf(trawl_type)
    H = np.arange(1, nlags+1)
    theoretical_acf = acf_func(H, theoretical_theta)
    inferred_acf = acf_func(H, inferred_theta)

    f, ax = plt.subplots()
    ax.plot(H, theoretical_acf, label='theoretical')
    ax.plot(H, inferred_acf, label='inferred')
    ax.plot(H, empirical_acf, label='empirical')
    plt.legend()
    return f
