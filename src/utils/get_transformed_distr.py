# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions


def get_transformed_beta_distr(beta_hyperparams):
    """
    Returns the following distribution, from which we can both sample and
    compute pdfs, grad pdfs, etc:

    lower + (upper-lower) * Beta(alpha = concentration1, beta = concentration0)  
    i.e. Beta(alpha = concentration1, beta = concentration0) scaled to [lower,upper]
    """

    lower, upper, concentration1, concentration0 = beta_hyperparams

    # base beta dist
    beta_dist = tfp.distributions.Beta(concentration1, concentration0)

    # Transform Beta to have location and scale
    transformed_distr = tfd.TransformedDistribution(
        distribution=beta_dist,
        bijector=tfp.bijectors.Chain([
            tfp.bijectors.Shift(lower),
            tfp.bijectors.Scale(upper-lower)
        ])
    )

    return transformed_distr
