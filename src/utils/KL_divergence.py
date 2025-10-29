import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from functools import partial

tfp_dist = tfp.distributions


def convert_3_to_4_param_nig(params):

    jax_mu, jax_scale, beta = params
    gamma = 1 + jnp.abs(beta) / 5
    alpha = jnp.sqrt(gamma**2 + beta**2)
    tf_delta = jax_scale**2 * gamma**3 / alpha**2
    tf_mu = jax_mu - beta * tf_delta / gamma

    # jnp.vstack([tf_mu, tf_delta, alpha, beta]).transpose()
    return jnp.array([tf_mu, tf_delta, alpha, beta])


def monte_carlo_kl_nig(params1, params2, key, num_samples):
    """
    Monte Carlo approximation of KL divergence between two NIG distributions.

    Args:
        key: JAX random key for sampling.
        params1: Tuple (mu1, delta1, alpha1, beta1) for the first NIG distribution.
        params2: Tuple (mu2, delta2, alpha2, beta2) for the second NIG distribution.
        num_samples: Number of samples to use for the Monte Carlo approximation (static argument).

    Returns:
        KL divergence estimate.
    """
    # Unpack parameters
    mu1, delta1, alpha1, beta1 = params1
    mu2, delta2, alpha2, beta2 = params2

    # Create the NIG distributions
    nig1 = tfp_dist.NormalInverseGaussian(
        loc=mu1, scale=delta1, tailweight=alpha1, skewness=beta1, validate_args=True
    )
    nig2 = tfp_dist.NormalInverseGaussian(
        loc=mu2, scale=delta2, tailweight=alpha2, skewness=beta2, validate_args=True
    )

    # Sample from the first distribution
    samples = nig1.sample(num_samples, seed=key)

    # Compute log probabilities for both distributions
    log_p = nig1.log_prob(samples)
    log_q = nig2.log_prob(samples)

    # Monte Carlo estimate of KL divergence
    kl_mc = jnp.mean(log_p - log_q)
    # kl_mc_std = jnp.std(log_p - log_q) / num_samples**0.5

    return kl_mc  # ,kl_mc_std


@partial(jax.jit, static_argnames=("num_samples",))
def monte_carlo_kl_3_param_nig(params1, params2, key, num_samples):

    key, subkey = jax.random.split(key)

    # vmap_monte_carlo_kl_nig = jax.vmap(
    #    monte_carlo_kl_nig, in_axes=(0, 0, 0, None))
    params1 = convert_3_to_4_param_nig(params1)
    params2 = convert_3_to_4_param_nig(params2)
    return monte_carlo_kl_nig(params1, params2, subkey, num_samples), key

    # result = vmap_monte_carlo_kl_nig(params1, params2, key, num_samples)
    # return jnp.mean(result)


vec_monte_carlo_kl_3_param_nig = jax.jit(jax.vmap(monte_carlo_kl_3_param_nig,
                                                  in_axes=(0, 0, 0, None)),
                                         static_argnames=('num_samples',))


if __name__ == '__main__':
    import numpy as np
    key = jax.random.PRNGKey(np.random.randint(2, 100000))
    num_samples = 10**5
    params1 = jnp.array((-2.5, 1.5, 3.2, 1.6))
    params2 = jnp.array((-2., 2.5, 3.9, 1.5))
    result = monte_carlo_kl_nig(params1, params2, key, num_samples)
    print(result)

    vec_params1 = jnp.vstack([(-2.5, 1.75, 3.9), (-2., 2.8, 3.2)])
    vec_params2 = jnp.vstack([(-2.25, 1.5, 3.2), (-2.6, 2.2, 3.9)])
    vec_key = jax.random.split(key, 2)
    result2 = vec_monte_carlo_kl_3_param_nig(
        vec_params1, vec_params2, vec_key, num_samples)
    print(result2)
