import scipy
import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import GMM
# from src.utils.modified_GMM_class import GMM

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions


def compute_gamma(beta):
    return 1 + jnp.abs(beta) / 5


def compute_alpha(gamma, beta):
    return jnp.sqrt(gamma**2 + beta**2)


def transform_to_tf_params(jax_mu, jax_scale, beta):
    """"returns mu, dleta, gamma, beta

    NOT

    mu, delta, alpha ,beta"""
    gamma = compute_gamma(beta)
    alpha = compute_alpha(gamma, beta)
    tf_delta = jax_scale**2 * gamma**3 / alpha**2
    tf_mu = jax_mu - beta * tf_delta / gamma
    return tf_mu, tf_delta, gamma, beta


def transform_to_jax_params(tf_mu, tf_delta, gamma, beta):
    """
    Transform TensorFlow parameters back to JAX parameters.
    """
    alpha = compute_alpha(gamma, beta)
    jax_scale = jnp.sqrt(tf_delta * alpha**2 / gamma**3)
    jax_mu = tf_mu + beta * tf_delta / gamma
    return jax_mu, jax_scale, beta


# def nig_moments(mu, delta, gamma, beta):
#    """
#    Calculate NIG distribution moments using gamma parameterization.
#    gamma = sqrt(alpha^2 - beta^2)
#    """
#
#    alpha = jnp.sqrt(gamma**2 + beta**2)
#
#    # First moment (mean)
#    mean = mu + delta * beta / gamma
#
#    # Second moment components
#    variance = delta * alpha**2 / (gamma**3)
#    E_X2 = mean**2 + variance
#
#    # Third moment components
#    skewness_term = 3 * delta * beta * (alpha**2 + 2 * beta**2) / (gamma**5)
#    E_X3 = mean**3 + 3 * mean * variance + skewness_term
#
#    # Fourth moment components - using the correct formulation
#    kurtosis_term = (3 * delta * (alpha**4 + 8 * alpha**2 * beta**2 + 8 * beta**4)) / (gamma**7)
#    E_X4 = mean**4 + 6 * mean**2 * variance + 4 * mean * skewness_term + kurtosis_term
#
#    return jnp.array([mean, E_X2, E_X3, E_X4])

# def nig_moments(mu, delta, gamma, beta):
#    """
#    Calculate NIG distribution moments using gamma parameterization.
#    gamma = sqrt(alpha^2 - beta^2)
#    """
#    alpha = np.sqrt(gamma**2 + beta**2)
#    mean = mu + delta * beta / gamma
#    variance = delta * alpha**2 / (gamma**3)
#    E_X2 = mean**2 + variance
#    skewness_term = (3 * delta * beta * (alpha**2 + 2 * beta**2)) / (gamma**5)
#    E_X3 = mean**3 + 3 * mean * variance + skewness_term
#
#    # Corrected kurtosis term using scipy-style formula
#    excess_kurtosis = 3 * delta * (alpha**4 + 8 * alpha**2 * beta**2 + 8 * beta**4) / (gamma**7) - 3
#    kurtosis_term = (excess_kurtosis + 3) * (variance**2)
#
#    E_X4 = mean**4 + 6 * mean**2 * variance + 4 * mean * skewness_term + kurtosis_term
#    return np.array([mean, E_X2, E_X3, E_X4])

def nig_moments(mu, delta, gamma, beta):

    # alpha, beta, mu, delta =  a/scale, b/scale, loc, scale
    alpha = (gamma**2+beta**2)**0.5
    a = alpha * delta
    b = beta * delta
    loc = mu
    scale = delta

    moments = [scipy.stats.norminvgauss(
        a=a, b=b, loc=loc, scale=scale).moment(i) for i in (1, 2, 3, 4)]
    return np.array(moments)


def moment_conditions_jax(theta_jax, observed_moments, trawl):
    """
    Calculate moment conditions for GMM estimation with theta_jax.
    Compute all four moments while estimating three parameters.
    """
    try:
        jax_mu, jax_scale, beta = theta_jax
        tf_mu, tf_delta, gamma, _ = transform_to_tf_params(
            jax_mu, jax_scale, beta)
        model_moments = nig_moments(tf_mu, tf_delta, gamma, beta)
        moment_errors = jnp.array([
            trawl - model_moments[0],
            trawl**2 - model_moments[1],
            trawl**3 - model_moments[2],
            trawl**4 - model_moments[3]
        ]).T
        return moment_errors

        # Debug: Print moments during optimization
        # print(f"Observed Moments: {observed_moments}")
        # print(f"Model Moments: {model_moments}")
    except:
        return jnp.inf * jnp.ones((len(trawl), 4))  # Match shape to 4 moments


def estimate_jax_parameters(trawl, initial_guess=None):
    """
    Estimate parameters using GMM with theta_jax (3 parameters).
    """
    observed_moments = [
        np.mean(trawl),
        np.mean(trawl ** 2),
        np.mean(trawl ** 3),
        np.mean(trawl ** 4),
    ]

    initial_guess = np.array([
        np.mean(trawl),  # jax_mu
        np.std(trawl),  # jax_scale
        0.0   # jax_beta
    ])

    class JAXGMM(GMM):
        def momcond(self, params):
            try:
                moment_errors = moment_conditions_jax(
                    params, observed_moments, self.endog)
                if np.any(np.isnan(moment_errors)) or np.any(np.isinf(moment_errors)):
                    return np.inf * np.ones_like(moment_errors)
                return np.array(moment_errors)  # Ensure NumPy array
            except:
                return np.inf * np.ones((len(self.endog), 4))

    instruments = np.ones((len(trawl), 4))  # Match to 4 moments
    exog = np.ones((len(trawl), 1))

    gmm_model = JAXGMM(endog=np.array(trawl), exog=exog,
                       instrument=instruments)

    try:
        result = gmm_model.fit(start_params=initial_guess, maxiter=1000)
        return result
        # jax_mu, jax_scale, jax_beta = result.params
        # return {
        #    "jax_mu": jax_mu,
        #    "jax_scale": jax_scale,
        #    "jax_beta": jax_beta
        # }
    except Exception as e:
        raise ValueError(f"Optimization failed: {e}")


# Example usage
if __name__ == "__main__":
    # Enable float64 precision in JAX
    # jax.config.update("jax_enable_x64", True)

    # Set random seed
    np.random.seed(42)

    # Generate sample data with known parameters
    true_jax_mu = 0.25
    true_jax_scale = 1.4
    beta = 0.5

    true_theta_jax = {
        "jax_mu": true_jax_mu,
        "jax_scale": true_jax_scale,
        "jax_beta": beta
    }

    # Transform to TensorFlow parameters for data generation
    tf_mu, tf_delta, gamma, beta = transform_to_tf_params(
        true_jax_mu, true_jax_scale, beta
    )

    # Generate synthetic data
    trawl_sampler = tfd.NormalInverseGaussian(
        loc=tf_mu,
        scale=tf_delta,
        tailweight=compute_alpha(gamma, beta),
        skewness=beta
    )
    trawl = trawl_sampler.sample(
        10**3, seed=PRNGKey(np.random.randint(1, 10000)))
    trawl = np.array(trawl)

    # Estimate parameters
    estimated_params = estimate_jax_parameters(trawl)

    # Print results
    print("\nTrue Parameters:")
    for param, value in true_theta_jax.items():
        print(f"{param}: {value:.4f}")

    print("\nEstimated Parameters:")
    for param, value in estimated_params.items():
        print(f"{param}: {value:.4f}")
