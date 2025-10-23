import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import GMM
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions


def compute_alpha(gamma, beta):
    """Compute alpha from gamma and beta."""
    return np.sqrt(gamma**2 + beta**2)


def transform_params(unconstrained_params):
    """Transform unconstrained parameters to constrained space."""
    mu, log_delta, log_gamma, beta = unconstrained_params
    delta = np.exp(log_delta)
    gamma = np.exp(log_gamma)
    return mu, delta, gamma, beta


def nig_moments(mu, delta, gamma, beta):
    """
    Calculate NIG distribution moments using gamma parameterization.
    gamma = sqrt(alpha^2 - beta^2)
    """
    alpha = compute_alpha(gamma, beta)
    mean = mu + delta * beta / gamma
    variance = delta * alpha**2 / (gamma**3)
    E_X2 = mean**2 + variance
    skewness_term = (3 * delta * beta * (alpha**2 + 2 * beta**2)) / (gamma**5)
    E_X3 = mean**3 + 3 * mean * variance + skewness_term
    kurtosis_term = (3 * delta * (alpha**4 + 8 * alpha **
                     2 * beta**2 + 8 * beta**4)) / (gamma**7)
    E_X4 = mean**4 + 6 * mean**2 * variance + \
        4 * mean * skewness_term + kurtosis_term
    return np.array([mean, E_X2, E_X3, E_X4])


def moment_conditions(unconstrained_params, observed_moments, trawl):
    """
    Calculate moment conditions for GMM estimation.
    Works with unconstrained parameters that are transformed internally.
    """
    try:
        # Transform parameters to constrained space
        mu, delta, gamma, beta = transform_params(unconstrained_params)
        model_moments = nig_moments(mu, delta, gamma, beta)
        moment_errors = np.array([
            trawl - model_moments[0],
            trawl**2 - model_moments[1],
            trawl**3 - model_moments[2],
            trawl**4 - model_moments[3]
        ]).T
        return moment_errors
    except:
        return np.inf * np.ones((len(trawl), len(observed_moments)))


def estimate_nig_parameters(trawl):
    """
    Estimate NIG parameters using GMM with parameter transformation for bounds.
    Returns parameters in terms of mu, delta, gamma, and beta.
    """
    # Calculate sample moments for initial guess
    E_X1 = np.mean(trawl)
    E_X2 = np.mean(trawl ** 2)
    E_X3 = np.mean(trawl ** 3)
    E_X4 = np.mean(trawl ** 4)
    observed_moments = [E_X1, E_X2, E_X3, E_X4]

    # Initial guess in unconstrained space
    initial_guess = [
        1.0,          # mu
        np.log(2.0),   # log_delta
        np.log(2.5),   # log_gamma
        0.5           # beta
    ]

    class NIGGMM(GMM):
        def momcond(self, params):
            try:
                moment_errors = moment_conditions(
                    params, observed_moments, self.endog)
                if np.any(np.isnan(moment_errors)) or np.any(np.isinf(moment_errors)):
                    return np.inf * np.ones_like(moment_errors)
                return moment_errors
            except:
                return np.inf * np.ones((len(self.endog), len(observed_moments)))

    # Create instruments matrix and exogenous variables
    instruments = np.ones((len(trawl), 4))
    exog = np.ones((len(trawl), 1))

    # Initialize GMM with all required arguments
    gmm_model = NIGGMM(endog=trawl, exog=exog, instrument=instruments)

    try:
        result = gmm_model.fit(start_params=initial_guess, maxiter=10000)

        # Transform parameters back to constrained space
        mu, delta, gamma, beta = transform_params(result.params)

        # Compute alpha for reference
        alpha = compute_alpha(gamma, beta)

        return {
            "mu": mu,
            "delta": delta,
            "gamma": gamma,
            "beta": beta,
            "alpha": alpha
        }
    except Exception as e:
        raise ValueError(f"Optimization failed: {e}")


def plot_distributions(trawl, true_params, estimated_params):
    """
    Plot the true and estimated PDFs along with a histogram of the data using TensorFlow NIG.
    """
    plt.figure(figsize=(12, 7))

    # Create a range of x values for plotting
    x = np.linspace(np.min(trawl) - 1, np.max(trawl) +
                    1, 200).astype(np.float32)

    # Plot histogram of data
    plt.hist(trawl, bins=50, density=True,
             alpha=0.5, label='Data', color='gray')

    # Create NIG distributions with true and estimated parameters
    true_alpha = np.float32(compute_alpha(
        true_params['gamma'], true_params['beta']))
    est_alpha = np.float32(compute_alpha(
        estimated_params['gamma'], estimated_params['beta']))

    true_dist = tfd.NormalInverseGaussian(
        loc=np.float32(true_params['mu']),
        scale=np.float32(true_params['delta']),
        tailweight=true_alpha,
        skewness=np.float32(true_params['beta'])
    )

    est_dist = tfd.NormalInverseGaussian(
        loc=np.float32(estimated_params['mu']),
        scale=np.float32(estimated_params['delta']),
        tailweight=est_alpha,
        skewness=np.float32(estimated_params['beta'])
    )

    # Compute PDFs
    true_pdf = np.array(true_dist.prob(x))
    est_pdf = np.array(est_dist.prob(x))

    # Plot PDFs
    plt.plot(x, true_pdf, 'r-', label='True PDF', linewidth=2)
    plt.plot(x, est_pdf, 'b--', label='Estimated PDF', linewidth=2)

    plt.title('NIG Distribution: True vs Estimated PDF')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # Enable float64 precision in JAX
    jax.config.update("jax_enable_x64", True)

    # Set random seed
    np.random.seed(42)

    # Generate sample data with known parameters
    true_mu = -2.0
    true_delta = 2.0
    true_gamma = 1.5
    true_beta = 2.5

    true_params = {
        "mu": true_mu,
        "delta": true_delta,
        "gamma": true_gamma,
        "beta": true_beta
    }

    # Convert parameters for tensorflow probability NIG
    true_alpha = compute_alpha(true_gamma, true_beta)
    trawl = tfd.NormalInverseGaussian(
        loc=np.float32(true_mu),
        scale=np.float32(true_delta),
        tailweight=np.float32(true_alpha),
        skewness=np.float32(true_beta)
    ).sample(10**8, seed=PRNGKey(42))
    trawl = np.array(trawl)

    try:
        # Estimate parameters
        estimated_params = estimate_nig_parameters(trawl)

        # Print results
        print("\nTrue Parameters:")
        for param, value in true_params.items():
            print(f"{param}: {value:.4f}")

        print("\nEstimated Parameters:")
        for param, value in estimated_params.items():
            print(f"{param}: {value:.4f}")

        # Plot distributions
        plot_distributions(trawl, true_params, estimated_params)

    except ValueError as e:
        print(e)
