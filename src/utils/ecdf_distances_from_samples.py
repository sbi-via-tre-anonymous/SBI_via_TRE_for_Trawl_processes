import numpy as np
from scipy import stats
from scipy.integrate import quad
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import anderson_statistic


def check_samples(samples):

    valid_indicator = (samples >= 0) & (samples <= 1)
    assert sum(valid_indicator) == len(samples)


def kolmogorov_smirnov_uniform(samples):
    """KS distance from uniform[0,1] - for comparison"""
    return stats.kstest(samples, 'uniform', args=(0, 1))


def wasserstein_1_analytical(samples):
    """
    Analytical formula for 1-Wasserstein distance between empirical 
    distribution and Uniform[0,1]

    The formula is: W_1 = ∫|F_n(x) - x| dx from 0 to 1
    where F_n is the empirical CDF
    """
    samples = np.sort(samples)
    n = len(samples)

    if n == 0:
        return np.inf

    # Add boundary points
    x = np.concatenate([[0], samples, [1]])

    # Compute the area between ECDF and uniform CDF
    total_distance = 0

    for i in range(len(x) - 1):
        # ECDF value in this interval
        ecdf_val = i / n

        # Integrate |ecdf_val - x| from x[i] to x[i+1]
        # This splits into two cases depending on where ecdf_val intersects y=x

        if ecdf_val <= x[i]:
            # ECDF is below y=x throughout interval
            integral = (x[i+1]**2 - x[i]**2)/2 - ecdf_val*(x[i+1] - x[i])
        elif ecdf_val >= x[i+1]:
            # ECDF is above y=x throughout interval
            integral = ecdf_val*(x[i+1] - x[i]) - (x[i+1]**2 - x[i]**2)/2
        else:
            # ECDF crosses y=x at x=ecdf_val
            integral = (ecdf_val - x[i])**2/2 + (x[i+1] - ecdf_val)**2/2

        total_distance += integral

    return total_distance

# sanity check to use with n_uniform very large


def wasserstein_1_scipy(samples, n_uniform):
    """
    Approximate using scipy with samples from uniform distribution

    Parameters:
    -----------
    samples : array-like
        Samples to compare with uniform
    n_uniform : int
        Number of uniform samples to generate
    """
    uniform_samples = np.random.uniform(0, 1, n_uniform)
    return stats.wasserstein_distance(samples, uniform_samples)


def cramer_von_mises_uniform(samples):
    """Cramér-von Mises test against uniform[0,1]"""
    # Sort and clip samples
    samples = np.sort(samples)
    n = len(samples)

    if n == 0:
        return np.inf

    # CvM statistic
    i = np.arange(1, n + 1)
    cvm = 1/(12*n) + np.sum((samples - (2*i - 1)/(2*n))**2)

    return cvm

# anderson_statistic(cal_rank, dist= stats.uniform, fit = False)
# anderson_statistic(uncal_rank, dist= stats.uniform, fit = False)


def anderson_darling_uniform(samples, return_components=False):
    """
    Compute Anderson-Darling distance between samples and Uniform[0,1] distribution.

    This gives more weight to deviations in the tails (near 0 and 1).

    Parameters:
    -----------
    samples : array-like
        Samples to compare with Uniform[0,1]
    return_components : bool
        If True, return detailed components

    Returns:
    --------
    float : Anderson-Darling statistic
    dict : Components (if return_components=True)
    """
    # Sort samples and ensure they're in [0,1]
    samples = np.sort(samples)
    samples_valid = samples[(samples >= 0) & (samples <= 1)]

    if len(samples_valid) == 0:
        return np.inf if not return_components else (np.inf, {})

    n = len(samples_valid)

    # For uniform[0,1], theoretical CDF is F(x) = x
    # Empirical CDF values
    i = np.arange(1, n + 1)
    ecdf_values = i / n

    # Theoretical CDF values at sample points
    uniform_cdf = samples_valid

    # Anderson-Darling weights for uniform distribution
    # Weight = 1 / (F(x) * (1 - F(x))) = 1 / (x * (1 - x))
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    weights = 1 / (uniform_cdf * (1 - uniform_cdf) + epsilon)

    # Compute differences between empirical and theoretical CDF
    # At each sample point x_i, empirical CDF jumps to i/n
    cdf_diff = ecdf_values - uniform_cdf

    # Anderson-Darling statistic
    # A² = n * Σ w(x_i) * (F_n(x_i) - F(x_i))²
    ad_statistic = n * np.sum(weights * cdf_diff**2)

    # Alternative formula (equivalent) used in many implementations:
    # A² = -n - (1/n) * Σ(2i-1) * [ln(F(x_i)) + ln(1-F(x_{n+1-i}))]
    term1 = (2 * i - 1) * (np.log(uniform_cdf + epsilon) +
                           np.log(1 - uniform_cdf[::-1] + epsilon))
    ad_statistic_alt = -n - np.sum(term1) / n

    if return_components:
        return ad_statistic, {
            'samples': samples_valid,
            'ecdf': ecdf_values,
            'uniform_cdf': uniform_cdf,
            'weights': weights,
            'cdf_diff': cdf_diff,
            'weighted_diff': weights * cdf_diff**2,
            'ad_alt': ad_statistic_alt
        }

    return ad_statistic
