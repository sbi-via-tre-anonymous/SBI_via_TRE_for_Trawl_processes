if True:
    from path_setup import setup_sys_path
    setup_sys_path()


import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import GMM
import scipy
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.random import PRNGKey


def transform_to_constrained_jax(unconstrained_params, lower=np.array([-1., 0.5, -5.]),
                                 upper=np.array([1., 1.5, 5.])
                                 ):
    """
    Transform parameters from unconstrained space (-inf, inf) to constrained space (lower, upper)
    using a sigmoid transformation.
    """
    # Using sigmoid transformation: constrained = lower + (upper - lower) * sigmoid(unconstrained)
    constrained_params = lower + \
        (upper - lower) * (1 / (1 + np.exp(-unconstrained_params)))
    return constrained_params


def transform_jax_to_unconstrained(constrained_params, lower=np.array([-1., 0.5, -5.]),
                                   upper=np.array([1., 1.5, 5.])
                                   ):
    """
    Transform parameters from constrained space (lower, upper) to unconstrained space (-inf, inf)
    using the inverse sigmoid transformation.
    """
    # Inverse of sigmoid transformation
    constrained_ratio = (constrained_params - lower) / (upper - lower)
    # Avoid numerical issues at boundaries
    constrained_ratio = np.clip(constrained_ratio, 1e-10, 1 - 1e-10)
    unconstrained_params = -np.log(1/constrained_ratio - 1)
    return unconstrained_params


def transform_to_tf_params(jax_mu, jax_scale, beta):
    """"returns mu, dleta, gamma, beta

    NOT

    mu, delta, alpha ,beta"""
    gamma = 1 + jnp.abs(beta) / 5
    alpha = jnp.sqrt(gamma**2 + beta**2)
    tf_delta = jax_scale**2 * gamma**3 / alpha**2
    tf_mu = jax_mu - beta * tf_delta / gamma
    return tf_mu, tf_delta, gamma, beta


def transform_to_tf_params_with_alpha_not_gamma(jax_mu, jax_scale, beta):
    """"returns mu, dleta, gamma, beta

    NOT

    mu, delta, alpha ,beta"""
    gamma = 1 + jnp.abs(beta) / 5
    alpha = jnp.sqrt(gamma**2 + beta**2)
    tf_delta = jax_scale**2 * gamma**3 / alpha**2
    tf_mu = jax_mu - beta * tf_delta / gamma
    return tf_mu, tf_delta, alpha, beta


def transform_to_jax_params(tf_mu, tf_delta, gamma, beta):
    """
    Transform TensorFlow parameters back to JAX parameters.
    """
    alpha = jnp.sqrt(gamma**2 + beta**2)
    jax_scale = jnp.sqrt(tf_delta * alpha**2 / gamma**3)
    jax_mu = tf_mu + beta * tf_delta / gamma
    return jax_mu, jax_scale, beta


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


class JAXGMM(GMM):
    def __init__(self, endog, exog, instrument):
        super().__init__(endog, exog, instrument)

    # def momcond(self, params):
    #    try:
    #        unconstrained_params = transform_jax_to_unconstrained(params)
    #        moment_errors = moment_conditions_unconstrained(
    #            unconstrained_params, self.endog)
    #        if np.any(np.isnan(moment_errors)) or np.any(np.isinf(moment_errors)):
    #            return np.inf * np.ones_like(moment_errors)
    #        return np.array(moment_errors)
    #    except:
    #        return np.inf * np.ones((len(self.endog), 4))

    def momcond(self, unconstrained_params):
        try:
            # unconstrained_params = transform_jax_to_unconstrained(params)
            moment_errors = moment_conditions_unconstrained(
                unconstrained_params, self.endog)
            if np.any(np.isnan(moment_errors)) or np.any(np.isinf(moment_errors)):
                return np.inf * np.ones_like(moment_errors)
            return np.array(moment_errors)
        except:
            return np.inf * np.ones((len(self.endog), 4))


def moment_conditions_unconstrained(theta_unconstrained, trawl):

    theta_jax = transform_to_constrained_jax(theta_unconstrained)

    return moment_conditions_jax(theta_jax, trawl)


def moment_conditions_jax(theta_jax, trawl):
    """
    Calculate moment conditions for GMM estimation with theta_jax.
    Includes both distribution moments and ACF comparisons.
    Trims num_lags observations from all moments for consistent length.

    Parameters:
    -----------
    theta_jax : array-like
        Parameters [jax_mu, jax_scale, beta, acf_gamma, acf_eta]
    trawl : array-like
        Observed data
    num_lags : int
        Number of ACF lags to compare
    acf_func : function
        Function to compute theoretical ACF based on trawl_type
    """
    try:
        jax_mu, jax_scale, beta = theta_jax

        # Transform parameters and compute distributional moments
        tf_mu, tf_delta, gamma, _ = transform_to_tf_params(
            jax_mu, jax_scale, beta)
        model_moments = nig_moments(tf_mu, tf_delta, gamma, beta)

        # Compute marginal moment conditions with trimmed series
        marginal_errors = jnp.array([
            trawl - model_moments[0],
            trawl**2 - model_moments[1],
            trawl**3 - model_moments[2],
            trawl**4 - model_moments[3]
        ]).T

        moment_errors = marginal_errors

        return moment_errors
    except:
        # Return appropriate sized array of infinities
        return jnp.inf * jnp.ones((len(trawl), 4))


def estimate_jax_parameters(trawl, initial_guess=None, optim = 'bfgs'):
    """
    Estimate parameters using GMM with theta_jax, including ACF comparisons.

    Parameters:
    -----------
    trawl : array-like
        Observed data
    num_lags : int, optional
        Number of ACF lags to compare (default=5)
    trawl_type : str, optional
        Type of trawl process to use (default='exp')
    """

    assert initial_guess is not None

    # Update instruments matrix to account for additional ACF moments
    instruments = np.ones((len(trawl), 4))
    exog = np.ones((len(trawl), 1))

    gmm_model = JAXGMM(endog=np.array(trawl),
                       exog=exog,
                       instrument=instruments
                       )

    try:
        result = gmm_model.fit(start_params=transform_jax_to_unconstrained(
            initial_guess), optim_method = optim)
        # jax_mu, jax_scale, jax_beta = result.params
        # return {
        #    "jax_mu": jax_mu,
        #    "jax_scale": jax_scale,
        #    "jax_beta": jax_beta
        # }
        return result
    except Exception as e:
        print(f"Error in parameter estimation: {str(e)}")
        return None


if __name__ == '__main__':
    import os
    import yaml
    import matplotlib.pyplot as plt
    import numpy as np
    from statsmodels.tsa.stattools import acf as compute_empirical_acf
    from tqdm import tqdm
    import pandas as pd

    # Load dataset & configuration; later on double check the true_theta is the same in the dataset and in the MLE dataframe
    seq_len = 1000
    num_rows_to_load = 80  # how much data to load
    num_trawls_to_use = 500  # how much data to do GMM on
    ###########

    models_path = os.path.dirname(
        os.path.dirname(os.path.dirname(folder_path)))
    dataset_path = os.path.join(
        models_path, 'val_dataset', f'val_dataset_{seq_len}')
    val_x_path = os.path.join(dataset_path, 'val_x_joint.npy')
    val_thetas_path = os.path.join(dataset_path, 'val_thetas_joint.npy')

    # Load first few rows of val_x with memory mapping
    val_x = np.load(val_x_path, mmap_mode='r')[:num_rows_to_load]
    val_thetas = np.load(val_thetas_path)[:num_rows_to_load]

    val_x = val_x.reshape(-1, seq_len)
    val_thetas = val_thetas.reshape(-1, val_thetas.shape[-1])

    ########### empty lists to append results to ############
    true_theta_list = []
    GMM_list = []
    # result_list = []

    for index_ in tqdm(range(num_trawls_to_use)):  # tqdm(range(len(mle_df))):

        # trawl_idx = mle_df.iloc[index_].idx
        # true_theta_from_mle = mle_df.iloc[index_].true_theta
        # true_theta = true_thetas[trawl_idx]
        # assert all(np.isclose(true_theta_from_mle, true_theta))

        # true_trawl = true_trawls[trawl_idx]

        true_trawl = val_x[index_]
        true_theta = val_thetas[index_]

        # Estimate parameters using transformed approach
        result = estimate_jax_parameters(
            true_trawl,
            initial_guess=true_theta[2:]
        )
        # result_list.append(result_dict)
        true_theta_list.append(true_theta)

        if result is not None:

            GMM_list.append(transform_to_constrained_jax(result.params))

        else:

            GMM_list.append(None)

    df = pd.DataFrame({'true_theta': true_theta_list,
                      'GMM': GMM_list})

    df.to_pickle(
        f'margianl_GMM_seq_len_{seq_len}_num_trawls_to_use_{num_trawls_to_use}.pkl')
