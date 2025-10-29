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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
import yaml
import pandas as pd
from tqdm import tqdm
import warnings


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
        # Suppress scipy optimization warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=scipy.optimize.OptimizeWarning)
            warnings.filterwarnings("ignore", message=".*Desired error not necessarily achieved.*")
            
        result = gmm_model.fit(start_params=transform_jax_to_unconstrained(
                initial_guess),
                optim_args={'disp': 0},
                optim_method = optim)  # , maxiter=10)
        return result
    except Exception as e:
        # Uncomment the line below if you want to see the actual errors
        #print(f"Error in parameter estimation: {str(e)}")
        return None


def process_single_trawl(args):
    """
    Process a single trawl for GMM estimation.
    This function needs to be at module level for multiprocessing to work.
    
    Parameters:
    -----------
    args : tuple
        (index, true_trawl, true_theta)
    
    Returns:
    --------
    tuple : (index, true_theta, gmm_result)
    """
    index, true_trawl, true_theta = args
    
    try:
        # Estimate parameters using transformed approach
        result = estimate_jax_parameters(
            true_trawl,
            initial_guess=true_theta[2:]
        )
        
        if result is not None:
            gmm_result = transform_to_constrained_jax(result.params)
        else:
            gmm_result = None
            
        return index, true_theta, gmm_result
    
    except Exception as e:
        print(f"Error processing trawl {index}: {str(e)}")
        return index, true_theta, None


def parallel_gmm_estimation(val_x, val_thetas, num_trawls_to_use, n_jobs=None):
    """
    Perform GMM estimation in parallel.
    
    Parameters:
    -----------
    val_x : array
        Trawl data
    val_thetas : array
        True theta values
    num_trawls_to_use : int
        Number of trawls to process
    n_jobs : int, optional
        Number of parallel jobs. If None, uses all available CPUs.
    
    Returns:
    --------
    tuple : (true_theta_list, GMM_list)
    """
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    
    print(f"Using {n_jobs} CPU cores for parallel processing")
    
    # Prepare arguments for parallel processing
    args_list = [(i, val_x[i], val_thetas[i]) for i in range(num_trawls_to_use)]
    
    true_theta_list = [None] * num_trawls_to_use
    GMM_list = [None] * num_trawls_to_use
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all jobs
        future_to_index = {executor.submit(process_single_trawl, args): args[0] 
                          for args in args_list}
        
        # Process completed jobs with enhanced progress bar
        successful_count = 0
        failed_count = 0
        
        with tqdm(total=num_trawls_to_use, desc="Processing trawls", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Success: {postfix}') as pbar:
            
            for future in as_completed(future_to_index):
                try:
                    index, true_theta, gmm_result = future.result()
                    true_theta_list[index] = true_theta
                    GMM_list[index] = gmm_result
                    
                    if gmm_result is not None:
                        successful_count += 1
                    else:
                        failed_count += 1
                        
                    # Update progress bar with success rate
                    pbar.set_postfix_str(f"{successful_count}/{successful_count + failed_count}")
                    pbar.update(1)
                    
                except Exception as e:
                    index = future_to_index[future]
                    failed_count += 1
                    print(f"Error processing trawl {index}: {str(e)}")
                    true_theta_list[index] = val_thetas[index]
                    GMM_list[index] = None
                    
                    # Update progress bar with success rate
                    pbar.set_postfix_str(f"{successful_count}/{successful_count + failed_count}")
                    pbar.update(1)
    
    return true_theta_list, GMM_list


if __name__ == '__main__':
    
    # Suppress scipy optimization warnings globally
    # sometimes the warnings to do not actualyl get supressed and isntead are pritned to screen
    # which slows the process quite a bit on my pc
    # looks like an internal statsmodels error
    warnings.filterwarnings("ignore", category=scipy.optimize.OptimizeWarning)
    warnings.filterwarnings("ignore", message=".*Desired error not necessarily achieved.*")
    
    # Load dataset & configuration
    models_path =  os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)),'models_and_simulated_datasets')
    seq_len = 1000
    num_rows_to_load = 160  # how much data to load
    num_trawls_to_use = 10**4  # how much data to do GMM on
    
    # Optional: specify number of CPU cores to use (None = use all available)
    n_jobs = 8 #multiprocessing.cpu_count()-1  # or set to specific number like 4, 8, etc.
    dataset_path = os.path.join(
        models_path, 'validation_datasets', f'val_dataset_{seq_len}')
    val_x_path = os.path.join(dataset_path, 'val_x_joint.npy')
    val_thetas_path = os.path.join(dataset_path, 'val_thetas_joint.npy')

    # Load first few rows of val_x with memory mapping
    val_x = np.load(val_x_path, mmap_mode='r')[:num_rows_to_load]
    val_thetas = np.load(val_thetas_path)[:num_rows_to_load]

    val_x = val_x.reshape(-1, seq_len)
    val_thetas = val_thetas.reshape(-1, val_thetas.shape[-1])

    print(f"Loaded {len(val_x)} trawls, processing {num_trawls_to_use}")
    print(f"Available CPU cores: {multiprocessing.cpu_count()}")
    
    # Perform parallel GMM estimation
    true_theta_list, GMM_list = parallel_gmm_estimation(
        val_x, val_thetas, num_trawls_to_use, n_jobs=n_jobs
    )
    
    # Create DataFrame and save results
    df = pd.DataFrame({'true_theta': true_theta_list,
                      'GMM': GMM_list})
    
    results_path = os.path.join(models_path,'point_estimators','GMM',f'GMM_results_seq_len_{seq_len}')
    output_filename = f'marginal_GMM_seq_len_{seq_len}.pkl'
    df.to_pickle(os.path.join(results_path,output_filename))
    
    print(f"Results saved to {output_filename}")
    
    # Print some statistics
    successful_estimations = sum(1 for x in GMM_list if x is not None)
    print(f"Successful estimations: {successful_estimations}/{num_trawls_to_use} "
          f"({100*successful_estimations/num_trawls_to_use:.1f}%)")