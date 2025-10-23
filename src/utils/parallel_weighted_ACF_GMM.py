if True:
    from path_setup import setup_sys_path
    setup_sys_path()

from src.utils.acf_functions import get_acf
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf as compute_empirical_acf
from statsmodels.sandbox.regression.gmm import GMM
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import pandas as pd
from tqdm import tqdm
import multiprocessing
from functools import partial

import warnings
from scipy.optimize import OptimizeWarning
warnings.filterwarnings('ignore', category=OptimizeWarning)

# Define transformation functions to map between unconstrained and constrained space
def transform_to_constrained(unconstrained_params, lower=10.0, upper=20.0):
    """
    Transform parameters from unconstrained space (-inf, inf) to constrained space (lower, upper)
    using a sigmoid transformation.
    """
    # Using sigmoid transformation: constrained = lower + (upper - lower) * sigmoid(unconstrained)
    constrained_params = lower + \
        (upper - lower) * (1 / (1 + np.exp(-unconstrained_params)))
    return constrained_params


def transform_to_unconstrained(constrained_params, lower=10.0, upper=20.0):
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


def acf_moment_conditions(params, trawl, num_lags, acf_func):
    acf_gamma, acf_eta = params
    # Compute demeaned series for ACF calculation
    demeaned_trawl = trawl - np.mean(trawl)
    variance = np.var(trawl)

    # Initialize array for ACF errors
    acf_errors = np.zeros((len(trawl) - num_lags, num_lags))

    for k in range(1, num_lags + 1):
        # Calculate product of lagged values
        prod = demeaned_trawl[:-k] * demeaned_trawl[k:]
        # Calculate empirical products
        empirical_products = prod / variance

        # Calculate theoretical ACF
        theoretical_acf = acf_func(k, np.array([acf_gamma, acf_eta]))
        # Calculate error
        error = empirical_products[:len(trawl) - num_lags] - theoretical_acf
        acf_errors[:, k - 1] = error

    return acf_errors


def estimate_acf_parameters_transformed(trawl, num_lags, trawl_function_name,
                                        initial_guess=None, lower_bound=10.0, upper_bound=20.0):
    """
    Estimate ACF parameters using GMM with parameter transformation.
    """

    # Set initial guess in constrained space if not provided
    if initial_guess is None:
        if trawl_function_name == 'sup_IG':
            # Middle of range for [acf_gamma, acf_eta]
            initial_guess = np.array([15.0, 15.0])
        else:
            raise ValueError("Unsupported trawl function name")

    # Update instruments matrix to account for ACF moments only
    n = len(trawl)
    instruments = np.ones((len(trawl), num_lags))
    exog = np.ones((len(trawl), 1))

    # Create the transformed GMM model
    gmm_model = TransformedACFGMM(
        endog=np.array(trawl),
        exog=exog,
        instrument=instruments,
        num_lags=num_lags,
        trawl_function_name=trawl_function_name,
        lower_bound=lower_bound,
        upper_bound=upper_bound
    )

    try:
        # Transform initial guess to unconstrained space
        unconstrained_initial = gmm_model.transform_to_unconstrained(
            initial_guess)

        # Fit the model in unconstrained space
        result = gmm_model.fit(
            start_params=unconstrained_initial,
            # No bounds needed since we're in unconstrained space
            maxiter=35,
            optim_args={'disp': 0}  # Pass disp through optim_args

        )

        # Transform the results back to constrained space for interpretation
        constrained_params = gmm_model.transform_to_constrained(result.params)
        acf_gamma, acf_eta = constrained_params

        # For standard errors of moment conditions, use the constrained parameters
        final_moment_errors = gmm_model.momcond(result.params)
        std_errors = np.std(final_moment_errors, axis=0)

        # Create a result object with both unconstrained and constrained parameters
        result_dict = {
            "unconstrained_params": result.params,
            "constrained_params": constrained_params,
            "acf_gamma": acf_gamma,
            "acf_eta": acf_eta,
            "std_errors": std_errors,
            "original_result": result
        }

        return result_dict

    except Exception as e:
        print(f"Error in parameter estimation: {str(e)}")
        return None


class TransformedACFGMM(GMM):
    def __init__(self, endog, exog, instrument, num_lags, trawl_function_name, lower_bound=10.0, upper_bound=20.0):
        super().__init__(endog, exog, instrument)
        self.num_lags = num_lags
        self.acf_func = get_acf(trawl_function_name)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def transform_to_constrained(self, unconstrained_params):
        """Transform parameters from unconstrained space to constrained space"""
        return self.lower_bound + (self.upper_bound - self.lower_bound) * (1 / (1 + np.exp(-unconstrained_params)))

    def transform_to_unconstrained(self, constrained_params):
        """Transform parameters from constrained space to unconstrained space"""
        constrained_ratio = (constrained_params - self.lower_bound) / \
            (self.upper_bound - self.lower_bound)
        # Avoid numerical issues at boundaries
        constrained_ratio = np.clip(constrained_ratio, 1e-10, 1 - 1e-10)
        return -np.log(1/constrained_ratio - 1)

    def momcond(self, unconstrained_params):
        """
        Calculate moment conditions using transformed parameters.
        The optimization happens in unconstrained space, but the moment conditions
        use parameters transformed to the constrained space.
        """
        try:
            # Transform parameters to constrained space
            constrained_params = self.transform_to_constrained(
                unconstrained_params)

            # Use the constrained parameters in the original moment conditions
            moment_errors = acf_moment_conditions(
                constrained_params, self.endog, self.num_lags, self.acf_func)

            # Check for numerical issues
            has_nan = np.any(np.isnan(moment_errors))
            has_inf = np.any(np.isinf(moment_errors))

            if has_nan or has_inf:
                print(
                    f"WARNING: Found NaN ({has_nan}) or Inf ({has_inf}) in moment errors with params {constrained_params}")
                # Return a large but FINITE penalty
                return 1e6 * np.ones_like(moment_errors)

            return np.array(moment_errors)
        except Exception as e:
            print(
                f"EXCEPTION in momcond with params {unconstrained_params} (constrained: {self.transform_to_constrained(unconstrained_params)}): {str(e)}")
            # Return a large but FINITE penalty
            return 1e6 * np.ones((len(self.endog) - self.num_lags, self.num_lags))


# 3


########################################################

# Worker function for parallel processing
def process_single_trawl(args):
    """
    Process a single trawl sequence for parallel execution.

    Args:
        args: tuple containing (index, val_x, val_thetas, num_lags, trawl_function_name, lower_bound, upper_bound)

    Returns:
        tuple: (index, true_theta, GMM_result)
    """
    index_, val_x, val_thetas, num_lags, trawl_function_name, lower_bound, upper_bound = args

    true_trawl = val_x[index_]
    true_theta = val_thetas[index_]

    # Estimate parameters using transformed approach
    result_dict = estimate_acf_parameters_transformed(
        true_trawl, num_lags, trawl_function_name,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        initial_guess=true_theta[:2]
    )
    if result_dict is not None:
        GMM_result = result_dict['constrained_params']
    else:
        GMM_result = None
    return (index_, true_theta, GMM_result)


if __name__ == '__main__':
    import os
    import yaml
    import matplotlib.pyplot as plt
    import numpy as np
    from statsmodels.tsa.stattools import acf as compute_empirical_acf

    # Load dataset & configuration; later on double check the true_theta is the same in the dataset and in the MLE dataframe
    models_path =  os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)),'models_and_simulated_datasets')
    seq_len = 2000
    num_rows_to_load = 160  # how much data to load
    num_trawls_to_use = 10**4  # how much data to do GMM on
    ###########
    trawl_process_type = 'sup_ig_nig_5p'
    trawl_function_name = 'sup_IG'
    # Set bounds for parameters
    lower_bound = 10.0
    upper_bound = 20.0

    # Get parameters from config
    num_lags = 35
    acf_func = get_acf(trawl_function_name)

    # Load dataset

    dataset_path = os.path.join(
        models_path, 'validation_datasets', f'val_dataset_{seq_len}')
    val_x_path = os.path.join(dataset_path, 'val_x_joint.npy')
    val_thetas_path = os.path.join(dataset_path, 'val_thetas_joint.npy')

    # Load first few rows of val_x with memory mapping
    val_x = np.load(val_x_path, mmap_mode='r')[:num_rows_to_load]
    val_thetas = np.load(val_thetas_path)[:num_rows_to_load]

    val_x = val_x.reshape(-1, seq_len)
    val_thetas = val_thetas.reshape(-1, val_thetas.shape[-1])

    ########### Parallel processing setup ############

    # Get number of CPUs to use (all available)
    num_cpus = 4#multiprocessing.cpu_count()-1
    print(f"Using {num_cpus} CPUs for parallel processing")

    # Prepare arguments for parallel processing
    # Create a list of argument tuples for each index
    process_args = [
        (index_, val_x, val_thetas, num_lags,
         trawl_function_name, lower_bound, upper_bound)
        for index_ in range(num_trawls_to_use)
    ]

    # Use multiprocessing Pool to process in parallel
    with multiprocessing.Pool(processes=num_cpus) as pool:
        # Process all trawls in parallel with progress bar
        results = list(tqdm(
            pool.imap(process_single_trawl, process_args),
            total=num_trawls_to_use,
            desc="Processing trawls"
        ))

    # Sort results by index to maintain original order
    results.sort(key=lambda x: x[0])

    # Extract results into separate lists
    true_theta_list = [r[1] for r in results]
    GMM_list = [r[2] for r in results]

    # Create dataframe with results
    df = pd.DataFrame({'true_theta': true_theta_list,
                      'GMM': GMM_list})
    
    results_path = os.path.join(models_path,'point_estimators','GMM',f'GMM_results_seq_len_{seq_len}')
    output_filename = f'ACF_GMM_seq_len_{seq_len}_{num_lags}.pkl'


    df.to_pickle(os.path.join(results_path, output_filename))
 