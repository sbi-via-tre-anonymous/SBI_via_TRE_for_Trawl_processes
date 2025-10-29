from jax.scipy.special import logit
from jax.nn import sigmoid
import pandas as pd

import jax.numpy as jnp
from jax.random import PRNGKey
from functools import partial
from src.utils.get_model import get_model
from src.utils.get_trained_models import load_one_tre_model_only_and_prior_and_bounds
from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.reconstruct_beta_calibration import beta_calibrate_log_r
from src.model.Extended_model_nn import ExtendedModel, VariableExtendedModel
from src.utils.chebyshev_utils import interpolation_points_domain, vec_polyfit_domain, vec_sample_from_coeff
from src.utils.chebyshev_utils import vec_chebval_ab_for_multiple_x_per_envelope_and_multple_envelopes as vec_chebval
import numpy as np
import datetime
import pickle
import yaml
import jax
import os
import matplotlib
# matplotlib.use('Agg')  # Non-interactive backend
if True:
    from path_setup import setup_sys_path
    setup_sys_path()
    import matplotlib.pyplot as plt


def predict_2d(iso_reg, X):
    """
    Wrapper to handle any shape input for isotonic regression.

    Parameters:
    -----------
    iso_reg : IsotonicRegression
        Fitted isotonic regression model
    X : np.ndarray or jnp.ndarray
        Input array of any shape

    Returns:
    --------
    np.ndarray : Predictions with same shape as input
    """
    original_shape = X.shape
    X_flat = np.array(X.ravel())
    y_pred = iso_reg.predict(X_flat)
    return jnp.array(y_pred.reshape(original_shape))


def create_parameter_sweep_fn(apply_fn, tre_type, bounds, N):
    """
    Create a parameter sweep function with model function captured in closure.

    Args:
        apply_fn: Model application function (expects batch inputs), cached version
        tre_type: Parameter to vary ('beta', 'sigma', 'mu')
        bounds: Parameter bounds (min, max)
        N: Number of parameter values to evaluate

    Returns:
        A JIT-compiled function that takes (thetas, x_cache) and returns evaluations
    """
    # Determine which parameter to vary
    param_idx = {'beta': -1, 'sigma': -2, 'mu': -3}[tre_type]

    # Parameter values to evaluate (computed once)
    param_values = interpolation_points_domain(N, bounds[0], bounds[1])

    # Define the inner processing function with apply_fn in closure
    def process_param(p_val, thetas, x_cache):
        batch_size = thetas.shape[0]
        modified = thetas.at[:, param_idx].set(jnp.full(batch_size, p_val))
        results, _ = apply_fn(modified, x_cache)
        return results

    # Create the vectorized version (done once)
    vectorized_process = jax.vmap(process_param, in_axes=(0, None, None))

    # Define and return the JIT-compiled sweep function
    @jax.jit  # No static_argnums needed
    def parameter_sweep(thetas, x_cache):
        """
        Evaluate parameter sweep across the batch.

        Args:
            thetas: Batch of thetas [batch_size, param_dim]
            x_cache: Cached representation

        Returns:
            Evaluations with shape [batch_size, N]
        """
        all_results = vectorized_process(param_values, thetas, x_cache)
        return jnp.transpose(all_results).squeeze()

    return parameter_sweep


def model_apply_wrapper(model, params):

    # Define JIT-ed apply functions

    @jax.jit
    def apply_model_with_x(x, theta):
        """Apply model with a new x input, returning output and x_cache."""
        return model.apply(params, x, theta)

    @jax.jit
    def apply_model_with_x_cache(theta, x_cache):
        """Apply model with cached x representation, returning output and updated x_cache."""
        return model.apply(params, None, theta, x_cache=x_cache)

    return apply_model_with_x, apply_model_with_x_cache


def do_marginal_sampling(theta_values_batch, x_values_batch, vec_key, bounds, tre_type):
#def do_marginal_sampling(theta_values_batch, x_cache_values_batch, vec_key, bounds, tre_type):

    a, b = bounds


    _, x_cache_values_batch = apply_model_with_x(
        x_values_batch, theta_values_batch)

    # technically should multply by prior,but it s all a constant
    log_prob_envelope = evaluate_at_chebyshev_knots(
        theta_values_batch, x_cache_values_batch)
    uncal_prob_envelope = jnp.exp(log_prob_envelope)

    if calibration_type == 'beta':
        cal_log_prob_envelope = beta_calibrate_log_r(
            log_prob_envelope, calibration_params['params'])
        cal_prob_envelope = jnp.exp(cal_log_prob_envelope)
        del cal_log_prob_envelope

    elif calibration_type == 'isotonic':
        # intermediary = iso_regression.predict(jax.nn.sigmoid(log_prob_envelope))
        intermediary = predict_2d(
            iso_regression, jax.nn.sigmoid(log_prob_envelope))
        cal_prob_envelope = intermediary / (1-intermediary)

    cal_coeff = vec_polyfit_domain(cal_prob_envelope, a, b)
    uncal_coeff = vec_polyfit_domain(uncal_prob_envelope, a, b)

    cal_samples = vec_sample_from_coeff(cal_coeff, vec_key, a, b, num_samples)
    uncal_samples = vec_sample_from_coeff(
        uncal_coeff, vec_key, a, b, num_samples)

    # approximate density at posterior samples
    cal_prob_at_cal_samples = vec_chebval(cal_samples, cal_coeff, a, b)
    uncal_prob_at_uncal_samples = vec_chebval(uncal_samples, uncal_coeff, a, b)

    # approximate density at true value
    param_idx = {'beta': -1, 'sigma': -2, 'mu': -3}[tre_type]
    true_values = theta_values_batch[:, [param_idx]]

    cal_prob_at_true_value = vec_chebval(true_values, cal_coeff, a, b)
    uncal_prob_at_true_value = vec_chebval(true_values, uncal_coeff, a, b)

    cal_rank = np.mean(cal_prob_at_true_value <
                       cal_prob_at_cal_samples, axis=1)
    uncal_rank = np.mean(uncal_prob_at_true_value <
                         uncal_prob_at_uncal_samples, axis=1)

    split_keys = jax.vmap(lambda k: jax.random.split(k, num=2))(vec_key)
    next_vec_key = split_keys[:, 0]  # Use first split from each key

    return cal_rank, uncal_rank, next_vec_key


if __name__ == '__main__':

    tre_type = 'sigma'
    trained_classifier_path = os.path.join(os.getcwd(),'models_and_simulated_datasets', 'classifiers', 'TRE_full_trawl','selected_models',tre_type)  
    seq_len = 1500
    dummy_x = jnp.ones([1, seq_len])
    trawl_process_type = 'sup_ig_nig_5p'
    N = 128
    num_samples = 10**5 # set to 10**4 on CPU for fast computations
    num_rows_to_load = 160  # nr data points is 64 * num_rows_to_load
    num_envelopes_to_build_at_once = 64 # set to 8 or 16 on a CPU
    calibration_type = 'beta'
    # assert beta_calibration_indicator

    # get calibratiton

    if calibration_type == 'beta':

        calibratiton_file_name = f'beta_calibration_{seq_len}.pkl'

        with open(os.path.join(trained_classifier_path, calibratiton_file_name), 'rb') as file:
            calibration_params = pickle.load(file)

    elif calibration_type == 'isotonic':
        with open(os.path.join(trained_classifier_path,f'fitted_iso_{seq_len}.pkl'), 'rb') as file:
                               #f'fitted_iso_{seq_len}_{tre_type}.pkl'), 'rb') as file:
            iso_regression = pickle.load(file)

    assert tre_type in trained_classifier_path
    model, params, _, bounds = load_one_tre_model_only_and_prior_and_bounds(
        trained_classifier_path, dummy_x, trawl_process_type, tre_type)

    # LOAD DATA
    # Load dataset
    dataset_path = os.path.join(os.getcwd(), 'models_and_simulated_datasets',
                                'validation_datasets', f'val_dataset_{seq_len}')
    val_x_path = os.path.join(dataset_path, 'val_x_joint.npy')
    val_thetas_path = os.path.join(dataset_path, 'val_thetas_joint.npy')

    # Load first few rows of val_x with memory mapping
    val_x = np.load(val_x_path, mmap_mode='r')[:num_rows_to_load]
    val_thetas = np.load(val_thetas_path)[:num_rows_to_load]

    val_x = val_x.reshape(-1, seq_len)
    val_thetas = val_thetas.reshape(-1, val_thetas.shape[-1])


    # LOAD FUNCTIONS
    apply_model_with_x, apply_model_with_x_cache = model_apply_wrapper(
        model, params)
    evaluate_at_chebyshev_knots = create_parameter_sweep_fn(
        apply_model_with_x_cache, tre_type, bounds, N+1)

    # TEST
    num_batches = len(val_thetas) // num_envelopes_to_build_at_once + \
        (1 if len(val_thetas) % num_envelopes_to_build_at_once else 0)

    theta_batches = np.array_split(val_thetas, num_batches)
    x_batches = np.array_split(val_x, num_batches)

    key = jax.random.PRNGKey(3423424)#np.random.randint(1, 100000))
    key = jax.random.split(key, num_envelopes_to_build_at_once)

    cal_rank_list = []
    uncal_rank_list = []
    
    from tqdm import tqdm
    total_batches = len(theta_batches)

    for i, (batch_thetas, batch_x) in enumerate(tqdm(zip(theta_batches, x_batches), 
                                                  total=total_batches, 
                                                  desc="Processing batches")):

        cal_rank, uncal_rank, key = do_marginal_sampling(
            batch_thetas, batch_x, key, bounds, tre_type)
        cal_rank_list.append(cal_rank)
        uncal_rank_list.append(uncal_rank)

    cal_rank = np.concatenate(cal_rank_list)
    uncal_rank = np.concatenate(uncal_rank_list)

    results_path = os.path.join(os.getcwd(), 'models_and_simulated_datasets', 'classifiers', 'TRE_full_trawl',
                                'selected_models', 'per_classifier_coverage_check', str(tre_type))
    os.makedirs(results_path, exist_ok=True)


    np.save(file=os.path.join(results_path,
            f'{tre_type}_cal_ranks_cal_type_{calibration_type}_seq_len_{seq_len}_N_{N}.npy'), arr=cal_rank)
    np.save(file=os.path.join(results_path,
            f'{tre_type}_uncal_ranks_seq_len_{seq_len}_N_{N}.npy'), arr=uncal_rank)

    # kolmogorov_smirnov_uniform(cal_rank), kolmogorov_smirnov_uniform(uncal_rank)
    # cramer_von_mises_uniform(cal_rank), cramer_von_mises_uniform(uncal_rank)
    # anderson_darling_uniform(cal_rank), anderson_darling_uniform(uncal_rank)
