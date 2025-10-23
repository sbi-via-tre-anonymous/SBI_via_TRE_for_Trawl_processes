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
from src.utils.chebyshev_utils import interpolation_points_domain, vec_polyfit_domain, vec_sample_from_coeff, sample_from_coeff, integrate_from_sampled, polyfit_domain, chebval_ab_for_one_x
from src.utils.chebyshev_utils import vec_chebval_ab_for_multiple_x_per_envelope_and_multple_envelopes as vec_chebval
import numpy as np
from tqdm import tqdm
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


def create_parameter_sweep_fn_for_2nd_acf_params(apply_fn, N):
    """
    Create a parameter sweep function with model function captured in closure.

    Args:
        apply_fn: Model application function (expects batch inputs), cached version
        bounds: Parameter bounds (min, max)
        N: Number of parameter values to evaluate

    Returns:
        A JIT-compiled function that takes (thetas, x_cache) and returns evaluations
    """
    # vary 2nd parameter, which has index 1
    param_idx = 1
    bounds = (10, 20)  # BOUNDS FOR 2ND ACF PARAMETER
    tre_type = 'acf'

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


# @partial(jax.jit, static_argnames=('nr_samples',))
@jax.jit
def estimate_first_density_p1(x_cache_to_use):

    thetas = jnp.zeros((N, 5))
    thetas = thetas.at[:, 0].set(
        interpolation_points_domain(N, bounds[0], bounds[1]))
    x_cache_to_use_expanded = jnp.broadcast_to(
        x_cache_to_use, (N, x_cached_shape))  # x_cache_size
    log_f_x_y = evaluate_at_chebyshev_knots(thetas, x_cache_to_use_expanded)

    return log_f_x_y


@partial(jax.jit, static_argnames=('nr_samples',))
def estimate_first_density_p2(f_x_y, key, true_theta, nr_samples):

    # f_x_y = jnp.exp(log_f_x_y)
    f_x = vec_integrate_from_sampled(f_x_y)
    cheb_coeff_f_x = polyfit_domain(f_x, bounds[0], bounds[1])

    key, subkey = jax.random.split(key)
    samples = sample_from_coeff(
        cheb_coeff_f_x, subkey, bounds[0], bounds[1],  nr_samples)

    prob_at_samples = chebval_ab_for_one_x(samples, cheb_coeff_f_x, a, b)

    prob_at_true_value = chebval_ab_for_one_x(
        true_theta[0], cheb_coeff_f_x, a, b)

    return samples, prob_at_samples, prob_at_true_value, key


def estimate_first_density(x_cache_to_use, key, true_theta, nr_samples):

    log_f_x_y = estimate_first_density_p1(x_cache_to_use)

    if calibration_type == 'beta':
        log_f_x_y = beta_calibrate_log_r(log_f_x_y,
                                         calibration_params['params'])
        f_x_y = jnp.exp(log_f_x_y)

    elif calibration_type == 'isotonic':

        # intermediary = iso_regression.predict(jax.nn.sigmoid(log_prob_envelope))
        intermediary = predict_2d(
            iso_regression, jax.nn.sigmoid(log_f_x_y))
        f_x_y = intermediary / (1-intermediary)

    else:
        f_x_y = jnp.exp(log_f_x_y)

    return estimate_first_density_p2(f_x_y, key, true_theta, nr_samples)
    # also return proability at true value and samples

    # return samples, prob_at_samples, prob_at_true_value, key


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


@jax.jit
def do_acf_sampling_p1(theta_first_component_to_use, x_cache_to_use):

    bounds = (10, 20)
    a, b = bounds

    thetas = jnp.zeros((nr_samples, 5)).at[:, 0].set(
        theta_first_component_to_use)
    x_cache_to_use_expanded = jnp.broadcast_to(
        x_cache_to_use, (nr_samples, x_cached_shape))  # x_cache_size

    log_prob_envelope = evaluate_at_chebyshev_knots(
        thetas, x_cache_to_use_expanded)

    return log_prob_envelope


@partial(jax.jit, static_argnames=('nr_samples'))
def do_acf_sampling_p2(prob_envelope, vec_key, nr_samples):
    bounds = (10, 20)
    a, b = bounds

    split_keys = jax.vmap(lambda k: jax.random.split(k, num=2))(vec_key)
    next_vec_key = split_keys[:, 0]  # Use first split from each key

    coeff = vec_polyfit_domain(prob_envelope, a, b)
    conditional_samples = vec_sample_from_coeff(
        coeff, vec_key, a, b, 1)

    # approximate density at posterior samples
    cond_prob_at_samples = vec_chebval(conditional_samples, coeff, a, b)
    # true_cond_prob = chebval_ab_for_one_x(true_theta[1],coeff,a,b)
    return conditional_samples, cond_prob_at_samples, next_vec_key


def do_acf_sampling(theta_first_component_to_use, x_cache_to_use, vec_key, nr_samples):

    # jitted
    log_prob_envelope = do_acf_sampling_p1(
        theta_first_component_to_use, x_cache_to_use)

    # apply calibration step, which is potetntially nonjitted
    if calibration_type == 'beta':
        log_prob_envelope = beta_calibrate_log_r(log_prob_envelope,
                                                 calibration_params['params'])
        prob_envelope = jnp.exp(log_prob_envelope)

    elif calibration_type == 'isotonic':

        # intermediary = iso_regression.predict(jax.nn.sigmoid(log_prob_envelope))
        intermediary = predict_2d(
            iso_regression, jax.nn.sigmoid(log_prob_envelope))
        prob_envelope = intermediary / (1-intermediary)

    else:
        prob_envelope = jnp.exp(log_prob_envelope)

    # jitted again
    return do_acf_sampling_p2(prob_envelope, vec_key, nr_samples)


# @jax.jit
def get_cond_prob_at_true_value(true_theta, x_cache_to_use):

    log_prob_envelope = evaluate_at_chebyshev_knots(
        true_theta[jnp.newaxis, :], x_cache_to_use[jnp.newaxis, :])

    if calibration_type == 'beta':
        log_prob_envelope = beta_calibrate_log_r(log_prob_envelope,
                                                 calibration_params['params'])
        prob_envelope = jnp.exp(log_prob_envelope)

    elif calibration_type == 'isotonic':

        # intermediary = iso_regression.predict(jax.nn.sigmoid(log_prob_envelope))
        intermediary = predict_2d(
            iso_regression, jax.nn.sigmoid(log_prob_envelope))
        prob_envelope = intermediary / (1-intermediary)

    else:
        prob_envelope = jnp.exp(log_prob_envelope)

    coeff = polyfit_domain(prob_envelope, a, b)
    cond_prob_at_true_sample = chebval_ab_for_one_x(true_theta[1], coeff, a, b)
    return cond_prob_at_true_sample


if __name__ == '__main__':

    tre_type = 'acf'
    trained_classifier_path = os.path.join(os.getcwd(),'models_and_simulated_datasets','classifiers','TRE_full_trawl','selected_models',tre_type)
    seq_len = 2000
    dummy_x = jnp.ones([1, seq_len])
    trawl_process_type = 'sup_ig_nig_5p'
    N = 128
    num_rows_to_load = 160  # nr data points is 64 * num_rows_to_load
    # num_envelopes_to_build_at_once = 5
    bounds = (10, 20)
    a, b = bounds
    # root_nr_samples = 100
    nr_samples = 10**4
    calibration_type = 'isotonic'

    # for sampling of the 1st component
    key = jax.random.PRNGKey(134234)#np.random.randint(1, 100000))
    # for sampling 2nd component, conditional on 1st componentt
    vec_key = jax.random.PRNGKey(9239)#np.random.randint(1, 100000))
    vec_key = jax.random.split(vec_key, nr_samples)

    # Create a partial function with fixed a, b

    def integrate_partial(samples):
        return integrate_from_sampled(samples, a=a, b=b)
    vec_integrate_from_sampled = jax.jit(jax.vmap(integrate_partial))

    rank_list = []

    # get calibratiton

    if calibration_type == 'beta':

        calibratiton_file_name = f'beta_calibration_{seq_len}.pkl'

        with open(os.path.join(trained_classifier_path, calibratiton_file_name), 'rb') as file:
            calibration_params = pickle.load(file)

    elif calibration_type == 'isotonic':
        with open(os.path.join(trained_classifier_path, f'fitted_iso_{seq_len}_{tre_type}.pkl'), 'rb') as file:
            iso_regression = pickle.load(file)

    assert tre_type in trained_classifier_path
    model, params, _, __bounds = load_one_tre_model_only_and_prior_and_bounds(
        trained_classifier_path, dummy_x, trawl_process_type, tre_type)

    # HARD CODED BOUNDS

    # LOAD DATA
    # Load dataset
    dataset_path = os.path.join(os.getcwd(), 'models',
                                'val_dataset', f'val_dataset_{seq_len}')
    val_x_path = os.path.join(dataset_path, 'val_x_joint.npy')
    val_thetas_path = os.path.join(dataset_path, 'val_thetas_joint.npy')

    # Load first few rows of val_x with memory mapping
    # more efficiently, we can pre-compute the val_x_cache to avoid recomputing it
    # each time we do this check
    # val_x = np.load(val_x_path, mmap_mode='r')[:num_rows_to_load]
    val_thetas = np.load(val_thetas_path)[:num_rows_to_load]

    # val_x = val_x.reshape(-1, seq_len)
    val_thetas = val_thetas.reshape(-1, val_thetas.shape[-1])

    val_x_cache_path = os.path.join(
        dataset_path, f'val_x_cache_{tre_type}_{val_thetas.shape[0]}.npy')
    assert os.path.exists(val_x_cache_path),'pre-compute x_cache first to reduce computational time'
    val_x_cache = jnp.load(val_x_cache_path)

    # LOAD FUNCTIONS
    apply_model_with_x, apply_model_with_x_cache = model_apply_wrapper(
        model, params)
    evaluate_at_chebyshev_knots = create_parameter_sweep_fn_for_2nd_acf_params(
        apply_model_with_x_cache,  N+1)

   # _, __ = apply_model_with_x(
   #     jnp.array(val_x[[0]]), jnp.array(val_thetas[[0]]))
    x_cached_shape = val_x_cache.shape[-1]

    # for i, (batch_thetas, batch_x) in enumerate(zip(theta_batches, x_batches)):
    for i, (theta_to_use, x_cache_to_use) in tqdm(enumerate(zip(val_thetas, val_x_cache)),
                                                  total=len(val_thetas),
                                                  desc="Processing"):

        # get x_cache
        # _, x_cache_to_use = apply_model_with_x(
        #    jnp.array(x_to_use.reshape(1, -1)), jnp.array(theta_to_use.reshape(1, -1)))

        # sample 1st component of the acf
        samples_1_comp, prob_at_1_comp, prob_at_1_true, key = estimate_first_density(
            x_cache_to_use, key, theta_to_use, nr_samples)
        # break it into batches to make avoid memory issues
        # samples_1_comp_batches = np.array_split(
        #    samples_1_comp, nr_samples)

        # samples_2_comp = []
        # prob_at_2_comp = []

        # for theta_first_component_to_use in samples_1_comp_batches:

        conditional_samples, cond_prob_at_samples, vec_key = do_acf_sampling(samples_1_comp,
                                                                             x_cache_to_use, vec_key,  nr_samples)

        samples_2_comp = conditional_samples
        prob_at_2_comp = cond_prob_at_samples


        prob_at_2_true = get_cond_prob_at_true_value(
            jnp.array(theta_to_use), x_cache_to_use)
        # to compute the true conditional probability, multiply them 

        true_probability = prob_at_1_true * prob_at_2_true
        prob_at_samples = prob_at_1_comp * prob_at_2_comp.squeeze()
        rank_list.append(np.mean(true_probability <
                                 prob_at_samples).item())

    results_path = os.path.join(os.getcwd(), 'models_and_simulated_datasets','classifiers', 
                                'per_classifier_coverage_check', str(tre_type))
    os.makedirs(results_path, exist_ok=True)
    
    if calibration_type in ['isotonic', 'beta']:
        file_path = f'{tre_type}_cal_ranks_cal_type_{calibration_type}_seq_len_{seq_len}_N_{N}.npy'
    else:
        file_path = f'{tre_type}_uncal_ranks_seq_len_{seq_len}_N_{N}.npy'

    np.save(file=os.path.join(results_path, file_path), arr=rank_list)
