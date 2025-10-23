# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from functools import partial
from src.utils.get_model import get_model
from src.utils.get_trained_models import load_one_tre_model_only_and_prior_and_bounds
from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.reconstruct_beta_calibration import beta_calibrate_log_r
import os
import pickle
from tqdm import tqdm
import numpy as np
from src.utils.reconstruct_beta_calibration import beta_calibrate_log_r
from src.utils.chebyshev_utils import chebint_ab, interpolation_points_domain, integrate_from_sampled, polyfit_domain,  \
    vec_polyfit_domain, sample_from_coeff, chebval_ab_jax, vec_sample_from_coeff,\
    chebval_ab_for_one_x, vec_chebval_ab_for_multiple_x_per_envelope_and_multple_envelopes,\
    vec_integrate_from_samples


def create_parameter_sweep_fn(tre_type, apply_fn_dict, bounds_dict, N):
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
    param_idx = {'beta': -1, 'sigma': -2, 'mu': -3, 'acf': -4}[tre_type]
    apply_fn = apply_fn_dict[tre_type]
    bounds = bounds_dict[tre_type]

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


def apply_calibration(log_r, tre_type, calibration_type, calibration_dict):
    """inputs log_r, outputs exponential of the calibrated classifier"""

    if calibration_type == 'None':

        return jnp.exp(log_r)

    elif calibration_type == 'beta':

        log_r = beta_calibrate_log_r(
            log_r, calibration_dict[tre_type])
        return jnp.exp(log_r)

    elif calibration_type == 'isotonic':

        # exp(logit(p)) = p / (1-p)
        # exp( logit( iso( sigma( )))) = iso( sigma( )) / (1 - iso( sigma( )))
        intermediary = predict_2d(calibration_dict[tre_type],
                                  jax.nn.sigmoid(log_r))
        return intermediary / (1-intermediary)

def estimate_first_density_enclosure(tre_type, parameter_sweeps_dict, bounds_dict, N):
    
    @jax.jit  # @partial(jax.jit, static_argnames=('tre_type',))
    def estimate_first_density(x_cache_to_use):
    
        tre_type = 'acf'
        evaluate_at_chebyshev_knots = parameter_sweeps_dict[tre_type]
        bounds = bounds_dict[tre_type]
        x_cached_shape = x_cache_to_use.shape[-1]
    
        thetas = jnp.zeros((N, 5))
        thetas = thetas.at[:, 0].set(
            interpolation_points_domain(N, bounds[0], bounds[1]))
        x_cache_to_use_expanded = jnp.broadcast_to(
            x_cache_to_use, (N, x_cached_shape))  # x_cache_size
    
        log_f_x_y = evaluate_at_chebyshev_knots(thetas, x_cache_to_use_expanded)
    
        return log_f_x_y
    
    return estimate_first_density


# @partial(jax.jit, static_argnames=('tre_type',))
def get_cond_prob_at_true_value(true_theta, x_cache_to_use, tre_type, parameter_sweeps_dict, bounds_dict, calibration_type, calibration_dict):

    a, b = bounds_dict[tre_type][0], bounds_dict[tre_type][1]
    param_idx = {'beta': -1, 'sigma': -2, 'mu': -3, 'acf': -4}[tre_type]

    log_prob_at_cheb_knots = parameter_sweeps_dict[tre_type](
        true_theta, x_cache_to_use[jnp.newaxis, :])

    prob_at_cheb_knots = apply_calibration(
        log_prob_at_cheb_knots, tre_type, calibration_type,calibration_dict) 

    coeff = polyfit_domain(prob_at_cheb_knots, a, b)
    true_conditional_prob = chebval_ab_for_one_x(
        true_theta[0, param_idx], coeff, a, b)
    normalizing_constant = integrate_from_sampled(prob_at_cheb_knots, a, b)
    return true_conditional_prob / normalizing_constant


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import seaborn as sns

    tre_types_list = ['acf', 'mu', 'sigma', 'beta']
    seq_len = 1000
    trawl_process_type = 'sup_ig_nig_5p'
    N = 128
    num_samples = 10**4
    num_rows_to_load = 160  # nr data points is 64 * num_rows_to_load
    batch_size_for_evaluating_x_cache = 64
    key = jax.random.PRNGKey(3525)#np.random.randint(1, 100000)
    vec_key = jax.random.PRNGKey(23434)#np.random.randint(1, 100000)
    vec_key = jax.random.split(vec_key, num_samples)

    dummy_x = jnp.ones([1, seq_len])
    calibration_type = 'beta'

    assert calibration_type in ('None', 'beta', 'isotonic')

    ### load data ###
    dataset_path = os.path.join(os.getcwd(), 'models_and_simulated_datasets',
                                'validation_datasets', f'val_dataset_{seq_len}')
    val_x_path = os.path.join(dataset_path, 'val_x_joint.npy')
    val_thetas_path = os.path.join(dataset_path, 'val_thetas_joint.npy')

    # Load first few rows of val_x with memory mapping
    val_x = np.load(val_x_path, mmap_mode='r')[:num_rows_to_load]
    val_thetas = np.load(val_thetas_path)[:num_rows_to_load]

    val_x = val_x.reshape(-1, seq_len)
    val_thetas = val_thetas.reshape(-1, val_thetas.shape[-1])

    ### load models and precompute x_cache ###
    models_dict = dict()
    apply_fn_dict = dict()
    appl_fn_to_get_x_cache_dict = dict()
    parameter_sweeps_dict = dict()
    #if calibration_type == 'beta':
    #    beta_calibration_params = dict()
#
    #elif calibration_type == 'isotonic':
    #    iso_calibration_dict = dict()
    calibration_dict = dict()

    bounds_dict = {'acf': [10., 20.], 'beta': [-5., 5.],
                   'mu': [-1., 1.], 'sigma': [0.5, 1.5]}
    


    for tre_type in tre_types_list:  

        trained_classifier_path = os.path.join(os.getcwd(), 'sbi_via_tre_for_trawl_processes','models_and_simulated_datasets','classifiers','TRE_full_trawl','selected_models',tre_type)
        model, params, _, __bounds = load_one_tre_model_only_and_prior_and_bounds(
            trained_classifier_path, dummy_x, trawl_process_type, tre_type)

        # load model
        models_dict[tre_type] = model

        # load apply_fn
        apply_fn_to_get_x_cache, apply_fn = model_apply_wrapper(model, params)
        apply_fn_dict[tre_type] = apply_fn
        appl_fn_to_get_x_cache_dict[tre_type] = apply_fn_to_get_x_cache

        # load calibratitons params
        if calibration_type == 'beta':
            with open(os.path.join(trained_classifier_path, f'beta_calibration_{seq_len}.pkl'), 'rb') as file:
                #beta_calibration_params[tre_type] = pickle.load(file)['params']
                calibration_dict[tre_type] = pickle.load(file)['params']

        elif calibration_type == 'isotonic':
            # Load the model
            with open(os.path.join(trained_classifier_path, f'fitted_iso_{seq_len}.pkl'), 'rb') as file:#f'fitted_iso_{seq_len}_{tre_type}.pkl'
                #iso_calibration_dict[tre_type] = pickle.load(file)
                calibration_dict[tre_type] = pickle.load(file)


        # create parameter sweeps here:
        parameter_sweeps_dict[tre_type] = create_parameter_sweep_fn(
            tre_type, apply_fn_dict, bounds_dict, N+1)  # +1 just to make sure we do not confuse the first two acf components

        # Load x_cache or precmpue it and save it
        val_x_cache_path = os.path.join(
            dataset_path, f'val_x_cache_{tre_type}_{val_x.shape[0]}.npy')
        if os.path.exists(val_x_cache_path):
            val_x_cache = jnp.load(val_x_cache_path)

        else:

            num_batches = val_x.shape[0] // batch_size_for_evaluating_x_cache
            assert val_x.shape[0] % num_batches == 0, 'total number of datapoints is not divisible by the batch size'
            theta_batches = np.array_split(val_thetas, num_batches)
            x_batches = np.array_split(val_x, num_batches)
            val_x_cache_list = []

            for theta_batch, x_batch in zip(theta_batches, x_batches):
                _, x_cache_to_append = apply_fn_to_get_x_cache(
                    x_batch, theta_batch)
                val_x_cache_list.append(x_cache_to_append)

            val_x_cache_list = jnp.concatenate(val_x_cache_list)
            np.save(file=val_x_cache_path, arr=val_x_cache_list)
            del val_x_cache_list

    # load x_cache for each tre_type
    val_x_cache_dict = dict()
    for tre_type in tre_types_list:
        val_x_cache_dict[tre_type] = jnp.load(os.path.join(
            dataset_path, f'val_x_cache_{tre_type}_{val_x.shape[0]}.npy'))

    # to use with acf  ### ignore for now

    def acf_integrate_partial_enclosure(bounds_dict):
        
        def acf_integrate_partial(samples):
            return integrate_from_sampled(samples, a=bounds_dict['acf'][0], b=bounds_dict['acf'][1])
        vec_integrate_2nd_component_acf_from_sampled = jax.jit(
            jax.vmap(acf_integrate_partial))
        
        return acf_integrate_partial, vec_integrate_2nd_component_acf_from_sampled
    
    #added enclosures so i can call these functions from a different script, for the application
    estimate_first_density = estimate_first_density_enclosure(tre_type, parameter_sweeps_dict, bounds_dict, N)
    acf_integrate_partial, vec_integrate_2nd_component_acf_from_sampled  = acf_integrate_partial_enclosure(bounds_dict)

    #########################

    # do posterior sampling one by one
    # true_theta, true_x in zip(val_thetas, val_x):

    rank_list = []

    for i in tqdm(range(val_thetas.shape[0])):

        true_theta = jnp.array(val_thetas[i])[jnp.newaxis, :]

        # ACF sampling
        tre_type = 'acf'
        true_x_cache = val_x_cache_dict[tre_type][i]

        # get 2d log probabilities on a 2d cheb gridi
        two_d_log_prob = estimate_first_density(true_x_cache)
        # calibrate 2d grid probabilities
        two_d_prob = apply_calibration(
            two_d_log_prob, tre_type, calibration_type, calibration_dict) 
        # get 1d prob for the first component
        f_x = vec_integrate_2nd_component_acf_from_sampled(
            two_d_prob)  # vec_integrate_from_sampled(two_d_prob)
        # get 1d coeff
        cheb_coeff_f_x = polyfit_domain(
            f_x, bounds_dict[tre_type][0], bounds_dict[tre_type][1])
        # sample from 1st dimension
        key, subkey = jax.random.split(key)
        first_comp_samples = sample_from_coeff(
            cheb_coeff_f_x, subkey, bounds_dict[tre_type][0], bounds_dict[tre_type][1], num_samples)
        normalizing_constant_acf = integrate_from_sampled(
            f_x, bounds_dict[tre_type][0], bounds_dict[tre_type][1])
        first_comp_densities = chebval_ab_jax(first_comp_samples, cheb_coeff_f_x,
                                              bounds_dict[tre_type][0],
                                              bounds_dict[tre_type][1]) / normalizing_constant_acf

        true_density = chebval_ab_for_one_x(true_theta[0, 0], cheb_coeff_f_x,
                                            bounds_dict[tre_type][0],
                                            bounds_dict[tre_type][1]) / normalizing_constant_acf
        # sampling of first component finished
        thetas_ = jnp.zeros([num_samples, 5])
        col_index = 0
        thetas_ = thetas_.at[:, col_index].set(first_comp_samples)
        sample_densities = jnp.copy(first_comp_densities)
        del true_x_cache

        # sequentially sample starting the 2nd acf component, then mu sigma beta
        for tre_type in tre_types_list:

            x_cache_to_use = val_x_cache_dict[tre_type][i]
            x_cache_to_use_expanded = jnp.broadcast_to(
                x_cache_to_use, (num_samples, x_cache_to_use.shape[-1]))
            log_conditional_prob_at_cheb_knots = parameter_sweeps_dict[tre_type](
                thetas_, x_cache_to_use_expanded)

            conditional_prob_at_cheb_knots = apply_calibration(
                log_conditional_prob_at_cheb_knots, tre_type, calibration_type, calibration_dict) 

            conditional_density_cheb_coeff = vec_polyfit_domain(
                conditional_prob_at_cheb_knots, bounds_dict[tre_type][0], bounds_dict[tre_type][1])

            split_keys = jax.vmap(
                lambda k: jax.random.split(k, num=2))(vec_key)
            last_component_samples = vec_sample_from_coeff(
                conditional_density_cheb_coeff, vec_key, bounds_dict[tre_type][0], bounds_dict[tre_type][1], 1)
            vec_key = split_keys[:, 0]  # Use first split from each key

            normalizing_constants = vec_integrate_from_samples(
                conditional_prob_at_cheb_knots, bounds_dict[tre_type][0], bounds_dict[tre_type][1])
            conditional_prob = vec_chebval_ab_for_multiple_x_per_envelope_and_multple_envelopes(last_component_samples,
                                                                                                conditional_density_cheb_coeff,
                                                                                                bounds_dict[tre_type][0], bounds_dict[tre_type][1]).squeeze() / normalizing_constants
            # get true conditional probability
            true_conditional_density = get_cond_prob_at_true_value(
                true_theta, x_cache_to_use, tre_type, parameter_sweeps_dict, bounds_dict, calibration_type, calibration_dict)

            sample_densities *= conditional_prob
            true_density *= true_conditional_density

            col_index += 1
            thetas_ = thetas_.at[:, col_index].set(
                last_component_samples.squeeze())

        rank_to_add = np.mean(true_density < sample_densities)
        rank_list.append(rank_to_add.item())

        del rank_to_add
        del true_density
        del sample_densities
        del conditional_prob

    save_path = os.path.join(os.getcwd(), 'models_and_simulated_datasets',
                             'classifiers', 'coverage_check_ranks_NRE_and_TRE')

    np.save(file=os.path.join(
        save_path, f'seq_sampling_TRE_{seq_len}_{calibration_type}_{N}_{num_rows_to_load}.npy'), arr=rank_list)
