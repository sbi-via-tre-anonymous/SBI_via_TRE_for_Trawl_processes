import time
import os
import yaml
import pickle
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import numpy as np
import distrax
from jax.nn import sigmoid
from jax.scipy.special import logit
if True:
    from path_setup import setup_sys_path
    setup_sys_path()

from src.utils.reconstruct_beta_calibration import beta_calibrate_log_r
from statsmodels.tsa.stattools import acf as compute_empirical_acf
from src.utils.classifier_utils import get_projection_function
from src.model.Extended_model_nn import ExtendedModel, VariableExtendedModel
from src.utils.get_model import get_model
from src.utils.get_data_generator import get_theta_and_trawl_generator


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


###############################################################################
###################### LOAD TRAINED MODELS FOR INFERENCE ######################
###################### POSTERIOR SAMPLING AND CHECKS ##########################
###############################################################################

def load_one_tre_model_only_and_prior_and_bounds(folder_path, dummy_x, trawl_process_type, tre_type):

    assert dummy_x.ndim == 2  # should actually be a trawl
    if trawl_process_type == 'sup_ig_nig_5p':
        dummy_theta = jnp.ones((dummy_x.shape[0], 5))
    else:
        raise ValueError

    with open(os.path.join(folder_path, 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
        use_tre = config['tre_config']['use_tre']
        use_summary_statistics = config['tre_config']['use_summary_statistics']

    # sanity checks !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # assert use_tre == True #used to be true before NRE calibration; check the if statement before looading the model as well!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    assert use_summary_statistics == False
    assert config['trawl_config']['trawl_process_type'] == trawl_process_type

    params_path = [f for f in os.listdir(
        folder_path) if f.startswith("params") and f.endswith(".pkl")]
    assert len(params_path) == 1
    # Then load params, config and model
    with open(os.path.join(folder_path, params_path[0]), 'rb') as file:
        params = pickle.load(file)

    model = get_model(config, False)

    if use_tre:  # use_tre:

        model = VariableExtendedModel(base_model=model, trawl_process_type=trawl_process_type,
                                      tre_type=tre_type,
                                      use_summary_statistics=use_summary_statistics)

    _ = model.init(PRNGKey(0), dummy_x, dummy_theta)

    ### get the prior ###
    acf_prior_hyperparams = config['trawl_config']['acf_prior_hyperparams']
    eta_bounds = acf_prior_hyperparams['eta_prior_hyperparams']
    gamma_bounds = acf_prior_hyperparams['gamma_prior_hyperparams']

    marginal_distr_hyperparams = config['trawl_config']['marginal_distr_hyperparams']
    mu_bounds = marginal_distr_hyperparams['loc_prior_hyperparams']
    scale_bounds = marginal_distr_hyperparams['scale_prior_hyperparams']
    beta_bounds = marginal_distr_hyperparams['beta_prior_hyperparams']
    # gamma, eta, mu, scale , beta
    bounds = (gamma_bounds, eta_bounds, mu_bounds, scale_bounds, beta_bounds)
    lower_bounds = jnp.array([i[0] for i in bounds])
    upper_bounds = jnp.array([i[1] for i in bounds])
    priors = 1 / (upper_bounds - lower_bounds)

    if not use_tre:

        print('Loading function load_one_tre_model_only_and_prior_and_bounds with an NRE. this is only supposed to be allowed\
              during the calibration f the NRE. might be ok even othterwise, just have not checked it.')
        return model, params, priors, bounds

    if tre_type == 'acf':
        prior = priors[0] * priors[1]
        bounds_to_return = (gamma_bounds, eta_bounds)
    elif tre_type == 'mu':
        prior = priors[2]
        bounds_to_return = mu_bounds
    elif tre_type == 'sigma':
        prior = priors[3]
        bounds_to_return = scale_bounds
    elif tre_type == 'beta':
        prior = priors[4]
        bounds_to_return = beta_bounds
    else:
        raise ValueError

    return model, params, prior, jnp.array(bounds_to_return)


# load TRE component
# presample
# define function to evaluate it at certain values based on the sample values
# do calibration
# perhaps finetune


def load_trained_models_for_posterior_inference(folder_path, dummy_x, trawl_process_type,
                                                use_tre, use_summary_statistics, calibration_file_name):
    """
    Loads the trained models, may it be NRE or TRE, and returns two functions,
    which approximate the log likelihood and log posterior. There are multiple
    possibilities: NRE / TRE, and within TRE, we may chose to use the full trawls
    or the summary statistics. If we use the full trawls, the TRE ACF uses
    the empirical acf as input, which complicates the loading of the trained models.

    In the training script, we passed the correct x to the model full trawl,
    summary_statistics or empirical_acf. We do the same here. to this end, we 
    first do some sanity checks and then load all models. we have an indicator
    use_empirical_acf. if set to true, we modify the input to the TRE ACF model.
    """
    
    #We proceed this way because in MCMC / posterior inference / checks, we
    #will use this for a fixed value of x, and i it's inefficient to recompute the 
    #empirical acf each time, and also returning a list of models clutters the
    #posterior inference script.

    assert dummy_x.ndim == 2  # should actually be a traw
    if trawl_process_type == 'sup_ig_nig_5p':
        dummy_theta = jnp.ones((dummy_x.shape[0], 5))
    else:
        raise ValueError

    if use_summary_statistics:

        project_trawl = get_projection_function()
        dummy_x = project_trawl(dummy_x)

    # if using tre, check there are exactly 4 subfolders, each with exactly one
    # set of params
    if use_tre:

        # List all items in the directory and filter only folders
        folders = [f for f in os.listdir(folder_path) if os.path.isdir(
            os.path.join(folder_path, f))]
        assert set(['acf', 'beta', 'mu', 'sigma']).issubset(folders)
        folders = ['acf', 'beta', 'mu', 'sigma']

        folders = [os.path.join(folder_path, f) for f in folders]

    else:
        folders = [folder_path]

    config_list = []
    calibration_list = []
    params_list = []
    model_list = []

    for i in range(len(folders)):
        folder = folders[i]

        params_path = [f for f in os.listdir(
            folder) if f.startswith("params") and f.endswith(".pkl")]
        assert len(params_path) == 1

        # Then load params, config and model
        with open(os.path.join(folder, params_path[0]), 'rb') as file:
            params_list.append(pickle.load(file))

        if 'spline' in calibration_file_name:
            assert calibration_file_name[-4:] == '.npy'
            calibration_list.append(
                jnp.load(os.path.join(folder, calibration_file_name)))

            spline_cal = True

        elif ('beta' in calibration_file_name) or ('no_calibration' in calibration_file_name):
            assert calibration_file_name[-4:] == '.pkl'
            with open(os.path.join(folder, calibration_file_name), 'rb') as file:
                calibration_list.append(pickle.load(file))

            spline_cal = False
        else:
            print(calibration_file_name)
            raise ValueError('wrong calibration_name')

        with open(os.path.join(folder, 'config.yaml'), 'r') as file:
            config_list.append(yaml.safe_load(file))
            # sanity check
            assert use_tre == config_list[-1]['tre_config']['use_tre']
            assert use_summary_statistics == config_list[-1]['tre_config']['use_summary_statistics']
            assert trawl_process_type == config_list[-1]['trawl_config']['trawl_process_type']

        model_ = get_model(config_list[-1], False)

        if use_tre:
            model_ = VariableExtendedModel(base_model=model_, trawl_process_type=trawl_process_type,
                                           tre_type=config_list[-1]['tre_config']['tre_type'],
                                           use_summary_statistics=use_summary_statistics)

        use_empirical_acf = False
        # Initialize
        if i == 0 and use_tre:
            acf_config = config_list[-1]
            acf_tre_config = acf_config['tre_config']
            assert folder[-3:] == 'acf'
            # can assume we are doing acf
            assert acf_tre_config['tre_type'] == 'acf'
            n_lags = acf_tre_config['nlags']

            if use_tre and (not use_summary_statistics) and acf_tre_config['replace_full_trawl_with_acf'] and (acf_tre_config['tre_type'] == 'acf'):

                use_empirical_acf = True

                # empirical_acf_x = jnp.array(
                #    compute_empirical_acf(np.array(dummy_x[0]), nlags=n_lags)[1:])[jnp.newaxis, :]
                empirical_dummy_x = jnp.ones((dummy_x.shape[0], n_lags))

                _ = model_.init(PRNGKey(0), empirical_dummy_x, dummy_theta)


            else:
                _ = model_.init(PRNGKey(0), dummy_x, dummy_theta)

        #############
        model_list.append(model_)
        #############

    ################ get bounds, assuming they're all the same ################
    acf_prior_hyperparams = config_list[-1]['trawl_config']['acf_prior_hyperparams']
    eta_bounds = acf_prior_hyperparams['eta_prior_hyperparams']
    gamma_bounds = acf_prior_hyperparams['gamma_prior_hyperparams']

    marginal_distr_hyperparams = config_list[-1]['trawl_config']['marginal_distr_hyperparams']
    mu_bounds = marginal_distr_hyperparams['loc_prior_hyperparams']
    scale_bounds = marginal_distr_hyperparams['scale_prior_hyperparams']
    beta_bounds = marginal_distr_hyperparams['beta_prior_hyperparams']
    # gamma, eta, mu, scale , beta
    bounds = (gamma_bounds, eta_bounds, mu_bounds, scale_bounds, beta_bounds)
    lower_bounds = jnp.array([i[0] for i in bounds])
    upper_bounds = jnp.array([i[1] for i in bounds])
    total_mass = jnp.prod(upper_bounds - lower_bounds)

    ###################### get functions ######################################

    # Create a "no-op" calibration parameter set for when no calibration is needed
    no_op_cal_params = (1.0, 1.0, 0.5)

    # Pre-process and convert all the model components to JAX-friendly structures
    model_applies = [model.apply for model in model_list]
    model_params = [params for params in params_list]

    # Process calibration parameters
    if spline_cal:

        calibration_params = calibration_list

    else:

        calibration_params = []
        for c in calibration_list:

            if c['use_beta_calibration']:
                calibration_params.append(
                    (c['params'][0], c['params'][1], c['params'][2]))
            else:
                calibration_params.append(no_op_cal_params)


    @jax.jit
    def approximate_log_likelihood_to_evidence(x, theta):
        total_log_r = 0.0

        print('here')

        for i in range(len(model_applies)):
            apply_fn = model_applies[i]
            params = model_params[i]

            # Apply model
            log_r, _ = apply_fn(variables=params, x=x,
                                theta=theta, train=False)

            if spline_cal:

                spline = distrax.RationalQuadraticSpline(boundary_slopes='identity',  # can be changed
                                                         params=calibration_params[i], range_min=0.0, range_max=1.0)

                log_r = logit(spline.forward(sigmoid(log_r)))

            else:
                # Always apply calibration - will be no-op for models that don't need it
                log_r = beta_calibrate_log_r(log_r, calibration_params[i])

            total_log_r += log_r

        return total_log_r

    def wrapper_for_cached_approximate_log_likelihood_to_evidence(x):

        x_cache_list = []

        for i in range(len(model_applies)):
            apply_fn = model_applies[i]
            params = model_params[i]

            # Apply model
            _, x_cache = apply_fn(variables=params, x=x,
                                  theta=dummy_theta, train=False)

            x_cache_list.append(x_cache)

        @jax.jit
        def cached_approximate_log_likelihood_to_evidence(theta):
            total_log_r = 0.0

            print('here cached')
            if theta.ndim < 2:
                theta = jnp.reshape(theta, (1, -1))

            for i in range(len(model_applies)):
                apply_fn = model_applies[i]
                params = model_params[i]
                x_cache = x_cache_list[i]

                # Apply model
                log_r, _ = apply_fn(variables=params, x=None,
                                    theta=theta, x_cache=x_cache, train=False)

                # Always apply calibration - will be no-op for models that don't need it
                if spline_cal:
                    spline = distrax.RationalQuadraticSpline(boundary_slopes='identity',
                                                             params=calibration_params[i], range_min=0.0, range_max=1.0)

                    log_r = logit(spline.forward(sigmoid(log_r)))

                else:
                    log_r = beta_calibrate_log_r(log_r, calibration_params[i])

                total_log_r += log_r

            return total_log_r.squeeze(-1)

        return cached_approximate_log_likelihood_to_evidence

    @jax.jit
    def approximate_log_posterior(x, theta):
        log_likelihood = approximate_log_likelihood_to_evidence(x, theta)
        in_bounds = jnp.all((theta > lower_bounds) &
                            (theta < upper_bounds))
        log_prior = jnp.where(in_bounds, - jnp.log(total_mass), -jnp.inf)
        return log_likelihood + log_prior

    return approximate_log_likelihood_to_evidence, wrapper_for_cached_approximate_log_likelihood_to_evidence  # approximate_log_posterior, \
    # use_empirical_acf, (model_applies, model_params, calibration_params)


if __name__ == '__main__':

    folder_path = r'D:\sbi_ambit\SBI_for_trawl_processes_and_ambit_fields\models\new_classifier\TRE_full_trawl\selected_models'
    trawl_process_type = 'sup_ig_nig_5p'
    use_tre = True
    use_summary_statistics = False
    dummy_x = jnp.load('trawl.npy')[[0], :]
    dummy_theta = jnp.ones([dummy_x.shape[0], 5])
    # calibration_file_name = 'spline_calibration_1500.npy' #OR
    calibration_file_name = 'beta_calibration_1500.pkl'
    # x = ....
    approx_like, wrapper_approx_like_cache = load_trained_models_for_posterior_inference(folder_path, dummy_x, trawl_process_type,
                                                                                         use_tre, use_summary_statistics, calibration_file_name)
    approx_like_cached = wrapper_approx_like_cache(dummy_x)

    n_runs = 500
    start = time.time()
    for _ in range(n_runs):
        # Force execution completion
        result = approx_like(dummy_x, dummy_theta).block_until_ready()
    end = time.time()
    time_function1 = (end - start) / n_runs

    # Time function2
    start = time.time()
    for _ in range(n_runs):
        # Force execution completion
        result = approx_like_cached(dummy_theta).block_until_ready()
    end = time.time()
    time_function2 = (end - start) / n_runs

    print(f"Function 1 average time: {time_function1:.6f} seconds")
    print(f"Function 2 average time: {time_function2:.6f} seconds")

    vmaped_cached = jax.jit(jax.vmap(approx_like_cached))
    vmaped_cached(jnp.array([dummy_theta, dummy_theta])).shape
