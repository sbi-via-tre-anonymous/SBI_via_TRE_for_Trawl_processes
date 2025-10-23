import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from functools import partial
from src.utils.get_model import get_model
from src.utils.estimate_bias import estimate_bias_from_MAP, estimate_bias_from_posterior_samples_batched
from src.utils.get_trained_models import load_one_tre_model_only_and_prior_and_bounds
from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.reconstruct_beta_calibration import beta_calibrate_log_r
from src.utils.KL_divergence import convert_3_to_4_param_nig
import os
import pickle
from tqdm import tqdm
import numpy as np
from src.utils.reconstruct_beta_calibration import beta_calibrate_log_r
from src.utils.chebyshev_utils import chebint_ab, interpolation_points_domain, integrate_from_sampled, polyfit_domain,  \
    vec_polyfit_domain, sample_from_coeff, chebval_ab_jax, vec_sample_from_coeff,\
    chebval_ab_for_one_x, vec_chebval_ab_for_multiple_x_per_envelope_and_multple_envelopes,\
    vec_integrate_from_samples
import statsmodels as sm
from sequential_posteror_sampling import create_parameter_sweep_fn, model_apply_wrapper, predict_2d, \
    apply_calibration, estimate_first_density_enclosure, get_cond_prob_at_true_value
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow_probability.substrates.jax as tfp
tfp_dist = tfp.distributions

@jax.jit
def corr_sup_ig_envelope(h, params):
    gamma, eta = params
    return jnp.exp(eta * (1 - jnp.sqrt(1 + 2*h/gamma**2)))

def run_analysis_for_dataset(dataset_name, base_path):
    """
    Run analysis for a single dataset
    """
    tre_types_list = ['acf', 'mu', 'sigma', 'beta']
    trawl_process_type = 'sup_ig_nig_5p'
    N = 128
    num_samples = 1000
    key = jax.random.PRNGKey(100)
    vec_key = jax.random.PRNGKey(999)
    vec_key = jax.random.split(vec_key, num_samples)
    calibration_type = 'beta'
    nlags = 40  # Number of lags for ACF
    
    # Load data set
    data_path = os.path.join(base_path, 'MSTL_results_14', dataset_name, f'{dataset_name}_MSTL.csv')
    data_df = pd.read_csv(data_path)
    data_column = data_df['resid']

    
    # Prepare data - use full series
    data_column = data_column.replace([np.inf, -np.inf], np.nan).dropna()
    x = data_column.values
    x = jnp.array(x).reshape(1, -1)  # Shape: (1, seq_len)
    
    seq_len = x.shape[1]
    print(f"Processing {dataset_name} - Series length: {seq_len}")
    
    # Create dummy_x with actual sequence length for model loading
    dummy_x = jnp.ones([1, seq_len])
    
    # Load models
    models_dict = dict()
    apply_fn_dict = dict()
    appl_fn_to_get_x_cache_dict = dict()
    parameter_sweeps_dict = dict()
    calibration_dict = dict()
    x_cache_dict = dict()
    bounds_dict = {'acf': [10., 20.], 'beta': [-5., 5.], 'mu': [-1., 1.], 'sigma': [0.5, 1.5]}
    
    for tre_type in tre_types_list:
        trained_classifier_path = os.path.join(os.getcwd(),'models_and_simulated_datasets','classifiers','TRE_full_trawl','selected_models', str(tre_type))
        #trained_classifier_path = f'D:\\sbi_foler\\sbi_via_tre_for_trawl_processes\\models_and_simulated_datasets\\classifiers\\TRE_full_trawl\\selected_models\\{tre_type}'
        model, params, _, __bounds = load_one_tre_model_only_and_prior_and_bounds(
            trained_classifier_path, dummy_x, trawl_process_type, tre_type)
        
        models_dict[tre_type] = model
        apply_fn_to_get_x_cache, apply_fn = model_apply_wrapper(model, params)
        apply_fn_dict[tre_type] = apply_fn
        appl_fn_to_get_x_cache_dict[tre_type] = apply_fn_to_get_x_cache
        
        # Load calibration with length_2000
        calibration_file = os.path.join(trained_classifier_path, 'beta_calibration_2000.pkl')
        with open(calibration_file, 'rb') as file:
            calibration_dict[tre_type] = pickle.load(file)['params']
        
        parameter_sweeps_dict[tre_type] = create_parameter_sweep_fn(
            tre_type, apply_fn_dict, bounds_dict, N+1)
        
        # Compute x_cache for the full series
        x_normalized = (x - jnp.mean(x, axis=1, keepdims=True)) / jnp.std(x, axis=1, keepdims=True)
        _, x_cache = apply_fn_to_get_x_cache(x_normalized, jnp.ones((1, 5)))
        x_cache_dict[tre_type] = x_cache
    
    def acf_integrate_partial(samples):
        return integrate_from_sampled(samples, a=bounds_dict['acf'][0], b=bounds_dict['acf'][1])
    vec_integrate_2nd_component_acf_from_sampled = jax.jit(jax.vmap(acf_integrate_partial))
    
    estimate_first_density = estimate_first_density_enclosure('acf', parameter_sweeps_dict, bounds_dict, N)
    
    # Run inference for the single time series
    print(f"Running inference for {dataset_name}...")
    
    tre_type = 'acf'
    true_x_cache = x_cache_dict[tre_type][0]  # Only one time series
    
    two_d_log_prob = estimate_first_density(true_x_cache)
    two_d_prob = apply_calibration(two_d_log_prob, tre_type, calibration_type, calibration_dict)
    f_x = vec_integrate_2nd_component_acf_from_sampled(two_d_prob)
    cheb_coeff_f_x = polyfit_domain(f_x, bounds_dict[tre_type][0], bounds_dict[tre_type][1])
    
    key, subkey = jax.random.split(key)
    first_comp_samples = sample_from_coeff(cheb_coeff_f_x, subkey, bounds_dict[tre_type][0], bounds_dict[tre_type][1], num_samples)
    normalizing_constant_acf = integrate_from_sampled(f_x, bounds_dict[tre_type][0], bounds_dict[tre_type][1])
    first_comp_densities = chebval_ab_jax(first_comp_samples, cheb_coeff_f_x,
                                          bounds_dict[tre_type][0], bounds_dict[tre_type][1]) / normalizing_constant_acf
    
    thetas_ = jnp.zeros([num_samples, 5])
    thetas_ = thetas_.at[:, 0].set(first_comp_samples)
    sample_densities = jnp.copy(first_comp_densities)
    
    for col_index, tre_type in enumerate(tre_types_list, 1):
        x_cache_to_use = x_cache_dict[tre_type][0]
        x_cache_to_use_expanded = jnp.broadcast_to(x_cache_to_use, (num_samples, x_cache_to_use.shape[-1]))
        log_conditional_prob_at_cheb_knots = parameter_sweeps_dict[tre_type](thetas_, x_cache_to_use_expanded)
        
        conditional_prob_at_cheb_knots = apply_calibration(log_conditional_prob_at_cheb_knots, tre_type, calibration_type, calibration_dict)
        conditional_density_cheb_coeff = vec_polyfit_domain(conditional_prob_at_cheb_knots, bounds_dict[tre_type][0], bounds_dict[tre_type][1])
        
        split_keys = jax.vmap(lambda k: jax.random.split(k, num=2))(vec_key)
        last_component_samples = vec_sample_from_coeff(conditional_density_cheb_coeff, vec_key, bounds_dict[tre_type][0], bounds_dict[tre_type][1], 1)
        vec_key = split_keys[:, 0]
        
        normalizing_constants = vec_integrate_from_samples(conditional_prob_at_cheb_knots, bounds_dict[tre_type][0], bounds_dict[tre_type][1])
        conditional_prob = vec_chebval_ab_for_multiple_x_per_envelope_and_multple_envelopes(
            last_component_samples, conditional_density_cheb_coeff, bounds_dict[tre_type][0], bounds_dict[tre_type][1]).squeeze() / normalizing_constants
        
        sample_densities *= conditional_prob
        thetas_ = thetas_.at[:, col_index].set(last_component_samples.squeeze())
    
    # Transform back to original scale
    result_samples = np.array(thetas_)
    mean_x, std_x = np.mean(x), np.std(x)
    argmax = int(np.argmax(sample_densities))
    map_marginal_params_standardized = result_samples[argmax][2:].copy()  # Get mu, sigma, beta before transformation

    result_samples[:, 2] = result_samples[:, 2] * std_x + mean_x  # mu
    result_samples[:, 3] = result_samples[:, 3] * std_x  # sigma
    
    # Get MAP estimate
    results_MAP = result_samples[argmax]
    
    # Save results
    save_path = os.path.join(base_path, 'MSTL_results_14', dataset_name)
    np.save(os.path.join(save_path, 'posterior_samples.npy'), result_samples)
    np.save(os.path.join(save_path, 'MAP_results.npy'), results_MAP)
    np.save(os.path.join(save_path, 'sample_densities.npy'), sample_densities)

    h = jnp.arange(0, nlags+1, 1)  # Changed from nlags to nlags+1
    
    # Calculate empirical ACF from the original series
    empirical_acf = sm.tsa.stattools.acf(np.array(x.squeeze()), nlags=nlags)
    
    # Calculate ACF from posterior samples
    param_samples = result_samples[:, :2]
    acf_curves = jax.vmap(lambda pars: corr_sup_ig_envelope(h, pars))(param_samples)
    posterior_mean = np.mean(np.array(acf_curves), axis=0)
    posterior_median = np.median(np.array(acf_curves), axis=0)
    
    # MAP ACF
    map_acf_params = results_MAP[:2]
    map_acf = corr_sup_ig_envelope(h, map_acf_params)

    # Compute 95% CI from posterior ACF samples
    acf_lower_95 = np.percentile(np.array(acf_curves), 2.5, axis=0)
    acf_upper_95 = np.percentile(np.array(acf_curves), 97.5, axis=0)
    
    # Compute 95% CI for mu, sigma, beta parameters
    mu_ci = np.percentile(result_samples[:, 2], [2.5, 97.5])
    sigma_ci = np.percentile(result_samples[:, 3], [2.5, 97.5])
    beta_ci = np.percentile(result_samples[:, 4], [2.5, 97.5])
    
    # Effective decorrelation time (lag where MAP ACF < 0.05)
    map_acf_at_50 = corr_sup_ig_envelope(jnp.arange(0, 50, 1), results_MAP[:2])
    effective_decorr_lag = np.where(np.array(map_acf_at_50) < 0.05)[0]
    if len(effective_decorr_lag) == 0:
        raise ValueError(f"MAP ACF does not decay below 0.05 within 50 lags for {dataset_name}")
    effective_decorr_time = int(effective_decorr_lag[0])
    
    # Lag-1 correlation from posterior samples with 95% CI
    lag1_acf_samples = acf_curves[:, 1]
    lag1_acf_mean = np.mean(lag1_acf_samples)
    lag1_acf_ci = np.percentile(lag1_acf_samples, [2.5, 97.5])
    
    # Save parameter CIs and ACF statistics
    param_stats = {
        'MAP': results_MAP,
        'mu_ci': mu_ci,
        'sigma_ci': sigma_ci,
        'beta_ci': beta_ci,
        'effective_decorr_time': effective_decorr_time,
        'lag1_acf_mean': lag1_acf_mean,
        'lag1_acf_ci': lag1_acf_ci
    }
    np.save(os.path.join(save_path, 'parameter_statistics.npy'), param_stats)
    
    if dataset_name == 'AZPS':
        # for AZPS we display a further plot, which requires debiasing the empiricla ACF
        # Estimate bias using MAP parameters
        print(f"Estimating CI for {dataset_name}...")
        key, bias_key = jax.random.split(key)
        # Use normalized data for bias estimation
        observed_trawl = x_normalized.squeeze()
        num_replications_for_empirical_ACF_CI = 1000  # Adjust for faster inference 
        
        mean_bias, lower_ci_bias, upper_ci_bias = estimate_bias_from_MAP(
            results_MAP, observed_trawl, num_replications_for_empirical_ACF_CI, nlags, bias_key
        )
        # Save results
        np.save(os.path.join(save_path, 'bias_estimates.npy'), {'mean': mean_bias, 'lower_ci': lower_ci_bias, 'upper_ci': upper_ci_bias})
    

        # Empirical ACF confidence intervals using bias quantiles
        empirical_upper = empirical_acf - lower_ci_bias  # lower bias → upper bound
        empirical_lower = empirical_acf - upper_ci_bias  # upper bias → lower bound
        
        # Bartlett's formula for ACF standard errors
        n = len(x.squeeze())
        varacf = np.ones(nlags+1) / n
        varacf[0] = 0
        # For lag k>1: var(acf[k]) = (1/n) * (1 + 2*sum(acf[j]^2 for j=1 to k-1))
        varacf[2:] = (1 + 2 * np.cumsum(empirical_acf[1:-1]**2)) / n
        se_bands = 1.96 * np.sqrt(varacf)
        
        # Create combined marginal density and ACF plot
        fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(12, 4.25))
        
        # Left plot: Marginal density
        bins = 25
        data_array = np.array(x).squeeze()
        map_marginal_params_nig = convert_3_to_4_param_nig(map_marginal_params_standardized)
        x_min, x_max = data_array.min(), data_array.max()
        x_range = np.linspace(x_min, x_max, 10000)
        x_range_standardized = (x_range - mean_x) / std_x
        
        try:
            map_nig_dist = tfp_dist.NormalInverseGaussian(*map_marginal_params_nig, validate_args=True)
            map_pdf_standardized = map_nig_dist.prob(x_range_standardized)
            map_pdf = map_pdf_standardized / std_x
        except Exception as e:
            print(f"Warning: NIG parameters may be invalid. Standardized marginal params: {map_marginal_params_standardized}")
            raise ValueError
        
        ax1.hist(data_array, bins=bins, density=True, alpha=0.3, color='gray', 
                 edgecolor='black', linewidth=0.5)
        sns.kdeplot(data=data_array, ax=ax1, color='blue', linewidth=2, label='KDE', alpha=0.8)
        ax1.plot(x_range, map_pdf, 'r-', linewidth=2, label='MAP NIG', alpha=0.8)
        ax1.set_xlim(x_min, x_max)
        ax1.set_xlabel('Value', fontsize=14)
        ax1.set_ylabel('Density', fontsize=14)
        ax1.set_title('Marginal Density', fontsize=16)
        ax1.legend(loc='best', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Right plot: ACF
        ax2.plot(h, empirical_acf, 'o-', alpha=0.9, label='Empirical ACF', 
                 color='darkblue', markersize=4, zorder=3)
        ax2.fill_between(h, empirical_lower, empirical_upper, alpha=0.2, color='darkblue', 
                          label='95% CI', zorder=1)
        ax2.plot(h, map_acf, '-', linewidth=2, label='MAP ACF', color='red', zorder=3)
        ax2.fill_between(h, acf_lower_95, acf_upper_95, alpha=0.2, color='red', 
                          label='95% CI (theoretical)', zorder=1)
        ax2.plot(h, posterior_mean, '--', alpha=0.7, label='Posterior mean', color='orange', linewidth=1.5)
        ax2.plot(h, posterior_median, ':', alpha=0.7, label='Posterior median', color='green', linewidth=1.5)
        ax2.plot(h, se_bands, linestyle='--', color='gray', alpha=0.5, linewidth=0.8, label='Bartlett 95% bands')
        ax2.plot(h, -se_bands, linestyle='--', color='gray', alpha=0.5, linewidth=0.8)
        ax2.axhline(y=0, linestyle='-', color='black', alpha=0.3, linewidth=0.5)
        ax2.set_xlabel('Lag', fontsize=14)
        ax2.set_ylabel('ACF', fontsize=14)
        ax2.set_title('Autocorrelation Function', fontsize=16)
        ax2.legend(loc='upper right', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.2, 1.0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), 'src', 'visualisations', 'Figure6.pdf'), dpi=600, bbox_inches="tight")
        plt.close()
        
    
        
        print(f"Completed analysis for {dataset_name}")
    
    return result_samples, results_MAP

def main():
    """
    Main function to run analysis for all datasets
    """
    # Set base paths
    project_base = os.getcwd()
    data_base = os.path.join(project_base, 'application_pre_processing_Figure5')
    
    # Change to project directory to ensure imports work
    os.chdir(project_base)
    
    # List of datasets to process
    datasets = ['AZPS', 'ERCO', 'CISO',  'NYIS', 'PJM', 'MISO', 'FPL', 'DUK',  'BPAT'] #['NYIS', 'PJM', 'MISO']#,  #
    
    # Process each dataset for both data types
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        all_results[dataset_name] = {}
        
        # Process MSTL results
        try:
            print(f"\nProcessing MSTL data for {dataset_name}...")
            samples_mstl, map_mstl = run_analysis_for_dataset(dataset_name, data_base)
            all_results[dataset_name] = {'samples': samples_mstl, 'MAP': map_mstl}
        except Exception as e:
            print(f"Error processing MSTL data for {dataset_name}: {e}")

    
    # Create summary plots comparing all datasets
    print(f"\n{'='*60}")
    print("Creating summary plots...")
    print(f"{'='*60}")
    
    
    # Summary table of MAP estimates
    summary_data = []
    for dataset_name in datasets:
        if dataset_name in all_results:
            map_result = all_results[dataset_name]['MAP']
                
            #Load parameter statistics
            save_path = os.path.join(data_base, 'MSTL_results_14', dataset_name)
            param_stats = np.load(os.path.join(save_path, 'parameter_statistics.npy'), allow_pickle=True).item()
                
            summary_data.append({
                    'Dataset': dataset_name,
                    'Gamma': map_result[0],
                    'Eta': map_result[1],
                    'Mu': map_result[2],
                    'Mu_CI_lower': param_stats['mu_ci'][0],
                    'Mu_CI_upper': param_stats['mu_ci'][1],
                    'Sigma': map_result[3],
                    'Sigma_CI_lower': param_stats['sigma_ci'][0],
                    'Sigma_CI_upper': param_stats['sigma_ci'][1],
                    'Beta': map_result[4],
                    'Beta_CI_lower': param_stats['beta_ci'][0],
                    'Beta_CI_upper': param_stats['beta_ci'][1],
                    'Effective_Decorr_Time': param_stats['effective_decorr_time'],
                    'Lag1_ACF': param_stats['lag1_acf_mean'],
                    'Lag1_ACF_CI_lower': param_stats['lag1_acf_ci'][0],
                    'Lag1_ACF_CI_upper': param_stats['lag1_acf_ci'][1]
            })
    
    summary_df = pd.DataFrame(summary_data) 
    summary_df.to_csv(os.path.join(os.getcwd(), 'src', 'visualisations', 'Table3.csv'), index=False)
    print("\nSummary of MAP estimates saved to 'summary_MAP_estimates.csv'")
    
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()