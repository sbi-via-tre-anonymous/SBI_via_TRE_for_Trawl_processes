if True:
    from path_setup import setup_sys_path
    setup_sys_path()

import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from src.utils.acf_functions import get_acf
from src.utils.KL_divergence import vec_monte_carlo_kl_3_param_nig


def compare_point_estimators(true_theta, infered_theta, mle_or_gmm, num_lags):

    # acf errors
    H = np.arange(1, num_lags + 1)
    # this should be num_trawls, num_lags shaped
    theoretical_acf = acf_func(H, true_theta[:, :2])
    infered_acf = acf_func(H, infered_theta[:, :2])
    acf_differences = np.abs(theoretical_acf - infered_acf)
    # might want to convert these to integrals or to sums
    med_L1_acf = np.median(np.mean(acf_differences, axis=1))
    L1_acf = np.mean(np.mean(acf_differences, axis=1))
    L2_acf = np.mean(np.sqrt(np.mean(acf_differences**2, axis=1)))
    rMSE_acf = np.mean(np.mean(acf_differences**2, axis=1))**0.5

    # marginal errors
    marginal_medAEs = np.median(
        np.abs((true_theta - infered_theta)[:, 2:]), axis=0)
    marginal_MAEs = np.mean(
        np.abs((true_theta - infered_theta)[:, 2:]), axis=0)
    marginal_MSEs = np.mean(
        np.square((true_theta - infered_theta)[:, 2:]), axis=0)**0.5

    print('Starting KL divergence calculations')

    batched_true_thetas = [jnp.array(true_theta[i:i + 100, 2:])
                           for i in range(0, len(true_theta), 100)]
    batched_inferred_thetas = [
        jnp.array(infered_theta[i:i + 100, 2:]) for i in range(0, len(infered_acf), 100)]

    forward_kl = []
    rev_kl = []

    for i in range(len(batched_true_thetas)):

        num_samples = 7500
        params1 = batched_true_thetas[i]
        params2 = batched_inferred_thetas[i]
        vec_key = jax.random.split(jax.random.PRNGKey(12414), len(params1))

        forward_kl.append(vec_monte_carlo_kl_3_param_nig(
            params1, params2, vec_key, num_samples)[0].mean().item())
        rev_kl.append(vec_monte_carlo_kl_3_param_nig(
            params2, params1, vec_key, num_samples)[0].mean().item())

    acf_estimation_error = {
        'med_L1_acf': med_L1_acf,
        'L1_acf': L1_acf,
        'L2_acf': L2_acf,
        'rMSE_acf': rMSE_acf
    }

    marginal_estimation_error = {
        'marginal_medAE': marginal_medAEs,
        'marginal_MAEs': marginal_MAEs,
        'marginal_MSEs': marginal_MSEs,
        'forward_kl': np.mean(forward_kl),
        'rev_kl': np.mean(rev_kl),
    }

    np.save(os.path.join(results_path, mle_or_gmm + f'_acf_estimation_error_{seq_len}_{num_lags}.npy'),
            acf_estimation_error)
    np.save(os.path.join(results_path, mle_or_gmm + f'_marginal_estimation_error_{seq_len}.npy'),
            marginal_estimation_error)


## next, the function above broken in two two function.
## the GMM estimators have nones, and there are different numbers of Nans in the ACF
## and marginal estimators, so we have to break the estimation error calculation into
# two parts, one for each

def compare_acf_point_estimators(true_theta, infered_theta_acf, mle_or_gmm, num_lags):

    # acf errors
    H = np.arange(1, num_lags + 1)
    # this should be num_trawls, num_lags shaped
    theoretical_acf = acf_func(H, true_theta[:, :2])
    infered_acf = acf_func(H, infered_theta_acf)  # infered_theta[:, :2])
    acf_differences = np.abs(theoretical_acf - infered_acf)
    # might want to convert these to integrals or to sums
    med_L1_acf = np.median(np.mean(acf_differences, axis=1))
    L1_acf = np.mean(np.mean(acf_differences, axis=1))
    L2_acf = np.mean(np.sqrt(np.mean(acf_differences**2, axis=1)))
    rMSE_acf = np.mean(np.mean(acf_differences**2, axis=1))**0.5

    acf_estimation_error = {
        'med_L1_acf': med_L1_acf,
        'L1_acf': L1_acf,
        'L2_acf': L2_acf,
        'rMSE_acf': rMSE_acf
    }

    np.save(os.path.join(results_path, mle_or_gmm + f'_acf_estimation_error_{seq_len}_{num_lags}.npy'),
            acf_estimation_error)


def compare_marginal_point_estimators(true_theta, infered_theta, mle_or_gmm):

    # marginal errors
    marginal_medAEs = np.median(
        np.abs((true_theta - infered_theta)), axis=0)
    marginal_MAEs = np.mean(
        np.abs((true_theta - infered_theta)), axis=0)
    marginal_MSEs = np.mean(
        np.square((true_theta - infered_theta)), axis=0)**0.5

    print('Starting KL divergence calculations')

    batched_true_thetas = [jnp.array(true_theta[i:i + 100, :])
                           for i in range(0, len(true_theta), 100)]
    batched_inferred_thetas = [
        jnp.array(infered_theta[i:i + 100]) for i in range(0, len(infered_theta), 100)]

    forward_kl = []
    rev_kl = []

    for i in range(len(batched_true_thetas)):

        num_samples = 7500
        params1 = batched_true_thetas[i]
        params2 = batched_inferred_thetas[i]
        vec_key = jax.random.split(jax.random.PRNGKey(12414), len(params1))

        forward_kl.append(vec_monte_carlo_kl_3_param_nig(
            params1, params2, vec_key, num_samples)[0].mean().item())
        rev_kl.append(vec_monte_carlo_kl_3_param_nig(
            params2, params1, vec_key, num_samples)[0].mean().item())

    marginal_estimation_error = {
        'marginal_medAE': marginal_medAEs,
        'marginal_MAEs': marginal_MAEs,
        'marginal_MSEs': marginal_MSEs,
        'forward_kl': np.mean(forward_kl),
        'rev_kl': np.mean(rev_kl),
    }
    print(marginal_estimation_error)

    np.save(os.path.join(results_path, mle_or_gmm + f'_marginal_estimation_error_{seq_len}.npy'),
            marginal_estimation_error)
    return marginal_estimation_error
    # np.load(os.path.join(results_path,'acf_estimation_error.npy'),allow_pickle=True)


if __name__ == '__main__':

    seq_len = 1000
    num_lags = 35

    acf_func = jax.vmap(get_acf('sup_IG'), in_axes=(None, 0))
    
    # analyze point estimators for one of TRE, NRE, GMM and NBE
    MLE_TRE = False
    MLE_NRE = False
    GMM = False
    NBE = True
    
    if NBE:

        nbe_type = 'direct'  # or kl, rev, sym
    assert MLE_TRE + MLE_NRE + GMM + NBE == 1

    if MLE_TRE and not MLE_NRE and not GMM and not NBE:

        folder_path  =  os.path.join(os.getcwd(),'models_and_simulated_datasets','point_estimators','TRE')
        results_path = os.path.join(folder_path, f'TRE_results_seq_len_{seq_len}')
        os.makedirs(results_path, exist_ok=True)

        df = pd.read_pickle(os.path.join(
            results_path, 'TRE_MLE_results_no_calibration.pkl'))
        true_theta = np.array([np.array(i) for i in df.true_theta.values])
        infered_theta = np.array([np.array(i) for i in df.MLE.values])
        compare_point_estimators(true_theta, infered_theta, 'MLE', num_lags)

    elif MLE_NRE and not MLE_TRE and not GMM and not NBE:

        folder_path  =  os.path.join(os.getcwd(),'models_and_simulated_datasets','point_estimators','NRE')
        results_path = os.path.join(folder_path, f'NRE_results_seq_len_{seq_len}')
        os.makedirs(results_path, exist_ok=True)

        df = pd.read_pickle(os.path.join(
            results_path, f'NRE_MLE_results_no_calibration.pkl'))

        true_theta = np.array([np.array(i) for i in df.true_theta.values])
        infered_theta = np.array([np.array(i) for i in df.MLE.values])
        compare_point_estimators(true_theta, infered_theta, 'MLE', num_lags)

    elif NBE and not MLE_NRE and not MLE_TRE and not GMM:

        folder_path  =  os.path.join(os.getcwd(),'models_and_simulated_datasets','point_estimators','NBE')
        #results_path = os.path.join(folder_path, f'NBE_{seq_len}')
        results_path = os.path.join(folder_path,f'NBE_results_seq_len_{seq_len}')
        os.makedirs(results_path, exist_ok=True)

        infer_acf_theta = np.load(os.path.join(results_path, f'infered_theta_acf_{seq_len}.npy'))
        infer_mar_theta = np.load(os.path.join(results_path, f'direct_infered_marginal_{seq_len}.npy'))
        
        infered_theta = np.concatenate([infer_acf_theta, infer_mar_theta], axis=1)
        true_theta = np.load(os.path.join(results_path,f'true_theta_{seq_len}.npy'))
        compare_point_estimators(true_theta, infered_theta, 'NBE', num_lags)


    elif GMM:

        folder_path  =  os.path.join(os.getcwd(),'models_and_simulated_datasets','point_estimators','GMM')
        results_path = os.path.join(folder_path,f'GMM_results_seq_len_{seq_len}')
        os.makedirs(results_path, exist_ok=True)


        ####  do marginal ####
        df_marginal = pd.read_pickle(os.path.join(
            results_path, f'marginal_GMM_seq_len_{seq_len}.pkl'))  # f'ACF_{seq_len}_{num_lags}.pkl'))
        df_acf      = pd.read_pickle(os.path.join(
            results_path, f'ACF_GMM_seq_len_{seq_len}_{num_lags}.pkl'))

        df_marginal = df_marginal.replace({None: np.nan}).dropna()
        df_acf = df_acf.replace({None: np.nan}).dropna()


        # there are problems reading the pickle saved from the cluster because of differentt
        # numpy version. to this end, i save as a csv and then change the strings to arrays after
        # reading
        
        true_marginal_theta = np.vstack(df_marginal.true_theta.values)[:,2:]
        true_acf_theta = np.vstack(df_acf.true_theta.values)[:,:2]
        
        infered_marginal_theta = np.vstack(df_marginal.GMM.values)
        infered_acf_theta = np.vstack(df_acf.GMM.values)
        
        compare_marginal_point_estimators(
            true_marginal_theta, infered_marginal_theta, 'GMM')
        
        compare_acf_point_estimators(
            true_acf_theta, infered_acf_theta, 'GMM', num_lags)


    else:
        raise ValueError
