from reconstruct_beta_calibration import beta_calibrate_log_r
import jax.numpy as jnp
import numpy as np
import jax
from jax.nn import sigmoid
import os
if True:
    from path_setup import setup_sys_path
    setup_sys_path()

import optax
import pandas as pd
import pickle


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


def apply_log_calibration(log_r, tre_type, calibration_type):
    """inputs log_r, outputs exponential of the calibrated classifier"""

    if calibration_type == 'None':

        return log_r

    elif calibration_type == 'beta':

        log_r = beta_calibrate_log_r(
            log_r, beta_calibration_params[tre_type])

        return log_r

    elif calibration_type == 'isotonic':

        # exp(logit(p)) = p / (1-p)
        # exp( logit( iso( sigma( )))) = iso( sigma( )) / (1 - iso( sigma( )))
        intermediary = predict_2d(iso_calibration_dict[tre_type],
                                  jax.nn.sigmoid(log_r))
        return jnp.log(intermediary / (1-intermediary))


def compute_metrics(log_r, Y):

    extended_bce_loss = optax.losses.sigmoid_binary_cross_entropy(
        logits=log_r, labels=Y)

    # this is due to numerical instability in the logit function and should be 0
    mask = jnp.logical_and(Y == 0, log_r == -jnp.inf)

    # Replace values where mask is True with 0, otherwise keep original values
    extended_bce_loss = jnp.where(mask, 0.0, extended_bce_loss)

    bce_loss = jnp.mean(extended_bce_loss)

    classifier_output = sigmoid(log_r)
    # half of them are 0s, half of them are 1, so we have to x2
    # S = 2 * jnp.mean(log_r * Y)
    S = jnp.mean(log_r[Y == 1])
    B = 2 * jnp.mean(classifier_output)
    accuracy = jnp.mean(
        (classifier_output > 0.5).astype(jnp.float32) == Y)

    return bce_loss, S, B  # , accuracy


def load_log_r_Y_TRE(seq_len, calibration_type):
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                             'models_and_simulated_datasets', 'classifiers', 
                             'TRE_full_trawl', 'selected_models'
                             )
    tre_types = ('acf', 'beta', 'mu', 'sigma')
    log_r = 0

    for tre_type in tre_types:

        best_model_path = os.path.join(base_path, tre_type, 'validation_results')

        log_r_path = os.path.join(
            best_model_path, f'log_r_{seq_len}_{tre_type}.npy')
        Y_path = os.path.join(best_model_path, f'Y_{seq_len}_{tre_type}.npy')
        log_r_to_add = jnp.load(log_r_path).squeeze()

     
        log_r_to_add = apply_log_calibration(
            log_r_to_add, tre_type, calibration_type)

        log_r += log_r_to_add

        if tre_type == 'acf':
            Y = jnp.load(Y_path)
        else:
            Y_new = jnp.load(Y_path)
            assert jnp.all(Y == Y_new)

    return log_r, Y


def load_NRE_BCE_S_B(seq_len):
    """only returns beta-calibratetd values, not isotonic regression."""
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                             'models_and_simulated_datasets', 'classifiers', 
                             'NRE_full_trawl','best_model', 'validation_results')

    df_NRE = pd.read_excel(os.path.join(base_path, f'BCE_S_B_{seq_len}.xlsx'),
                           header=0,     # Use first row as column names
                           index_col=0)  # Use first column as row index)

    df_NRE = df_NRE[['uncal', 'beta']]
    df_NRE.columns = ['uncal', 'cal']
    return df_NRE


def final_metrics_NRE_and_TRE_validation(seq_len, calibration_type):
    """this will read the validaton NRE_metrics, produce the TRE metrics,
    because we validated each of the individual NREs within TREs but not
    the TRE: BCE, S, B. The  
    'NRE_TRE_coverage_figures_and_ecdf_metrics.py' script will produce the deviations
    from uniform posterior cdf
    """

    # load NRE BCE,S,B metrics
    df_NRE = load_NRE_BCE_S_B(seq_len)

    # load TRE BCE,S,B,mertics
    uncal_log_r_TRE, Y_TRE = load_log_r_Y_TRE(seq_len, 'None')
    cal_log_r_TRE, _ = load_log_r_Y_TRE(seq_len, calibration_type)

    uncal_TRE = jnp.array(compute_metrics(uncal_log_r_TRE, Y_TRE))
    cal_TRE = jnp.array(compute_metrics(cal_log_r_TRE, Y_TRE))

    df_TRE = pd.DataFrame(data=np.stack([uncal_TRE, cal_TRE], axis=1), index=('BCE', 'S', 'B'),
                          columns=('uncal', 'cal'))
    df_combined = pd.concat([df_NRE, df_TRE], axis=1, keys=['NRE', 'TRE'])

    return df_combined


if __name__ == '__main__':
    
    save_path = os.path.join(os.path.dirname(os.getcwd()),'visualisations','aux_metrics')
    os.makedirs(save_path, exist_ok=True)


    for seq_len in (1000, 1500, 2000):
        calibration_type = 'beta'

        base_path_to_save_to = os.path.join(os.path.dirname(
            os.path.dirname(os.getcwd())), 'models_and_simulated_datasets', 'classifiers')

        # load calibrations files
        beta_calibration_params = dict()
        iso_calibration_dict = dict()

        for tre_type in ('acf', 'beta', 'mu', 'sigma'):

            trained_classifier_path = os.path.join(
                base_path_to_save_to, 'TRE_full_trawl', 'selected_models', tre_type)

            with open(os.path.join(trained_classifier_path, f'beta_calibration_{seq_len}.pkl'), 'rb') as file:
                beta_calibration_params[tre_type] = pickle.load(file)['params']

            with open(os.path.join(trained_classifier_path, f'fitted_iso_{seq_len}.pkl'), 'rb') as file:
                iso_calibration_dict[tre_type] = pickle.load(file)

        df = final_metrics_NRE_and_TRE_validation(seq_len, calibration_type)
        df = df.astype(float)
        file_name = f'BCE_S_B_for_NRE_and_TRE_{seq_len}_{calibration_type}.xlsx'
        output_path = os.path.join(save_path, file_name)
        df.to_excel(output_path, engine='openpyxl')
        print(f"Saved: {output_path}")
