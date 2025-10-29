# -*- coding: utf-8 -*-
from netcal.metrics.confidence import ECE, MCE, ACE
from src.utils.plot_calibration_map import plot_calibration_map
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from betacal import BetaCalibration
from jax.scipy.special import logit
import pandas as pd
import distrax
import optax
import jax.numpy as jnp
from jax.random import PRNGKey
from functools import partial
from statsmodels.tsa.stattools import acf as compute_empirical_acf
from src.utils.get_model import get_model
from src.utils.reconstruct_beta_calibration import beta_calibrate_log_r
from src.utils.get_trained_models import load_one_tre_model_only_and_prior_and_bounds
from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.classifier_utils import get_projection_function, tre_shuffle
from netcal.presentation import ReliabilityDiagram
from src.model.Extended_model_nn import VariableExtendedModel  # ,ExtendedModel
from jax.nn import sigmoid
import numpy as np
import datetime
import pickle
import yaml
import jax
import os
import netcal
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
if True:
    from path_setup import setup_sys_path
    setup_sys_path()
    import matplotlib.pyplot as plt


plt.rcParams["text.usetex"] = False


# generate calibration dataset
def generate_dataset_from_Y_equal_1(classifier_config, nr_batches):

    # Get params and
    tre_config = classifier_config['tre_config']
    use_tre = tre_config['use_tre']
    use_summary_statistics = tre_config['use_summary_statistics']
    replace_acf = tre_config['replace_full_trawl_with_acf']
    tre_type = tre_config['tre_type']

    assert not use_summary_statistics
    assert (not replace_acf) or (not (use_tre and tre_type == 'acf'))
    
    if use_summary_statistics:
        project_trawl = get_projection_function()

    trawl_config = classifier_config['trawl_config']
    batch_size = trawl_config['batch_size']
    key = jax.random.split(PRNGKey(np.random.randint(1, 10000)), batch_size)

    # Get data generators
    theta_acf_simulator, theta_marginal_simulator, trawl_simulator = get_theta_and_trawl_generator(
        classifier_config)

    # Generate calibration data
    # cal_trawls = []
    cal_thetas = []
    cal_x = []

    for _ in range(nr_batches):


        theta_acf_cal, key = theta_acf_simulator(key)
        theta_marginal_jax_cal, theta_marginal_tf_cal, key = theta_marginal_simulator(
            key)
        trawl_cal, key = trawl_simulator(
            theta_acf_cal, theta_marginal_tf_cal, key)

        ########################################
        if use_summary_statistics:
            raise ValueError('not within the scope of the paper')

        elif (not use_summary_statistics) and replace_acf and use_tre and tre_type == 'acf':

            raise ValueError
  
        else:
            x_cal = trawl_cal

            ########################################
        theta_cal = jnp.concatenate(
            [theta_acf_cal, theta_marginal_jax_cal], axis=1)

        ### DO THE SHUFFLING WHEN CALIBRATING; THE DATASET WILL JUST ###
        ### CONTAIN SAMPLES  FROM THE JOINT  ###

        cal_thetas.append(np.array(theta_cal))
        cal_x.append(np.array(x_cal))

    # cal_trawls = jnp.array(cal_trawls)  # , axis=0)
    cal_x = np.array(cal_x)       # , axis=0)
    cal_thetas = np.array(cal_thetas)  # , axis=0)

    return cal_x, cal_thetas  # cal_trawls, cal_x, cal_thetas, Y


def calibrate(trained_classifier_path, nr_batches, seq_len):

    # Load config file of the trained classifier
    with open(os.path.join(trained_classifier_path, "config.yaml"), 'r') as f:
        classifier_config = yaml.safe_load(f)
    
    # load hyperparams from the config file
    trawl_config = classifier_config['trawl_config']
    tre_config = classifier_config['tre_config']
    trawl_process_type = trawl_config['trawl_process_type']
    use_tre = tre_config['use_tre']
    tre_type = tre_config['tre_type']
    use_summary_statistics = tre_config['use_summary_statistics']
    replace_acf = tre_config['replace_full_trawl_with_acf']
    
    #sanity checks
    assert not use_summary_statistics
    assert (not replace_acf) or (not use_tre or tre_type != 'acf')

    # cheeck if log ratio, corresponding predicted probability output of the classifier
    # and label , 0 or 1, for which predicted probability is > 0.5
    # are precomputed; if they are not, we have to simulate the data, 
    # save it, then cache the summary statistics, save these 
    # and then pass these through the MLP head 
    
    calibration_results_path =  os.path.join(trained_classifier_path, 'calibration_results')

    log_r_path = os.path.join(
        calibration_results_path, f'log_r_{seq_len}_{tre_type}.npy' if use_tre else f'log_r_{seq_len}.npy')
    pred_prob_Y_path = os.path.join(
        calibration_results_path, f'pred_prob_Y_{seq_len}_{tre_type}.npy' if use_tre else f'pred_prob_Y_{seq_len}.npy' )
    Y_path = os.path.join(calibration_results_path, f'Y_{seq_len}_{tre_type}.npy' if use_tre else f'Y_{seq_len}.npy')

        
    # if at least one is missing: 
    # simulate the data, then precompute the cached statistics for later use
    # and then precompute log_r, pred_prob and  corresponding predicted Y label
    if not (os.path.isfile(log_r_path) and os.path.isfile(pred_prob_Y_path) and os.path.isfile(Y_path)):
        
        print('precomputed log_r, prob_prob_Y or pred_Y_label missing; initiating their computation now; consider \
              using a GPU or a dataset with a samll nr of batches')
        
        dataset_path = os.path.join(os.path.dirname(
                os.path.dirname(os.path.dirname(trained_classifier_path))), 'calibration_datasets', f'cal_dataset_{seq_len}')
        
        # calibration dataset paths
        cal_x_path = os.path.join(dataset_path, 'cal_x_joint.npy')
        cal_thetas_path = os.path.join(dataset_path, 'cal_thetas_joint.npy')

        if os.path.isfile(cal_x_path) and os.path.isfile(cal_thetas_path):
            print('Validation dataset already created')
            # Don't load the entire arrays at once - we'll load and process in batches later
        else:
            from copy import deepcopy
            print('Generating dataset')
            classifier_config_ = deepcopy(classifier_config)
            classifier_config_['trawl_config']['seq_len'] = seq_len
            cal_x, cal_thetas = generate_dataset_from_Y_equal_1(
                classifier_config_, nr_batches)
            print('Generated dataset')
    
            np.save(file=cal_x_path, arr=cal_x)
            np.save(file=cal_thetas_path, arr=cal_thetas)


        # Load model to compute the cached summary statistics 
        # model, _, _ = get_model(classifier_config)
        model, params, _, __ = load_one_tre_model_only_and_prior_and_bounds(trained_classifier_path,
                                                                            jnp.ones([1, seq_len]),
                                                                            trawl_process_type, tre_type)
        # save x_cache
        cal_x_cache_path = os.path.join(trained_classifier_path, 'calibration_results') 
        cal_x_cache_path = os.path.join(cal_x_cache_path, f'x_cache_{tre_type}_{seq_len}.npy' if use_tre else f'x_cache_{seq_len}.npy')

        # check if cached statistics are available, if not recompute
        if os.path.isfile(cal_x_cache_path):
    
            print('cached x is already saved')
            cal_thetas_array = np.load(cal_thetas_path, mmap_mode='r')
            cal_x_cache_array = np.load(cal_x_cache_path, mmap_mode='r')
    
        else:
            print('cached x is not available; initiating computation')
            # compute and save x_cache
            cal_thetas_array = np.load(cal_thetas_path, mmap_mode='r')
            cal_x_array = np.load(cal_x_path, mmap_mode='r')
    
            nr_batches, batch_size, _ = cal_x_array.shape
            assert _ == seq_len
    
            # dummy to get x_cache shape
            dummy_x_ = jnp.ones([1, seq_len])
            dummy_theta_ = jnp.ones([1, 5])
            _, dummy_x_cache_batch = model.apply(
                variables=params, x=dummy_x_, theta=dummy_theta_)
    
            x_cache_shape = dummy_x_cache_batch.shape[-1]
            full_shape = (nr_batches, batch_size, x_cache_shape)
    
            cal_x_cache_array = np.lib.format.open_memmap(cal_x_cache_path, mode='w+',
                                                          dtype=np.float32, shape=full_shape)
    
            for i in range(nr_batches):
    
                cal_thetas_batch = jnp.array(cal_thetas_array[i])
                cal_x_batch = jnp.array(cal_x_array[i])
    
                _, x_cache_batch = model.apply(
                    params, cal_x_batch, cal_thetas_batch)
                cal_x_cache_array[i] = np.array(x_cache_batch)
    
                if i % 50 == 0:
    
                    cal_x_cache_array.flush()
    
            cal_x_cache_array.flush()
    
            print('finished caching x')
            del cal_x_array
            del cal_x_cache_array
            del cal_thetas_array

        # compute log_r, pred_prob_Y, Y with the saved data
        cal_thetas_array = np.load(cal_thetas_path, mmap_mode='r')
        cal_x_cache_array = np.load(cal_x_cache_path, mmap_mode='r')
        log_r = []
        Y = []
        pred_prob_Y = []

        for i in range(nr_batches):

            cal_theta_batch = jnp.array(cal_thetas_array[i])
            cal_x_cache_batch = jnp.array(cal_x_cache_array[i])

            cal_x_cache_batch, cal_theta_batch, cal_Y_to_append = tre_shuffle(
                cal_x_cache_batch, cal_theta_batch, jnp.roll(cal_theta_batch, -1, axis=0), classifier_config)

            log_r_to_append, _ = model.apply(
                params, None, cal_theta_batch, x_cache=cal_x_cache_batch)
            pred_prob_Y_to_append = jax.nn.sigmoid(log_r_to_append)

            log_r.append(log_r_to_append)
            Y.append(cal_Y_to_append)
            pred_prob_Y.append(pred_prob_Y_to_append)

        np.save(arr=np.concatenate(log_r, axis=0),       file=log_r_path)
        np.save(arr=np.concatenate(pred_prob_Y, axis=0), file=pred_prob_Y_path)
        np.save(arr=np.concatenate(Y, axis=0),           file=Y_path)

        del cal_x_cache_array

    # else:
    # log_r, pred_prob_Y, Y are precomputed
    
    log_r = np.load(log_r_path)
    pred_prob_Y = np.load(pred_prob_Y_path)
    Y = np.load(Y_path)

    print(f"pred_prob_Y shape: {pred_prob_Y.shape}")
    print(f"pred_prob_Y dtype: {pred_prob_Y.dtype}")
    print(f"Y shape: {np.array(Y).shape}")
    
    ### DO CALIBRATION ###

    def compute_metrics(log_r, classifier_output, Y):
        extended_bce_loss = optax.losses.sigmoid_binary_cross_entropy(
            logits=log_r, labels=Y)
        mask = jnp.logical_and(Y == 0, log_r == -jnp.inf)
        extended_bce_loss = jnp.where(mask, 0.0, extended_bce_loss)
        bce_loss = jnp.mean(extended_bce_loss)
        S = jnp.mean(log_r[Y == 1])
        B = 2 * jnp.mean(classifier_output)
        accuracy = jnp.mean(
            (classifier_output > 0.5).astype(jnp.float32) == Y)
        return bce_loss, S, B

    # perform isotonic regression, Beta and Plat scaling
    lr = LogisticRegression(C=99999999999)
    iso = IsotonicRegression(y_min=10**(-6), y_max=1 -
                             10**(-6), out_of_bounds='clip')
    bc = BetaCalibration(parameters="abm")

    # These sklearn methods work with numpy arrays
    lr.fit(pred_prob_Y, np.array(Y))
    iso.fit(pred_prob_Y, np.array(Y))
    bc.fit(pred_prob_Y, np.array(Y))
    
    # Save the calibration files
    filename = os.path.join(
            trained_classifier_path, f"fitted_iso_{seq_len}.pkl") #f"fitted_iso_{seq_len}_{tre_type}.pkl" if use_tre else f"fitted_iso_{seq_len}.pkl")
    with open(filename, 'wb') as file:
            pickle.dump(iso, file)


    beta_calibration_dict = {'use_beta_calibration': True,
                             'params': bc.calibrator_.map_}

    # open a text file
    with open(os.path.join(trained_classifier_path, f'beta_calibration_{seq_len}.pkl'), 'wb') as f:
        #f'beta_calibration_{seq_len}_{tre_type}.pkl' if use_tre else f'beta_calibration_{seq_len}.pkl'), 'wb') as f:
        # serialize the list
        pickle.dump(beta_calibration_dict, f)

    linspace = np.linspace(0.0001, 0.9999, 200)
    pr = [lr.predict_proba(linspace.reshape(-1, 1))[:, 1],
          iso.predict(linspace), bc.predict(linspace)]  # , spline.forward(linspace)]
    methods_text = ['logistic', 'isotonic', 'beta']  # , 'splines']

    # get calibrated datasets
    calibrated_pr = [lr.predict_proba(pred_prob_Y)[:, 1],
                     iso.predict(pred_prob_Y), bc.predict(pred_prob_Y)]

    log_r_jax = jnp.array(log_r.squeeze())
    pred_prob_Y_jax = jnp.array(pred_prob_Y.squeeze())
    Y_jax = jnp.array(Y)

    metrics = []
    metrics.append(compute_metrics(log_r_jax, pred_prob_Y_jax, Y_jax))

    # True for the validation set;
    # this sometimes requires adjusting the number of bins lower to not give numerical errors
    # can be set to False to get some partial results quicker
        
    # ECE metrics are only computable for component NREs within the TRE
    if use_tre:
        ece_false = []
        ece_true = []
        mce_true = []
        mce_false = []
        ace_true = []
        ace_false = []

        # do ECE metrics for the uncalbirated classifier
        ece_false.append(ECE(bins=5, equal_intervals=False).measure(
            np.array(pred_prob_Y_jax), np.array(Y_jax)))
        ece_true.append(ECE(bins=20, equal_intervals=True).measure(
            np.array(pred_prob_Y_jax), np.array(Y_jax)))
        mce_false.append(MCE(bins=5, equal_intervals=False).measure(
            np.array(pred_prob_Y_jax), np.array(Y_jax)))
        mce_true.append(MCE(bins=20, equal_intervals=True).measure(
            np.array(pred_prob_Y_jax), np.array(Y_jax)))
        ace_false.append(ACE(bins=5, equal_intervals=False).measure(
            np.array(pred_prob_Y_jax), np.array(Y_jax)))
        ace_true.append(ACE(bins=20, equal_intervals=True).measure(
            np.array(pred_prob_Y_jax), np.array(Y_jax)))
        
    # no do ECE metrics for the calbirated classifiers
    for i in range(len(methods_text)):
        # Convert one calibrated result at a time
        calibrated_pr_jax = jnp.array(calibrated_pr[i])
        logit_calibrated = logit(calibrated_pr_jax)
        metrics.append(compute_metrics(
            logit_calibrated, calibrated_pr_jax, Y_jax))
        
        if use_tre:
        
            ece_false.append(ECE(bins=5, equal_intervals=False).measure(
                np.array(calibrated_pr_jax), np.array(Y_jax)))
            ece_true.append(ECE(bins=20, equal_intervals=True).measure(
                np.array(calibrated_pr_jax), np.array(Y_jax)))

            mce_false.append(MCE(bins=5, equal_intervals=False).measure(
                np.array(calibrated_pr_jax), np.array(Y_jax)))
            mce_true.append(MCE(bins=20, equal_intervals=True).measure(
                np.array(calibrated_pr_jax), np.array(Y_jax)))

            ace_false.append(ACE(bins=5, equal_intervals=False).measure(
                np.array(calibrated_pr_jax), np.array(Y_jax)))
            ace_true.append(ACE(bins=20, equal_intervals=True).measure(
                np.array(calibrated_pr_jax), np.array(Y_jax)))
            
    if use_tre:
        miscal_metrics = np.vstack([ece_false, ace_false, mce_false,
                                   ece_true, ace_true, mce_true])
        data = np.concatenate(
            [np.array(metrics).transpose(), miscal_metrics], axis=0)
        df = pd.DataFrame(data,
                      columns=['uncal'] + methods_text,
                      index=['BCE', 'S', 'B']+['ECE_f', 'ACE_f', 'MCE_f', 'ECE_t', 'ACE_t', 'MCE_t'])


    else:
        df = pd.DataFrame(np.array(metrics).transpose(),
                      columns=['uncal', ' lr', 'isotonic', 'beta'], 
                      index=('BCE', 'S', 'B'))

    df.to_excel(os.path.join(calibration_results_path,
                f'BCE_S_B_{seq_len}_{tre_type}.xlsx' if use_tre else f'BCE_S_B_{seq_len}.xlsx'))#_with_splines.xlsx'))

 

def validate(trained_classifier_path, nr_batches, seq_len):

    # Load config file of the trained classifier
    with open(os.path.join(trained_classifier_path, "config.yaml"), 'r') as f:
        classifier_config = yaml.safe_load(f)
    
    # load hyperparams from the config file
    trawl_config = classifier_config['trawl_config']
    tre_config = classifier_config['tre_config']
    trawl_process_type = trawl_config['trawl_process_type']
    use_tre = tre_config['use_tre']
    tre_type = tre_config['tre_type']
    use_summary_statistics = tre_config['use_summary_statistics']
    replace_acf = tre_config['replace_full_trawl_with_acf']
    
    #sanity checks
    assert not use_summary_statistics
    assert (not replace_acf) or (not use_tre or tre_type != 'acf')

    # cheeck if log ratio, corresponding predicted probability output of the classifier
    # and label , 0 or 1, for which predicted probability is > 0.5
    # are precomputed; if they are not, we have to simulate the data, 
    # save it, then cache the summary statistics, save these 
    # and then pass these through the MLP head 
    
    validation_results_path =  os.path.join(trained_classifier_path, 'validation_results')

    log_r_path = os.path.join(
        validation_results_path, f'log_r_{seq_len}_{tre_type}.npy' if use_tre else f'log_r_{seq_len}.npy')
    pred_prob_Y_path = os.path.join(
        validation_results_path, f'pred_prob_Y_{seq_len}_{tre_type}.npy' if use_tre else f'pred_prob_Y_{seq_len}.npy' )
    Y_path = os.path.join(validation_results_path, f'Y_{seq_len}_{tre_type}.npy' if use_tre else f'Y_{seq_len}.npy')

        
    # if at least one is missing: 
    # simulate the data, then precompute the cached statistics for later use
    # and then precompute log_r, pred_prob and  corresponding predicted Y label
    if not (os.path.isfile(log_r_path) and os.path.isfile(pred_prob_Y_path) and os.path.isfile(Y_path)):
        
        print('precomputed log_r, prob_prob_Y or pred_Y_label missing; initiating their computation now; consider \
              using a GPU or a dataset with a samll nr of batches')
        
        dataset_path = os.path.join(os.path.dirname(
                os.path.dirname(os.path.dirname(trained_classifier_path))), 'validation_datasets', f'val_dataset_{seq_len}')
        
        # validation dataset paths
        val_x_path = os.path.join(dataset_path, 'val_x_joint.npy')
        val_thetas_path = os.path.join(dataset_path, 'val_thetas_joint.npy')

        if os.path.isfile(val_x_path) and os.path.isfile(val_thetas_path):
            print('Validation dataset already created')
            # Don't load the entire arrays at once - we'll load and process in batches later
        else:
            from copy import deepcopy
            print('Generating dataset')
            classifier_config_ = deepcopy(classifier_config)
            classifier_config_['trawl_config']['seq_len'] = seq_len
            val_x, val_thetas = generate_dataset_from_Y_equal_1(
                classifier_config_, nr_batches)
            print('Generated dataset')
    
            np.save(file=val_x_path, arr=val_x)
            np.save(file=val_thetas_path, arr=val_thetas)


        # Load model to compute the cached summary statistics 
        # model, _, _ = get_model(classifier_config)
        model, params, _, __ = load_one_tre_model_only_and_prior_and_bounds(trained_classifier_path,
                                                                            jnp.ones([1, seq_len]),
                                                                            trawl_process_type, tre_type)
        # save x_cache
        val_x_cache_path = os.path.join(trained_classifier_path, 'validation_results') 
        val_x_cache_path = os.path.join(val_x_cache_path, f'x_cache_{tre_type}_{seq_len}.npy' if use_tre else f'x_cache_{seq_len}.npy')

        # check if cached statistics are available, if not recompute
        if os.path.isfile(val_x_cache_path):
    
            print('cached x is already saved')
            val_thetas_array = np.load(val_thetas_path, mmap_mode='r')
            val_x_cache_array = np.load(val_x_cache_path, mmap_mode='r')
    
        else:
            print('cached x is not available; initiating computation')
            # compute and save x_cache
            val_thetas_array = np.load(val_thetas_path, mmap_mode='r')
            val_x_array = np.load(val_x_path, mmap_mode='r')
    
            nr_batches, batch_size, _ = val_x_array.shape
            assert _ == seq_len
    
            # dummy to get x_cache shape
            dummy_x_ = jnp.ones([1, seq_len])
            dummy_theta_ = jnp.ones([1, 5])
            _, dummy_x_cache_batch = model.apply(
                variables=params, x=dummy_x_, theta=dummy_theta_)
    
            x_cache_shape = dummy_x_cache_batch.shape[-1]
            full_shape = (nr_batches, batch_size, x_cache_shape)
    
            val_x_cache_array = np.lib.format.open_memmap(val_x_cache_path, mode='w+',
                                                          dtype=np.float32, shape=full_shape)
    
            for i in range(nr_batches):
    
                val_thetas_batch = jnp.array(val_thetas_array[i])
                val_x_batch = jnp.array(val_x_array[i])
    
                _, x_cache_batch = model.apply(
                    params, val_x_batch, val_thetas_batch)
                val_x_cache_array[i] = np.array(x_cache_batch)
    
                if i % 50 == 0:
    
                    val_x_cache_array.flush()
    
            val_x_cache_array.flush()
    
            print('finished caching x')
            del val_x_array
            del val_x_cache_array
            del val_thetas_array

        # compute log_r, pred_prob_Y, Y with the saved data
        val_thetas_array = np.load(val_thetas_path, mmap_mode='r')
        val_x_cache_array = np.load(val_x_cache_path, mmap_mode='r')
        log_r = []
        Y = []
        pred_prob_Y = []

        for i in range(nr_batches):

            val_theta_batch = jnp.array(val_thetas_array[i])
            val_x_cache_batch = jnp.array(val_x_cache_array[i])

            val_x_cache_batch, val_theta_batch, val_Y_to_append = tre_shuffle(
                val_x_cache_batch, val_theta_batch, jnp.roll(val_theta_batch, -1, axis=0), classifier_config)

            log_r_to_append, _ = model.apply(
                params, None, val_theta_batch, x_cache=val_x_cache_batch)
            pred_prob_Y_to_append = jax.nn.sigmoid(log_r_to_append)

            log_r.append(log_r_to_append)
            Y.append(val_Y_to_append)
            pred_prob_Y.append(pred_prob_Y_to_append)

        np.save(arr=np.concatenate(log_r, axis=0),       file=log_r_path)
        np.save(arr=np.concatenate(pred_prob_Y, axis=0), file=pred_prob_Y_path)
        np.save(arr=np.concatenate(Y, axis=0),           file=Y_path)

        del val_x_cache_array

    # else:
    # log_r, pred_prob_Y, Y are precomputed
    
    log_r = np.load(log_r_path)
    pred_prob_Y = np.load(pred_prob_Y_path)
    Y = np.load(Y_path)
    

    print(f"pred_prob_Y shape: {pred_prob_Y.shape}")
    print(f"pred_prob_Y dtype: {pred_prob_Y.dtype}")
    print(f"Y shape: {np.array(Y).shape}")
    
    log_r_jax = jnp.array(log_r.squeeze())
    pred_prob_Y_jax = jnp.array(pred_prob_Y.squeeze())
    Y_jax = jnp.array(Y)
    
    ### DO CALIBRATION ###

    def compute_metrics(log_r, classifier_output, Y):
        extended_bce_loss = optax.losses.sigmoid_binary_cross_entropy(
            logits=log_r, labels=Y)
        mask = jnp.logical_and(Y == 0, log_r == -jnp.inf)
        extended_bce_loss = jnp.where(mask, 0.0, extended_bce_loss)
        bce_loss = jnp.mean(extended_bce_loss)
        S = jnp.mean(log_r[Y == 1])
        B = 2 * jnp.mean(classifier_output)
        accuracy = jnp.mean(
            (classifier_output > 0.5).astype(jnp.float32) == Y)
        return bce_loss, S, B


    # open a text file
    with open(os.path.join(trained_classifier_path, f'beta_calibration_{seq_len}.pkl'), 'rb') as f:
                           #f'beta_calibration_{seq_len}_{tre_type}.pkl' if use_tre else f'beta_calibration_{seq_len}.pkl'), 'rb') as f:
        # serialize the list
        beta_calibration_dict = pickle.load(f)

    # Load the model
    with open(os.path.join(trained_classifier_path, f'fitted_iso_{seq_len}.pkl'), 'rb') as file:
                           #f'fitted_iso_{seq_len}_{tre_type}.pkl' if use_tre else f'fitted_iso_{seq_len}.pkl'), 'rb') as file:
        iso_regression = pickle.load(file)

    methods_text = ['iso', 'beta']  

    # get calibrated datasets
    beta_cal_log_r = beta_calibrate_log_r(
        log_r, beta_calibration_dict['params']).squeeze()

    calibrated_pr = [jnp.array(iso_regression.predict(pred_prob_Y)),
                     sigmoid(beta_cal_log_r).squeeze( )
                     ]


    metrics = []
    metrics.append(compute_metrics(log_r_jax, pred_prob_Y_jax, Y_jax))
    
    # ECE metrics are only computable for component NREs within the TRE
    if use_tre:
        ece_false = []
        ece_true = []
        mce_true = []
        mce_false = []
        ace_true = []
        ace_false = []

        # do ECE metrics for the uncalbirated classifier
        ece_false.append(ECE(bins=5, equal_intervals=False).measure(
            np.array(pred_prob_Y_jax), np.array(Y_jax)))
        ece_true.append(ECE(bins=20, equal_intervals=True).measure(
            np.array(pred_prob_Y_jax), np.array(Y_jax)))
        mce_false.append(MCE(bins=5, equal_intervals=False).measure(
            np.array(pred_prob_Y_jax), np.array(Y_jax)))
        mce_true.append(MCE(bins=20, equal_intervals=True).measure(
            np.array(pred_prob_Y_jax), np.array(Y_jax)))
        ace_false.append(ACE(bins=5, equal_intervals=False).measure(
            np.array(pred_prob_Y_jax), np.array(Y_jax)))
        ace_true.append(ACE(bins=20, equal_intervals=True).measure(
            np.array(pred_prob_Y_jax), np.array(Y_jax)))
        
    # no do ECE metrics for the calbirated classifiers
    for i in range(len(methods_text)):
        # Convert one calibrated result at a time
        calibrated_pr_jax = jnp.array(calibrated_pr[i])
        logit_calibrated = logit(calibrated_pr_jax)
        metrics.append(compute_metrics(
            logit_calibrated, calibrated_pr_jax, Y_jax))
        
        if use_tre:
        
            ece_false.append(ECE(bins=5, equal_intervals=False).measure(
                np.array(calibrated_pr_jax), np.array(Y_jax)))
            ece_true.append(ECE(bins=20, equal_intervals=True).measure(
                np.array(calibrated_pr_jax), np.array(Y_jax)))

            mce_false.append(MCE(bins=5, equal_intervals=False).measure(
                np.array(calibrated_pr_jax), np.array(Y_jax)))
            mce_true.append(MCE(bins=20, equal_intervals=True).measure(
                np.array(calibrated_pr_jax), np.array(Y_jax)))

            ace_false.append(ACE(bins=5, equal_intervals=False).measure(
                np.array(calibrated_pr_jax), np.array(Y_jax)))
            ace_true.append(ACE(bins=20, equal_intervals=True).measure(
                np.array(calibrated_pr_jax), np.array(Y_jax)))
            
    if use_tre:
        miscal_metrics = np.vstack([ece_false, ace_false, mce_false,
                                   ece_true, ace_true, mce_true])
        data = np.concatenate(
            [np.array(metrics).transpose(), miscal_metrics], axis=0)
        df = pd.DataFrame(data,
                      columns=['uncal'] + methods_text,
                      index=['BCE', 'S', 'B']+['ECE_f', 'ACE_f', 'MCE_f', 'ECE_t', 'ACE_t', 'MCE_t'])


    else:
        df = pd.DataFrame(np.array(metrics).transpose(),
                      columns=['uncal', 'isotonic', 'beta'], 
                      index=('BCE', 'S', 'B'))

    df.to_excel(os.path.join(validation_results_path,
                f'BCE_S_B_{seq_len}_{tre_type}.xlsx' if use_tre else f'BCE_S_B_{seq_len}.xlsx'))





if __name__ == '__main__':
    nr_batches = 5000
    
    nre_base_path = os.path.join(os.getcwd(),'models_and_simulated_datasets','classifiers','NRE_full_trawl','best_model')
    tre_base_path = os.path.join(os.getcwd(),'models_and_simulated_datasets','classifiers','TRE_full_trawl','selected_models')
    tre_types = ('acf','beta','mu','sigma')
    
    paths_list = [nre_base_path]
    
    # calibrate and validate the NRE and then the NREs within TRE
    # gather paths to saved models first
    for tre_type in tre_types:
        paths_list.append(os.path.join(tre_base_path,tre_type))
        
    for trained_classifier_path in paths_list:
        
        # calibrate for all seq lengths: 1000, 1500, 2000
        calibrate(trained_classifier_path, nr_batches, 2000)
        calibrate(trained_classifier_path, nr_batches, 1500)
        calibrate(trained_classifier_path, nr_batches, 1000)

        # validate for all seq lengths: 1000, 1500, 2000
        validate(trained_classifier_path, nr_batches, 2000)
        validate(trained_classifier_path, nr_batches, 1500)
        validate(trained_classifier_path, nr_batches, 1000)


