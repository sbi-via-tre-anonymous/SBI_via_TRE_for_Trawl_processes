# -*- coding: utf-8 -*-

import jax
import os
import jax
import yaml
import pickle
import datetime
import time
import numpy as np
import jax.numpy as jnp
from functools import partial
from jax.random import PRNGKey
from src.utils.get_model import get_model
from src.utils.acf_functions import get_acf


acf_model_path = os.path.join(
    'models_and_simulated_datasets', 'NBE_ACF')
marginal_model_path = os.path.join(
    'models_and_simulated_datasets', 'NBE_marginal')

prefix = 'direct' #if 'direct' in marginal_model_path else 'kl'
# same script works for other types of NBEs, which are used in the
# supplementary material

# load config files
with open(os.path.join(acf_model_path, 'config.yaml'), 'r') as f:
    acf_config = yaml.safe_load(f)

with open(os.path.join(marginal_model_path, 'config.yaml'), 'r') as f:
    marginal_config = yaml.safe_load(f)

# load params
with open(os.path.join(acf_model_path, 'params.pkl'), 'rb') as f:
    acf_params = pickle.load(f)

with open(os.path.join(marginal_model_path, 'params.pkl'), 'rb') as f:
    marginal_params = pickle.load(f)


acf_model, _, __ = get_model(acf_config)
marginal_model, _, _ = get_model(marginal_config)


@jax.jit
def apply_acf_model(x):
    return acf_model.apply(acf_params, x)


@jax.jit
def apply_marginal_model(x):
    return marginal_model.apply(marginal_params, x)

# jit to fuse operations and reduce memory footprint


@jax.jit
def normalize_data(x):
    return (x - jnp.mean(x, axis=1, keepdims=True)) / \
        jnp.std(x, axis=1, keepdims=True)


if __name__ == '__main__':

    acf_func = jax.vmap(get_acf('sup_IG'), in_axes=(None, 0))
    num_rows_to_load = 160
    num_lags = 35

    MAE_m = dict()
    MSE_m = dict()
    L1_acf = dict()
    L2_acf = dict()

    for seq_len in (2000, 1500, 1000):
        
        point_estimators_path = os.path.join(os.getcwd(), 'models_and_simulated_datasets','point_estimators','NBE',f'NBE_results_seq_len_{seq_len}')
        os.makedirs(point_estimators_path,  exist_ok=True)

        dataset_path = os.path.join(os.getcwd(), 'models_and_simulated_datasets',
                                    'validation_datasets', f'val_dataset_{seq_len}')
    
        val_x_path = os.path.join(dataset_path, 'val_x_joint.npy')
        val_thetas_path = os.path.join(dataset_path, 'val_thetas_joint.npy')

        val_x = np.load(val_x_path, mmap_mode='r')[:num_rows_to_load]
        val_thetas = np.load(val_thetas_path)[:num_rows_to_load]

        val_x = val_x.reshape(-1, val_x.shape[-1])
        val_thetas = val_thetas.reshape(-1, val_thetas.shape[-1])
        
        
        # marginal
        infered_marginal = apply_marginal_model(val_x)
        infered_marginal = infered_marginal.at[:, 1].set(
            jnp.exp(infered_marginal[:, 1]))
        marginal_save_path = os.path.join(point_estimators_path, f'{prefix}_infered_marginal_{seq_len}.npy')
        np.save(file=marginal_save_path, arr=infered_marginal)

        # ACF
        val_x = normalize_data(val_x)
        infered_theta_acf = jnp.exp(apply_acf_model(val_x))
        ACF_save_path = os.path.join(point_estimators_path, f'infered_theta_acf_{seq_len}.npy')
        np.save(file=ACF_save_path, arr=infered_theta_acf)
        
        true_theta_path = os.path.join(point_estimators_path, f'true_theta_{seq_len}.npy')
        np.save(true_theta_path, val_thetas)



