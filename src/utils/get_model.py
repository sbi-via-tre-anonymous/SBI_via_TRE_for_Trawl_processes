# -*- coding: utf-8 -*-
import os
import yaml
import pickle
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import numpy as np

if True:
    from path_setup import setup_sys_path
    setup_sys_path()

from src.model.LSTM_based_nn import LSTMModel
from src.model.VariableLSTM_based_nn import VariableLSTMModel
from src.model.Dense_model import DenseModel
from src.model.Extended_model_nn import ExtendedModel
from statsmodels.tsa.stattools import acf as compute_empirical_acf


def get_model(config_file, initialize=True):
    model_name = config_file['model_config']['model_name']

    if model_name == 'VariableLSTMModel':
        return get_model_VariableLSTM(config_file, initialize)
    elif model_name == 'LSTMModel':
        return get_model_LSTM(config_file, initialize)
    elif model_name == 'DenseModel':
        return get_model_Dense(config_file, initialize)
    else:
        raise ValueError('model_name not recognized, please check config file')


def get_model_LSTM(config_file, initialize=True):

    # Sanity checks
    trawl_config = config_file['trawl_config']
    model_config = config_file['model_config']

    assert model_config['model_name'] == 'LSTMModel'
    assert model_config['with_theta'] in [True, False]
    ###################################################

    # Get hyperparams
    key = PRNGKey(config_file['prng_key'])
    key, subkey = jax.random.split(key)

    seq_len = trawl_config['seq_len']
    batch_size = trawl_config['batch_size']
    theta_size = trawl_config['theta_size']

    lstm_hidden_size = model_config['lstm_hidden_size']
    num_lstm_layers = model_config['num_lstm_layers']
    linear_layer_sizes = model_config['linear_layer_sizes']
    mean_aggregation = model_config['mean_aggregation']
    final_output_size = model_config['final_output_size']
    dropout_rate = model_config['dropout_rate']

    # Create model
    model = LSTMModel(
        lstm_hidden_size=lstm_hidden_size,
        num_lstm_layers=num_lstm_layers,
        linear_layer_sizes=linear_layer_sizes,
        mean_aggregation=mean_aggregation,
        final_output_size=final_output_size,
        dropout_rate=dropout_rate
    )

    if not initialize:
        return model

    # Initialize model

    # Dummy input
    # [batch_size, sequence_length, feature_size]
    dummy_input = jax.random.normal(subkey, (batch_size, seq_len, 1))

    # Low-dimensional parameter (can be of any size)
    if model_config['with_theta']:

        dummy_theta = jax.random.normal(subkey, (batch_size, theta_size))
        params = model.init(subkey, dummy_input, dummy_theta)

    else:

        params = model.init(subkey, dummy_input)

    return model, params, key


def get_model_VariableLSTM(config_file, initialize=True):

    # Sanity checks
    trawl_config = config_file['trawl_config']
    model_config = config_file['model_config']

    assert model_config['model_name'] == 'VariableLSTMModel'
    assert 'with_theta' not in model_config.keys()
    ###################################################

    # Get hyperparams
    key = PRNGKey(config_file['prng_key'])
    key, subkey = jax.random.split(key)

    # seq_len = trawl_config['seq_len']
    seq_len = 1500  # used to initialize, but shouldn t influence anything
    batch_size = trawl_config['batch_size']
    theta_size = trawl_config['theta_size']

    lstm_hidden_size = model_config['lstm_hidden_size']
    num_lstm_layers = model_config['num_lstm_layers']
    linear_layer_sizes = model_config['linear_layer_sizes']
    mean_aggregation = model_config['mean_aggregation']
    final_output_size = model_config['final_output_size']
    dropout_rate = model_config['dropout_rate']
    increased_size = model_config['increased_size']

    if 'variable_seq_len' not in trawl_config.keys():

        add_seq_len = False

    else:

        add_seq_len = trawl_config['variable_seq_len']

    # Create model
    model = VariableLSTMModel(
        lstm_hidden_size=lstm_hidden_size,
        num_lstm_layers=num_lstm_layers,
        linear_layer_sizes=linear_layer_sizes,
        mean_aggregation=mean_aggregation,
        final_output_size=final_output_size,
        increased_size=increased_size,
        dropout_rate=dropout_rate,
        add_seq_len=add_seq_len
    )

    if not initialize:
        return model

    # Initialize model

    # Dummy input
    # [batch_size, sequence_length, feature_size]
    dummy_input = jax.random.normal(subkey, (batch_size, seq_len, 1))

    # Low-dimensional parameter (can be of any size)
    # if model_config['with_theta']:

    dummy_theta = jax.random.normal(subkey, (batch_size, theta_size))
    params = model.init(subkey, dummy_input, dummy_theta)

    # else:
    #
    #    params = model.init(subkey, dummy_input)

    return model, params, key


def get_model_Dense(config_file, initialize=True):
    # Sanity checks
    trawl_config = config_file['trawl_config']
    model_config = config_file['model_config']

    assert model_config['model_name'] == 'DenseModel'
    assert model_config['with_theta'] in [True, False]
    ###################################################

    # Get hyperparams
    key = PRNGKey(config_file['prng_key'])
    key, subkey = jax.random.split(key)

    seq_len = trawl_config['input_size']  # 3trawl_config['seq_len']
    batch_size = trawl_config['batch_size']
    theta_size = trawl_config['theta_size']

    # adjust dimensionality of the input when using summary statistics
    # if 'tre_config' in config_file.keys():
    #    tre_config = config_file['tre_config']
    #    if tre_config['use_summary_statistics']:
    #        seq_len = tre_config['summary_statistics_input_size']

    linear_layer_sizes = model_config['linear_layer_sizes']
    final_output_size = model_config['final_output_size']
    dropout_rate = model_config['dropout_rate']

    # Create model
    model = DenseModel(
        linear_layer_sizes=linear_layer_sizes,
        final_output_size=final_output_size,
        dropout_rate=dropout_rate
    )

    if not initialize:
        return model

    # Initialize model

    # Dummy input
    # [batch_size, feature_size]
    dummy_input = jax.random.normal(subkey, (batch_size, seq_len))

    # Low-dimensional parameter (can be of any size)
    if model_config['with_theta']:
        dummy_theta = jax.random.normal(subkey, (batch_size, theta_size))
        params = model.init(subkey, dummy_input, dummy_theta)
    else:
        params = model.init(subkey, dummy_input)

    return model, params, key

