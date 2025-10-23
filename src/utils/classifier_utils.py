# -*- coding: utf-8 -*-
import os
import yaml
import pickle
import optax
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from flax.training import train_state
from jax.nn import sigmoid


if True:
    from path_setup import setup_sys_path
    setup_sys_path()

from src.utils.get_model import get_model
from src.utils.trawl_training_utils import loss_functions_wrapper


###############################################################################


def get_projection_function(path__):

    summary_path = os.path.join("models", "summary_statistics")
    acf_path = os.path.join(summary_path, "learn_acf", "best_model")
    marginal_path = os.path.join(summary_path, "learn_marginal", path__)

    # Load configs
    with open(os.path.join(acf_path, "config.yaml"), 'r') as f:
        acf_config = yaml.safe_load(f)

    with open(os.path.join(marginal_path, "config.yaml"), 'r') as f:
        marginal_config = yaml.safe_load(f)

    # Load models
    acf_model, _, __ = get_model(acf_config)
    marginal_model, _, __ = get_model(marginal_config)

    # Load params
    with open(os.path.join(acf_path, "params.pkl"), 'rb') as file:
        acf_params = pickle.load(file)

    with open(os.path.join(marginal_path, "params.pkl"), 'rb') as file:
        marginal_params = pickle.load(file)

    ###########################################################################
    # don't need a training state, but it's easier to use the predict function #
    # which is already defined; so we use fake optimizers to create states     #
    ###########################################################################

    acf_optimizer = optax.adam(learning_rate=0.1)
    mar_optimizer = optax.adam(learning_rate=0.1)

    acf_state = train_state.TrainState.create(
        apply_fn=acf_model.apply,
        params=acf_params,
        tx=acf_optimizer
    )

    marginal_state = train_state.TrainState.create(
        apply_fn=marginal_model.apply,
        params=marginal_params,
        tx=acf_optimizer
    )
    # above is a hack to load he predct function
    ###########################################################################

    predict_theta_acf, _, __, ___ = loss_functions_wrapper(
        acf_state, acf_config)
    predict_theta_mar, _, __, ___ = loss_functions_wrapper(
        marginal_state, marginal_config)

    @jax.jit
    def project(trawl):

        # first return acf parmas, then marginal params
        # train = False and the dropout_rng which is set to jax.random.PRNGKey(0) is not used
        acf_projection = predict_theta_acf(
            acf_params, trawl, jax.random.PRNGKey(0), False)
        marginal_projection = predict_theta_mar(
            marginal_params, trawl, jax.random.PRNGKey(0), False)

        return jnp.concatenate([acf_projection, marginal_projection], axis=1)

    return project


def tre_shuffle(x_a, theta_a, theta_b, classifier_config):

    batch_size = theta_a.shape[0]

    x = jnp.vstack([x_a, x_a])
    Y = jnp.concatenate([jnp.ones(batch_size), jnp.zeros(batch_size)])
    ################################################################

    tre_config = classifier_config['tre_config']
    trawl_config = classifier_config['trawl_config']

    use_tre = tre_config['use_tre']
    tre_type = tre_config['tre_type']
    trawl_process_type = trawl_config['trawl_process_type']

    if not use_tre:
        theta = jnp.vstack([theta_a, theta_b])

        return x, theta, Y

    # can assume we are using TRE from here on

    if trawl_process_type == 'sup_ig_nig_5p':

        if tre_type == 'beta':
            theta_modified = jnp.concatenate(
                [theta_a[:, :4], theta_b[:, -1:]], axis=1)

        elif tre_type == 'sigma':
            theta_modified = jnp.concatenate(
                [theta_a[:, :3], theta_b[:, -2:]], axis=1)

        elif tre_type == 'mu':
            theta_modified = jnp.concatenate(
                [theta_a[:, :2], theta_b[:, -3:]], axis=1)

        elif tre_type == 'acf':
            theta_modified = theta_b

        theta = jnp.vstack([theta_a, theta_modified])

    else:
        raise ValueError('trawl process type not found')

    return x, theta, Y
