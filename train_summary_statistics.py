if True:
    from path_setup import setup_sys_path
    setup_sys_path()

import os
import jax
import yaml
import wandb
import optax
import pickle
import datetime
import time
import numpy as np
import jax.numpy as jnp
from functools import partial
from jax.random import PRNGKey
from flax.training import train_state
from src.utils.get_model import get_model
from src.utils.acf_functions import get_acf
from src.utils.summary_statistics_plotting import plot_acfs, plot_marginals
from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.trawl_training_utils import loss_functions_wrapper
import matplotlib.pyplot as plt


def check_if_run_stopped():
    """Check if the current wandb run was manually stopped from the UI."""
    try:
        api = wandb.Api()
        run = api.run(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
        return run.state in ["finished", "failed", "crashed", "killed"]
    except Exception as e:
        print(f"Warning: Failed to check run status. Error: {e}")
        return False  # Assume it's running if there's an issue


def try_to_close_wandb():
    if wandb.run is not None:
        try:
            wandb.finish()
        except:
            pass
        # Small delay to ensure wandb is fully cleaned up
        time.sleep(5)



def train_and_evaluate(config):

    try:

        ###########################################################################
        # Check if we learn the acf or marginal and initialize wandb accordingly
        learn_config = config['learn_config']
        learn_acf = learn_config['learn_acf']
        learn_marginal = learn_config['learn_marginal']
        learn_both = learn_config['learn_both']
        use_kl_div = config['loss_config']['use_kl_div']
        kl_type = config['loss_config']['kl_type']

        assert learn_acf + learn_marginal == 1 and learn_both == False

        # Initialize wandb
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S") + \
            str(np.random.randint(1, 100))  # to make sure  names are different
        run_name = f"{timestamp}"

        project_name = 'summary_'
        project_name += ('acf_p_' + str(config['loss_config']
                         ['p'])) if learn_acf else 'marginal'

        if learn_marginal:

            if use_kl_div:
                project_name = str(kl_type) + '_' + project_name

            else:
                project_name = 'direct_' + project_name

        wandb.init(project=project_name, name=run_name, config=config,
                   tags=config['model_config']['model_name'])

        #######################################################################
        # Create folders for the experiment validation dataset and checkpoints
        base_checkpoint_dir = os.path.join("models", 'summary_statistics')
        # experiment_dir = os.path.join(base_checkpoint_dir,
        #                              "learn_acf" if learn_acf else "learn_marginal", wandb.run.name)
        if learn_acf:
            experiment_dir = os.path.join(
                base_checkpoint_dir, "learn_acf", wandb.run.name)

        elif learn_marginal:
            experiment_dir = os.path.join(
                base_checkpoint_dir, "learn_marginal", kl_type,  wandb.run.name)

        os.makedirs(experiment_dir, exist_ok=True)

        trawls_path = os.path.join(base_checkpoint_dir, 'trawls.npy')
        thetas_acf_path = os.path.join(base_checkpoint_dir, 'thetas_acf.npy')
        thetas_marginal_path = os.path.join(
            base_checkpoint_dir, 'thetas_marginal.npy')

        ###########################################################################
        # Get params and hyperparams for the data generating process
        trawl_config = config['trawl_config']
        batch_size = trawl_config['batch_size']
        val_batches = config["val_config"]["val_n_batches"]
        val_freq = config["val_config"]["val_freq"]

        # Get data generators
        theta_acf_simulator, theta_marginal_simulator, trawl_simulator = get_theta_and_trawl_generator(
            config)

        if not (os.path.isfile(trawls_path) and os.path.isfile(thetas_acf_path) and os.path.isfile(thetas_marginal_path)):

            val_data = []

            # Generate fixed validation set
            # Different seed for validation
            val_key = jax.random.split(
                PRNGKey(config['prng_key'] + 10), batch_size)
            for _ in range(val_batches):
                theta_acf_val, val_key = theta_acf_simulator(val_key)
                theta_marginal_jax_val, theta_marginal_tf_val, val_key = theta_marginal_simulator(
                    val_key)
                trawl_val, val_key = trawl_simulator(
                    theta_acf_val, theta_marginal_tf_val, val_key)

                val_data.append(
                    (trawl_val, theta_acf_val, theta_marginal_jax_val))

            # Convert validation data to JAX arrays
            # Saves it in the format [#batches, batch_size, vector_dimension]
            val_trawls = jnp.stack([x[0] for x in val_data])
            val_thetas_acf = jnp.stack([x[1] for x in val_data])
            val_thetas_marginal = jnp.stack([x[2] for x in val_data])

            # Save validation dataset
            np.save(trawls_path, np.array(val_trawls))
            np.save(thetas_acf_path, np.array(val_thetas_acf))
            np.save(thetas_marginal_path, np.array(val_thetas_marginal))

            print(f'{val_batches} batches simulated for the validation dataset.')

        val_trawls = jnp.load(trawls_path)

        if learn_acf:
            val_thetas = jnp.load(thetas_acf_path)
        else:
            val_thetas = jnp.load(thetas_marginal_path)

        ###########################################################################
        # Create model and initialize parameters for simulating data during training
        model, params, key = get_model(config)
        key = jax.random.split(PRNGKey(config['prng_key']+2351), batch_size)
        dropout_key = jax.random.PRNGKey(
            config['prng_key'] + 29354)  # for dropout

        # Initialize optimizer
        lr = config["optimizer"]["lr"]
        total_steps = config["train_config"]["n_iterations"]
        warmup_steps = 1000
        decay_steps = total_steps - warmup_steps

        # Constant learning rate for warmup_steps; Cosine decay afterwards.
        schedule_fn = optax.join_schedules([
            optax.constant_schedule(lr),
            optax.cosine_decay_schedule(
                init_value=lr,
                decay_steps=decay_steps,
                alpha=0.0075
            )
        ], boundaries=[warmup_steps])

        if config['optimizer']['name'] == 'adam':
            if 'weight_decay' in config['optimizer']:
                # AdamW = Adam with weight decay
                optimizer = optax.adamw(
                    learning_rate=schedule_fn,
                    weight_decay=config['optimizer']['weight_decay']
                )
            else:
                # Regular Adam if no weight_decay specified
                optimizer = optax.adam(learning_rate=schedule_fn)

        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer
        )

        ###########################################################################
        # Get params and hyperparams for the loss function
        loss_config = config['loss_config']
        num_KL_samples = loss_config['num_KL_samples']

        # Loss functions
        predict_theta, compute_loss, compute_loss_and_grad, \
            compute_validation_stats = loss_functions_wrapper(state, config)

        # Initialize best validation loss tracking
        best_val_loss = float('inf')
        best_iteration = -1
        best_model_path = os.path.join(experiment_dir, "best_model")
        os.makedirs(best_model_path, exist_ok=True)
        ###########################################################################

        # Training loop
        for iteration in range(config["train_config"]["n_iterations"]):

            theta_acf, key = theta_acf_simulator(key)
            theta_marginal_jax, theta_marginal_tf, key = theta_marginal_simulator(
                key)
            trawl, key = trawl_simulator(theta_acf, theta_marginal_tf, key)

            dropout_key, dropout_subkey_to_use = jax.random.split(dropout_key)

            # Compute loss and gradients
            if learn_acf:
                loss, grads = compute_loss_and_grad(
                    params, trawl, theta_acf, dropout_subkey_to_use, True, num_KL_samples)

            elif learn_marginal:
                loss, grads = compute_loss_and_grad(
                    params, trawl, theta_marginal_jax, dropout_subkey_to_use, True, num_KL_samples)

                # progressively increase the number of samples used for the MC approximation of the KL divergence
                # to balance speed in the early part of the training and accuracy towards the end
                if iteration == 2000:
                    num_KL_samples *= 2

                elif iteration == 4000:
                    num_KL_samples *= 2

                elif iteration == 18000:
                    num_KL_samples *= 2

                elif iteration == 22000:
                    num_KL_samples *= 2

            # Update model parameters
            state = state.apply_gradients(grads=grads)
            params = state.params

            # Logging and then validation
            loss_name = 'acf_loss' if learn_acf else (
                kl_type + '_marginal_loss' if use_kl_div else 'direct' + '_marginal_loss')
            train_loss, val_loss = 'train_' + loss_name, 'val_' + loss_name
            metrics = {
                train_loss: loss.item()
            }

            # Compute validation loss periodically
            if iteration % val_freq == 0 and iteration > 5000:

                val_loss, val_loss_std, dropout_key = compute_validation_stats(
                    params, val_trawls, val_thetas, dropout_key, num_KL_samples)

                # Log metrics under the same group for better visualization
                metrics.update({
                    "val_metrics/val_loss": val_loss.item(),
                    "val_metrics/val_loss_upper": val_loss.item() + 1.96 * val_loss_std.item() / val_trawls.shape[0]**0.5,
                    "val_metrics/val_loss_lower": val_loss.item() - 1.96 * val_loss_std.item() / val_trawls.shape[0]**0.5,
                })

                # Save just the parameters instead of full state
                params_filename = os.path.join(
                    experiment_dir, f"params_iter_{iteration}.pkl")
                with open(params_filename, 'wb') as f:
                    pickle.dump(state.params, f)

                # Keep track of best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_iteration = iteration

                pred_theta = predict_theta(
                    params, trawl, dropout_key, False)

                if learn_acf:

                    for i in range(3):
                        fig_ = plot_acfs(
                            trawl[i], theta_acf[i], pred_theta[i], config)
                        wandb.log({f"Acf plot {i}": wandb.Image(fig_)})

                elif learn_marginal:

                    for i in range(3):
                        fig_ = plot_marginals(
                            trawl[i], theta_marginal_jax[i], pred_theta[i], config)
                        wandb.log({f"Marginal plot {i}": wandb.Image(fig_)})

                        ############### EXTRA PLOTS ###################
                        tbeta = val_thetas[:, :, -1].flatten()
                        pbeta = jnp.array([predict_theta(
                            params, trawl, dropout_key, False)[:, -1] for trawl in val_trawls])
                        pbeta = pbeta.flatten()

                    try:  # PLOT TRUE VS INFERED BETAS
                        f_hist, ax = plt.subplots()
                        ax.hist(np.array(pbeta), label='pred b',
                                bins=25, alpha=0.5)
                        ax.hist(np.array(tbeta), label='true b',
                                bins=25, alpha=0.5)
                        ax.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
                        ax.set_yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
                        plt.legend()

                        wandb.log({"beta_hist": wandb.Image(f_hist)})

                    except:
                        pass

                    try:
                        f_beta, ax = plt.subplots()
                        ax.scatter(tbeta, pbeta)
                        ax.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
                        ax.set_yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
                        wandb.log({"beta_beta_plots": wandb.Image(f_beta)})

                    except:
                        pass

            wandb.log(metrics)

        # Save best model info
        best_model_info_path = os.path.join(
            best_model_path, "best_model_info.txt")

        with open(best_model_info_path, 'w') as f:
            f.write(f"Best model iteration: {best_iteration}\n")
            f.write(f"Best validation loss: {best_val_loss:.6f}\n")

        config_save_path = os.path.join(best_model_path, "config.yaml")
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f)

    finally:
        # At the very end of the function
        wandb.finish()


if __name__ == "__main__":
    # import glob
    # Loop over configs
    # Load config file
    from copy import deepcopy

    base_config_file_path = "config_files/summary_statistics/LSTM/marginal/base_config_rev.yaml"

    with open(base_config_file_path, 'r') as f:
        base_config = yaml.safe_load(f)

    model_name = base_config['model_config']['model_name']
    configurations = []

    if 'LSTM' in base_config_file_path:
        assert model_name == 'LSTMModel'

        for lstm_hidden_size in (16, 32, 48):
            for num_lstm_layers in (2, 3, 1):
                for linear_layer_sizes in ([32, 16, 8], [25, 16, 8, 4], [48, 24, 12, 4], [16, 8, 4, 2]):

                    for mean_aggregation in (False,):  # True):
                        for dropout_rate in (0.075,):  # , 0.1, 0.2):
                            for lr in (0.005, 0.0005):

                                if (num_lstm_layers <= 2 or lstm_hidden_size < 64) and (linear_layer_sizes[0] <= 2 * lstm_hidden_size) and (dropout_rate < 0.1 or lstm_hidden_size >= 64):

                                    config_to_use = deepcopy(base_config)
                                    config_to_use['model_config'] = {'model_name': model_name,
                                                                     'lstm_hidden_size': lstm_hidden_size,
                                                                     'num_lstm_layers': num_lstm_layers,
                                                                     'linear_layer_sizes': linear_layer_sizes,
                                                                     'mean_aggregation': mean_aggregation,
                                                                     'final_output_size': base_config['model_config']['final_output_size'],
                                                                     'dropout_rate': dropout_rate,
                                                                     'with_theta': False
                                                                     }
                                    config_to_use['optimizer']['lr'] = lr
                                    config_to_use['prng_key'] = np.random.randint(
                                        1, 10**5)

                                    configurations.append(config_to_use)

    elif 'CNN' in base_config_file_path:
        assert model_name == 'CNN'
        config_to_use = deepcopy(base_config)

        for max_lag in (30, 35, 40):
            for conv_channels in ([16, 32, 16], [64, 32, 16, 8]):
                for conv_kernels in ([15, 5], [25, 15], [35, 10]):
                    for dropout_rate in (0.025, 0.15):
                        for fc_sizes in ([32, 16, 8], [64, 32, 16], [48, 24, 12, 6]):
                            for lr in (0.0025, 0.0005):

                                config_to_use = deepcopy(base_config)

                                config_to_use['model_config'] = {'model_name': model_name,
                                                                 'max_lag': max_lag,
                                                                 'conv_channels': conv_channels,
                                                                 'fc_sizes': fc_sizes,
                                                                 'final_output_size': base_config['model_config']['final_output_size'],
                                                                 'conv_kernels': conv_kernels,
                                                                 'dropout_rate': dropout_rate,
                                                                 'with_theta': False
                                                                 }
                                config_to_use['optimizer']['lr'] = lr
                                config_to_use['prng_key'] = np.random.randint(
                                    1, 10**5)

                                configurations.append(config_to_use)

    elif 'Transformer' in base_config_file_path:
        assert model_name == 'TimeSeriesTransformerBase'

        for hidden_size in (32, 64, 16):
            for num_heads in (3, 2):
                for num_layers in (3, 1):
                    for mlp_dim in (32, 48, 16):
                        for linear_layer_sizes in ([32, 16, 6], [64, 32, 16, 6], [16, 8, 4]):
                            for dropout_rate in (0.05,):
                                for lr in (0.00005, 0.000005):
                                    for freq_attention in (True, False):

                                        config_to_use = deepcopy(base_config)
                                        config_to_use['model_config'] = {'model_name': model_name,
                                                                         'hidden_size': hidden_size,
                                                                         'num_heads': num_heads,
                                                                         'num_layers': num_layers,
                                                                         'final_output_size': base_config['model_config']['final_output_size'],
                                                                         'mlp_dim': mlp_dim,
                                                                         'linear_layer_sizes': linear_layer_sizes,
                                                                         'dropout_rate': dropout_rate,
                                                                         'with_theta': False,
                                                                         'freq_attention': freq_attention
                                                                         }
                                        config_to_use['optimizer']['lr'] = lr
                                        config_to_use['prng_key'] = np.random.randint(
                                            1, 10**5)

                                        configurations.append(config_to_use)


############# RUN THROUGH TEH CONFIGURATIONS WHILE DEALING WITH STOPPED RUNS ############
# Run through all configurations with improved handling
for config_idx, config in enumerate(configurations):
    print(f"Starting configuration {config_idx+1}/{len(configurations)}")

    # Make sure wandb is clean before starting
    try_to_close_wandb()

    try:
        success = train_and_evaluate(config)

        # Check if training was stopped early (None return value)
        if success is None:
            print(
                f"Configuration {config_idx+1} was manually stopped. Moving to next configuration.")
            # Extra delay after a manual stop to ensure clean startup for next run
            time.sleep(5)
            continue

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected in main loop.")
        # Try to clean up wandb
        try:
            try_to_close_wandb()

            continue

        except:
            pass

    except Exception as e:
        print(f"Error with configuration {config_idx+1}: {e}")
        print("Continuing to next configuration")

    # Short delay before next configuration
    time.sleep(3)

print("All configurations completed or program interrupted")
