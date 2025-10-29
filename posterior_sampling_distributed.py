# -*- coding: utf-8 -*-

import sys
import os
import yaml
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from posterior_sampling_utils import run_mcmc_for_trawl, save_results, create_and_save_plots
from src.utils.get_trained_models import load_trained_models_for_posterior_inference as load_trained_models
import distrax


def process_batch(batch_indices, folder_path, true_trawls, true_thetas, wrapper_for_approx_likelihood_just_theta,
                  trawl_process_type, num_samples, num_warmup, num_burnin, num_chains, seed, seq_len, calibration_filename, double_cal):
    """
    Process a batch of trawl indices

    Parameters:
    -----------
    batch_indices : list
        List of trawl indices to process
    folder_path : str
        Path to the model folder
    """
    if 'no_cal' in calibration_filename:

        suffix = 'no_calibration'

    elif ('spline' in calibration_filename) or ('beta' in calibration_filename):

        suffix = calibration_filename[:-4]

    else:

        raise ValueError

    # Create results directory
    results_dir = f"mcmc_results_{trawl_process_type}_{seq_len}" + suffix
    if double_cal:
        results_dir += 'double_cal'
    results_dir = os.path.join(folder_path, results_dir)
    os.makedirs(results_dir, exist_ok=True)

    for idx in batch_indices:
        # Create directory for this trawl
        trawl_dir = os.path.join(results_dir, f"trawl_{idx}")
        os.makedirs(trawl_dir, exist_ok=True)

        # Skip if already completed
        if os.path.exists(os.path.join(trawl_dir, "results.pkl")):
            print(f"Trawl {idx} already processed, skipping...")
            continue

        try:

            assert (double_cal and seq_len < 2500) or (not double_cal)

            if not double_cal:
                # Run MCMC for this trawl
                results, posterior_samples = run_mcmc_for_trawl(
                    trawl_idx=idx,
                    true_thetas=true_thetas,
                    approximate_log_likelihood_to_evidence_just_theta=wrapper_for_approx_likelihood_just_theta(
                        jnp.reshape(true_trawls[idx], (1, -1))),
                    seed=seed + idx**2,
                    num_samples=num_samples,
                    num_warmup=num_warmup,
                    num_burnin=num_burnin,
                    num_chains=num_chains
                )
            else:

                _approx_log_like_ = wrapper_for_approx_likelihood_just_theta(
                    jnp.reshape(true_trawls[idx], (1, -1)))

                if double_cal:
                    double_cal_params_path = os.path.join(
                        folder_path, f'overall_NRE_spline_cal_of_TRE_{seq_len}', 'double_cal_spline_params.npy')
                    double_cal_spline_params = np.load(double_cal_params_path)
                    spline = distrax.RationalQuadraticSpline(boundary_slopes='identity',
                        params=double_cal_spline_params, range_min=0.0, range_max=1.0)

                @jax.jit
                def double_cal_approx_log_like(theta):
                    uncal = _approx_log_like_(theta)
                    cal = spline.forward(jax.nn.sigmoid(uncal))
                    return jax.scipy.special.logit(cal)

                # Run MCMC for this trawl
                results, posterior_samples = run_mcmc_for_trawl(
                    trawl_idx=idx,
                    # true_trawls=true_trawls,
                    true_thetas=true_thetas,
                    approximate_log_likelihood_to_evidence_just_theta=double_cal_approx_log_like,
                    seed=seed + idx**2,
                    num_samples=num_samples,
                    num_warmup=num_warmup,
                    num_burnin=num_burnin,
                    num_chains=num_chains
                )

            # Add true theta to results
            results['true_theta'] = true_thetas[idx].tolist()
            results['true_trawl'] = true_trawls[idx].tolist()

            # Save results
            save_results(results, os.path.join(trawl_dir, "results.pkl"))

            # Save memory by clearing results
            del results

            print(f"Completed trawl {idx}")

        except Exception as e:
            print(f"Error processing trawl {idx}: {str(e)}")
            # Save the error to a file
            with open(os.path.join(trawl_dir, "error.txt"), 'w') as f:
                f.write(f"Error processing trawl {idx}: {str(e)}")


def main():
    if len(sys.argv) < 4:
        print("Usage: python script_name.py <start_idx> <end_idx> <task_id>")
        sys.exit(1)

    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    task_id = int(sys.argv[3])
    total_tasks = 128  # Total number of cores/tasks
    seq_len = 1000
    calibration_filename = 'beta_calibration_1000.pkl'
    num_rows_to_load = 300

    ########################################################
    # if changing seq_len, also change calibration_filename#
    # current version allows using seq_len = 1000 and
    # calibration for 1500 for testing purposes
    # results come out with both numbers in the filename
    # for sanity checks
    ########################################################

    print(
        f"DEBUG: Python received args: start_idx={start_idx}, end_idx={end_idx}, task_id={task_id}")

    # Calculate the trawl indices for this task
    total_trawls = end_idx - start_idx + 1
    all_ranges = np.array_split(range(start_idx, end_idx + 1), total_tasks)
    print(f"DEBUG: Total trawls to process: {total_trawls}")
    print(f"DEBUG: Number of tasks: {total_tasks}")
    print(
        f"DEBUG: Split ranges for all tasks: {[list(r) for r in all_ranges]}")

    indices = all_ranges[task_id]
    print(f"DEBUG: Task {task_id} received indices: {list(indices)}")
    print(f"DEBUG: Length of all_ranges: {len(all_ranges)}")

    if len(indices) == 0:
        print(f"Task {task_id}: No trawls assigned.")
        return

    print(f"Task {task_id}: Processing trawls {indices[0]} to {indices[-1]}")

    # Load configuration
    path_to_NRE_full_trawl_parent_folder = '' # USER: pass this path if you want to rerun experiments
    folder_path = os.pah.join(path_to_NRE_full_trawl_parent_folder,'NRE_full_trawl')
    # alternatively, pass path to 'TRE_full_trawl/selected_models'
    # to do TRE sampling, although this is not featuring in the paper

    # Set up model configuration
    use_tre = 'TRE' in folder_path
    if not (use_tre or 'NRE' in folder_path):
        raise ValueError("Path must contain 'TRE' or 'NRE'")

    use_summary_statistics = 'summary_statistics' in folder_path
    if not (use_summary_statistics or 'full_trawl' in folder_path):
        raise ValueError(
            "Path must contain 'full_trawl' or 'summary_statistics'")

    if use_tre:
        classifier_config_file_path = os.path.join(
            folder_path, 'acf', 'config.yaml')
    else:
        classifier_config_file_path = os.path.join(folder_path, 'config.yaml')

    with open(classifier_config_file_path, 'r') as f:
        a_classifier_config = yaml.safe_load(f)
        trawl_process_type = a_classifier_config['trawl_config']['trawl_process_type']

    # Load dataset
    if use_tre:
      dataset_path = os.path.join(os.path.dirname(os.path.dirname(
          os.path.dirname(folder_path))), 'cal_dataset', f'cal_dataset_{seq_len}')
    else:
      dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
          os.path.dirname(folder_path)))), 'cal_dataset', f'cal_dataset_{seq_len}')
          
    cal_x_path = os.path.join(dataset_path, 'cal_x_joint.npy')
    cal_thetas_path = os.path.join(dataset_path, 'cal_thetas_joint.npy')

    # Load first few rows of cal_x with memory mapping
    cal_x = np.load(cal_x_path, mmap_mode='r')[:num_rows_to_load]

    # Load cal_thetas (adjust if it also needs to be limited)
    cal_thetas = np.load(cal_thetas_path)[:num_rows_to_load]


    # Apply the mask and reshape as before
    true_trawls = cal_x.reshape(-1, seq_len)
    true_thetas = cal_thetas.reshape(-1, cal_thetas.shape[-1])

    # Load approximate likelihood function
    _, wrapper_for_approx_likelihood_just_theta = load_trained_models(
        folder_path, true_trawls[[0], ::-1], trawl_process_type,
        use_tre, use_summary_statistics, calibration_filename
    )

    del cal_x
    

    # MCMC parameters
    num_samples = 25000  # Adjust as needed
    num_warmup = 10000
    num_burnin = 10000
    num_chains = 5
    seed = 97668
    double_cal = False

    # Process assigned batch
    process_batch(
        batch_indices=indices,
        folder_path=folder_path,
        true_trawls=true_trawls,
        true_thetas=true_thetas,
        wrapper_for_approx_likelihood_just_theta=wrapper_for_approx_likelihood_just_theta,
        trawl_process_type=trawl_process_type,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_burnin=num_burnin,
        num_chains=num_chains,
        seed=seed + task_id**2,
        seq_len=seq_len,
        calibration_filename=calibration_filename,
        double_cal=double_cal
    )

    print(f"Task {task_id}: Completed all assigned trawls")


if __name__ == '__main__':
    main()
