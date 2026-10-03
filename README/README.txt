Computationally expensive operations are conducted on a GPU, with all remaining operations on a Windows CPU workstation. All computer code is in Python and located inside sbi_via_tre_for_trawl_processes, 
which shall be our base folder for the purposes of this workflow. All figures and tables are saved in src\visualisations. See the last section of this README file for datasets download instructions.


**Simulation Study reproducibility**

Figure 1: 

Neural networks are trained using train_classifier_final.py, with JAX on a GPU. Multiple networks are trained, and the best-performing one is selected. The corresponding weights and biases are saved in 

models_and_simulated_datasets\classifiers\NRE_full_trawl and 
models_and_simulated_datasets\classifiers\TRE_full_trawl\selected_models. 

These weights and biases are saved as params.pkl or params_iter_x.pkl. Performance metrics (BCE,S,B, Acccuracy) are evaluated during training on a holdout dataset and logge at https://wandb.ai/.
Download metrics are stored at models_and_simulated_datasets\classifiers\TRE_full_trawl\metric_lots_during_training. The script plot_Figure1.py produces and saves Figure1.pdf in src\visualisations. 

Figure 2:

The script plot_Figure2.py produces and saves Figure2.pdf in src\visualisations. 

Table 1:

This first requires point estimators for NRE, TRE, NBE and GMM. In this first step, we proceed as follows: 
  
  - NRE: run get_MLE_point_estimators.py for seq_len = 1000, 1500 and 2000. The default seq_len value is set immediately below  if __name__ == '__main__'
  - TRE: run get_MLE_point_estimators.py for seq_len = 1000, 1500 and 2000, but comment NRE_path and uncomment TRE_path, which are immediately below   if __name__ == '__main__'
  - GMM: perform GMM separately for the ACF and marginal parameters; run src\utils\parallel_weighted_ACF_GMM.py and src\utils\parallel_weighted_GMM_marginal.py again with 
	seq_len = 1000, 1500, 2000.
  - NBE: run get_NBE_point_estimators.py

The results are saved in models_and_simulated_datasets\point_estimators\..., in 4 subfolders called NRE, TRE, GMM and NBE. Then run analyze_point_estimators.py with seq_len = 1000, 15000 and 2000 and for all of the four options above: NRE, TRE, GMM and NBE. This produces the values for Table 1. Then run models_and_simulated_datasets\point_estimators\produce_Table1.py to produce and save Table1 in src\visualisations.

Note: NBEs are trained with train_summary_statistics.py; model weights and biases for the best network are provided in models_and_simulated_datasets\NBE_ACF and models_and_simulated_datasets\NBE_marginal, based on which the calculations are performed. The script get_NBE_point_estimators.py is fast.

Note: The NRE, TRE and GMM take long but are feasible on a workstation (likely >8h even with parallel processing). Faster results can be obtained by setting num_trawls_to_use = 10**4 to a smaller number, 
although results will not match exactly.  

For Figure 3 and Table 2, we need to first run calibrate.py. This requires a GPU to evaluate the classifier neural networks over the calibration and validation datasets and then produces log_r, pred_prob_ and Y, which are then saved inside 
  
  - models_and_simulated_datasets\classifiers\NRE_full_trawl\best_model\** where ** is 'calibrations_results' and 'validation_results'
  - models_and_simulated_datasets\classifiers\TRE_full_trawl\selected_models\*\** where * is 'acf' 'beta' 'mu' or 'sigma' and ** is 'calibrations_results' and 'validation_results'

If these are deleted, calibrate.py will compute them, although this calculation is probably not feasible on CPU. All in all, calibrate.py runs the calbration scripts and outputs the parameters for the beta_calibration and isotonic regression, which are used later on; it also creates csv files inside 'validation_results' which are used for Table 2.

Figure 3:

We first need to generate posterior samples, which we also use for the W metric row in Table 2. Please run
 
  - coverage_check_NRE_within_TRE.py for seq_len = 1000, 1500 and 2000 and tre_type = sigma, beta and mu
  - coverage_check_acf_2_param_NRE.py for seq_len = 1000, 1500 and 2000                                  
  - sequential_posterior_sampling.py for seq_len = 1000, 1500 and 2000

These are run on a GPU, but are also feasible on a CPU, especially if N is set tot 64 instead of 128 and num_samples is set to a lower value in the last two scripts.

Then, for the bottom figure (component NREs within TRE): run get_ecdf_statistics.py to save to src\visualisations\Figure3bottom
For the top figure (NRE vs TRE): run NRE_TRE_coverage_figures_and_ecdf_metrics.py to save to src\visualisations\Figure3top

Note: For the top figure, we require NRE posterior samples to benchmark against the TRE. These are provided in models_and_simulated_datasets\classifiers\coverage_check_ranks_NRE_and_TRE\NRE_*,
and are produced and saved on the HPC with 128 nodes by running posterior_sampling_distributed.py. Using a GPU doesn't seem to help. I suspect the sequential nature of the NUTS sampling
algorithm favors CPU over GPU. Other non-adaptive samplers which are GPU friendly do not produce reasonable effective sample sizes (ESS). 


Table 2:

Run src\utils\validate_BCE_S_B_metrics_for_TRE_and_NRE.py and then get_BCE_S_B_ECE_metrics_for_final_table.py to produce and save results tot src\visualisations\Table2.xlsx.
Most of the metrics for Table 2 are produced by calibrate.py, ran a few steps above. The extras needed are the W1 metric based on the posterior samples and the combined (not individual)
TRE metrics. The latter are produced by validatet_BCE_S_B_metrtics_for_TRE_and_NRE.py



**Application reproducibility: Figures 4, Figure 5 and Table 3:**


Data is downloaded and made available in application_pre_processing_Figure4\all_years_at_once_electricity_data.csv.It can be downloaded again by running application_pre_processing_Figure4\get_electricity_data_all_years_at_once.py  This script requires an access key from https://www.eia.gov/opendata/.

Then application_pre_processing_Figure4\MSTL_script.py creates and populates the folder application_pre_processing_Figure4\MSTL_results_14 and saves Figure4.pdf in src\visualisations. Finally, application_Figure5_and_Table3.py uses the MSTL output, produces and saves Figure5.pdf and Table3.csv in src\visualisations. This runs fast, even without a GPU.
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


**Required datasets and computationally expensive operations**

  - calibration datasets
  - validation datasets

Download both from https://doi.org/10.5281/zenodo.17425612 and save under models_and_simulated_datasets\calibration_datasets and models_and_simulated_datasets\validation_datasets. Alternatively, they can be generated from scratch by running calibrate.py (GPU required).

  - evaluating the neural network classifiers over the above mentioned calibration and validation datasets

This is also computationally expensive and requires a GPU. We provide some pre-computed files based on which calibrate.py can be ran to do the calibration, as described in the workflow.

Note: If the datasets are not saved on disk, they will be simulated automatically by following the workflow.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
