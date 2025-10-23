Computationally expensive operations are conducted on a GPU, with remaining operations on a Windows CPU workstation. All computer code is in Python and is located inside 'sbi_via_tre_for_trawl_processes',
which shall be our base folder for the purposes of this workflow. All figures and tables are saved in src\visualisations.


**Required datasets and computationally expensive operations**

  - calibration datasets
  - validation datasets

These can be downloaded from 10.5281/zenodo.17425612 and are to be saved in models_and_simulated_datasets\calibration_datasets and models_and_simulated_datasets\validation_datasets. 
They can also be simulated from scratch by running calibrate.py. This is computationally expensive and should be done on a GPU.

  - evaluating the neural network classifiers over the above mentioned calibration and validation datasets

This is also computationally expensive and requires a GPU, and we provide some pre-computed files based on which calibrate.py can be ran to do the calibration, as we describe below, in this workflow.

Note: If the datasets are not saved on disk, they will be simulated automatically by following the workflow.


**Simulation Study reproducibility**

Figure 2: 

Training the neural networks is done with train_classifier_final.py, with JAX on a GPU. We train multiple networks and select the best one, for which we provide weights and biases in 
models_and_simulated_datasets\classifiers\NRE_full_trawl and models_and_simulated_datasets\classifiers\TRE_full_trawl\selected_models. These weights and biases are saved as params.pkl 
or params_iter_x.pkl. We evaluate the performance of our classifiers during training across a number of metrics (BCE,S,B, Acccuracy) on a holdout dataset. We log these metrics
during training on https://wandb.ai/, and download them from wandb in models_and_simulated_datasets\classifiers\TRE_full_trawl\metric_lots_during_training. The script plot_Figure2.py 
produces and saves Figure2.pdf in src\visualisations. 

Figure 3:

The script plot_Figure3.py produces and saves Figure3.pdf in src\visualisations. 

Table 1:

This first requires point estimators for NRE, TRE, NBE and GMM. In this first step, we proceed as follows: 
  
  - NRE: run get_MLE_point_estimators.py for seq_len = 1000, 1500 and 2000. Currently, seq_len is set to 1000, immediately below if __name__ == '__main__'
  - TRE: run get_MLE_point_estimators.py for seq_len = 1000, 1500 and 2000, but comment NRE_path and uncomment TRE_path, which are immediately below if __name__ == '__main__'
  - GMM: perform GMM separately for the ACF and marginal parameters; run src\utils\parallel_weighted_ACF_GMM.py and src\utils\parallel_weighted_GMM_marginal.py again with seq_len = 1000, 1500, 2000.
  - NBE: run get_NBE_point_estimators.py

The results are saved in models_and_simulated_datasets\point_estimators\..., in 4 subfolders called NRE, TRE, GMM and NBE. Then run analyze_point_estimators.py to get Table 1, which is saved
in src\visualisations.

Note: The NBEs are trained with train_summary_statistics.py; model weights and biases for the best network are provided in models_and_simulated_datasets\NBE_ACF and models_and_simulated_datasets\NBE_marginal,
based on which the calculations are performed. The script get_NBE_point_estimators.py is fast, even on a CPU.

Note: The NRE, TRE and GMM take long but are feasible on a workstation (likely >8h even with parallel processing). Faster results can be obtained by setting num_trawls_to_use = 10**4 to a smaller number,
although results will not match exactly.  

Note: For Figure 4 and Table 2, we need to first run calibrate.py. This requires a GPU to evaluate the classifier neural networks over the calibration and validation datasets and then produces and saves
log_r, pred_prob_ and Y, which are then saved inside 
  
  - models_and_simulated_datasets\classifiers\NRE_full_trawl\best_model\** where ** is 'calibrations_results' and 'validation_results'
  - models_and_simulated_datasets\classifiers\TRE_full_trawl\selected_models\*\** where * is 'acf' 'beta' 'mu' or 'sigma' and ** is 'calibrations_results' and 'validation_results'

If these are deleted, calibrate.py will compute these, although this calculation is probably not feasible on CPU. All in all, calibrate.py runs the calbiration scripts and outputs the parameters for the beta_calibration
and isotonic regression, which are to be used further; it also creates csv files inside 'validation_results' which are used for Table 2.

Figure 4:

top figure (NRE vs TRE):

bottom figure (component NREs within TRE):



Table 2:



**Application reproducibility: Figures 5, Figure 6 and Table 3:**


Data is downloaded and made available in application_pre_processing_Figure5\all_years_at_once_electricity_data.csv.It can be downloaded again by running application_pre_processing_Figure5\get_electricity_data_all_years_at_once.py
This script requires an access key from https://www.eia.gov/opendata/.
Then application_pre_processing_Figure5\MSTL_script.py creates and populates the folder application_pre_processing_Figure5\MSTL_results_14 and saves Figure5.pdf in src\visualisations. Finally, application_Figure6_and_Table3.py
uses the MSTL output, produces and saves Figure6.pdf and Table3.csv in src\visualisations. This runs fast, even without a GPU.