# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 21:12:08 2025

@author: dleon
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from get_ecdf_statistics import summarize_ecdf_metrics, compare_uncal_cal_ranks


def load_uncal_and_beta_cal_NRE_TRE_ranks(classifier_type, seq_len):

    base = os.path.join(os.getcwd(), 'models_and_simulated_datasets', 'classifiers',
                        'coverage_check_ranks_NRE_and_TRE')

    if classifier_type == 'NRE':
        uncal_ranks = np.load(os.path.join(
            base, f'{classifier_type}_{seq_len}no_calibration.npy'))
        cal_ranks = np.load(os.path.join(
            base, f'{classifier_type}_{seq_len}beta_calibration.npy'))

    elif classifier_type == 'TRE':
        uncal_ranks = np.load(os.path.join(
            base, f'seq_sampling_{classifier_type}_{seq_len}_None_128_160.npy'))
        cal_ranks = np.load(os.path.join(
            base, f'seq_sampling_{classifier_type}_{seq_len}_beta_128_160.npy'))

    return uncal_ranks, cal_ranks


if __name__ == '__main__':

    d = dict()
    plot_difference = True

    # Define colors for the two classifier types
    colors = {'NRE': '#1f77b4', 'TRE': '#ff7f0e'}
    seq_lengths = [1000, 1500, 2000]

    # Create figure with 3 subplots using constrained_layout
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True,
                             constrained_layout=True)

    # Store line objects for custom legend
    legend_elements = []

    for i, seq_len in enumerate(seq_lengths):
        ax = axes[i]

        for classifier_type in ('NRE', 'TRE'):
            uncal_ranks, cal_ranks = load_uncal_and_beta_cal_NRE_TRE_ranks(
                classifier_type, seq_len)
            d[(classifier_type, seq_len)] = compare_uncal_cal_ranks(
                uncal_ranks, cal_ranks)

            num_ranks_uncal = len(uncal_ranks)
            num_ranks_cal = len(cal_ranks)
            sorted_uncal_ranks = np.sort(uncal_ranks)
            sorted_cal_ranks = np.sort(cal_ranks)
            ecdf_uncal = np.arange(0, num_ranks_uncal) / num_ranks_uncal
            ecdf_cal = np.arange(0, num_ranks_cal) / num_ranks_cal

            if seq_len == 1500 and classifier_type == 'TRE':
                print('uncal ranks: ', np.mean(
                    np.abs(sorted_uncal_ranks - ecdf_uncal))**2)
                print('cal ranks: ', np.mean(
                    np.abs(sorted_cal_ranks - ecdf_cal))**2)

            if plot_difference:
                # Uncalibrated: solid line, lighter alpha
                line_uncal = ax.plot(sorted_uncal_ranks, ecdf_uncal - sorted_uncal_ranks,
                                     label=f'{classifier_type} uncal',
                                     color=colors[classifier_type],
                                     linestyle='-',
                                     linewidth=2,
                                     alpha=0.6)[0]

                # Calibrated: prominent dashed line, full alpha
                line_cal = ax.plot(sorted_cal_ranks, ecdf_cal - sorted_cal_ranks,
                                   label=f'{classifier_type} cal',
                                   color=colors[classifier_type],
                                   linestyle='--',  # Use simple dashed style
                                   linewidth=2.5,
                                   alpha=0.9)[0]

                # Store lines for legend (only from first subplot to avoid duplicates)
                if i == 0:
                    legend_elements.append(line_uncal)
                    legend_elements.append(line_cal)

                # ax.axhline(y=0.025, color='gray', linestyle=':',
                #           alpha=0.3, linewidth=0.8)
                # ax.axhline(y=-0.025, color='gray', linestyle=':',
                #           alpha=0.3, linewidth=0.8)

        # Reference line at y=0
        ax.plot(np.linspace(0, 1, 100), np.zeros(100),
                color='black', alpha=0.4, linewidth=1, linestyle=':')

        # Styling improvements
        ax.set_title(rf'$k$={seq_len}', fontsize=13, pad=10)

        # Only add x-label to the middle plot
        if i == 1:  # Middle subplot (index 1)
            ax.set_xlabel(r'Theoretical coverage level $\alpha$', fontsize=13)

        # Only add y-label to the leftmost plot
        if i == 0:
            ax.set_ylabel(
                r'Coverage deviation $\mathcal{C}_{\alpha} - \alpha$', fontsize=13)

        # Add subtle grid
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=13)

    # Create a custom legend on the rightmost subplot with proper line styles
    labels = ['NRE uncalibrated', 'NRE calibrated',
              'TRE uncalibrated', 'TRE calibrated']
    axes[-1].legend(legend_elements, labels,
                    loc='lower right',
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                    fontsize=12,
                    framealpha=0.95,
                    handlelength=3.0,  # Make legend lines longer to show dash pattern
                    handletextpad=0.8)  # Add some space between line and text

    fig.suptitle('Coverage Comparison for NRE and TRE', fontsize=14)

    path_to_save = os.path.join(os.getcwd(), 'src', 'visualisations',
                                'Figure4_top.pdf'
                                #'models_and_simulated_datasets', 'classifiers',
                                #'coverage_check_ranks_NRE_and_TRE',
                                #'calibration_comparison.pdf'
                                )
    plt.savefig(path_to_save, bbox_inches='tight', pad_inches=0.05, dpi=900)
    plt.show()  # Add this to see the plot
