# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from src.utils.ecdf_distances_from_samples import check_samples, wasserstein_1_analytical,\
    kolmogorov_smirnov_uniform, cramer_von_mises_uniform, anderson_darling_uniform
import matplotlib.pyplot as plt


def load_NRE_within_TRE_ranks(tre_type, seq_len, N, cal_type):

    beta_path = os.path.join(os.getcwd(), 'models_and_simulated_datasets', 'classifiers', 
                             'per_classifier_coverage_check', tre_type)
    uncal_ranks = np.load(os.path.join(
        beta_path, f'{tre_type}_uncal_ranks_seq_len_{seq_len}_N_{N}.npy'))
    cal_ranks = np.load(os.path.join(
        beta_path, f'{tre_type}_cal_ranks_cal_type_{cal_type}_seq_len_{seq_len}_N_{N}.npy'))

    return uncal_ranks, cal_ranks


def summarize_ecdf_metrics(ranks):

    return [wasserstein_1_analytical(ranks)] + list(kolmogorov_smirnov_uniform(ranks)[:2]) + \
        [cramer_von_mises_uniform(
            ranks), np.nan] + [anderson_darling_uniform(ranks, True)[1]['ad_alt'], np.nan]


def compare_uncal_cal_ranks(uncal_ranks, cal_ranks):

    columns = ['w1', 'ks', 'p-ks', 'cvm', 'p-cvm', 'ad', 'p-ad']
    rows = ['uncal', 'cal']
    data = [summarize_ecdf_metrics(
        uncal_ranks), summarize_ecdf_metrics(cal_ranks)]
    return pd.DataFrame(data, columns=columns, index=rows)


if __name__ == '__main__':

    d = dict()
    N = 128
    plot_difference = True

    colors = {'acf': '#1f77b4', 'beta': '#ff7f0e',
              'mu': '#2ca02c', 'sigma': '#d62728'}

    seq_lengths = [1000, 1500, 2000]
    calibration_type = 'beta'

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True,
                             constrained_layout=True)

    legend_elements_uncal = []
    legend_elements_cal = []

    # Loop through sequence lengths (outer loop for subplot selection)
    for i, seq_len in enumerate(seq_lengths):
        ax = axes[i]  # Select the correct subplot

        for tre_type in ('acf','beta', 'mu', 'sigma'): 
            uncal_ranks, cal_ranks = load_NRE_within_TRE_ranks(
                tre_type, seq_len, N, calibration_type)

            d[(tre_type, seq_len)] = compare_uncal_cal_ranks(
                uncal_ranks, cal_ranks)
            assert len(uncal_ranks) == len(cal_ranks)

            num_ranks = len(uncal_ranks)
            sorted_uncal_ranks = np.sort(uncal_ranks)
            sorted_cal_ranks = np.sort(cal_ranks)
            ecdf = np.arange(1, num_ranks + 1) / num_ranks

            if plot_difference:
                # Uncalibrated: solid line, lighter alpha
                # Uncalibrated: thinner and more transparent
                line_uncal = ax.plot(sorted_uncal_ranks, ecdf - sorted_uncal_ranks,
                                     label=f'{tre_type} uncal',
                                     color=colors[tre_type],
                                     linestyle='-',
                                     linewidth=1.2,
                                     alpha=0.4)[0]

                # Calibrated: thicker and more opaque
                line_cal = ax.plot(sorted_cal_ranks, ecdf - sorted_cal_ranks,
                                   label=f'{tre_type} cal',
                                   color=colors[tre_type],
                                   linestyle='--',
                                   linewidth=2.5,
                                   alpha=1.0)[0]

                # Store lines for legend (only from first subplot)
                if i == 0:
                    legend_elements_uncal.append(line_uncal)
                    legend_elements_cal.append(line_cal)

            else:
                raise ValueError("plot_difference must be True")

        # Reference line at y=0 for each subplot
        ax.plot(np.linspace(0, 1, 100), np.zeros(100),
                color='black', alpha=0.4, linewidth=1, linestyle=':')

        # Improve the plot styling
        ax.set_title(rf'$k$={seq_len}', fontsize=13, pad=10)

        # Only add x-label to the middle plot
        if i == 1:
            ax.set_xlabel(r'Theoretical coverage level $\alpha$', fontsize=13)

        # Only add y-label to the leftmost plot
        if i == 0:
            ax.set_ylabel(
                r'Coverage deviation $\mathcal{C}_{\alpha} - \alpha$', fontsize=13)

        # Add subtle grid
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=13)

    legend_elements = legend_elements_uncal + legend_elements_cal
    labels = [
        'ACF uncal', r'$\beta$ uncal', r'$\mu$ uncal', r'$\sigma$ uncal',
        'ACF cal',   r'$\beta$ cal',   r'$\mu$ cal',   r'$\sigma$ cal'
    ]

    axes[-1].legend(legend_elements, labels,
                    loc='lower right',
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                    fontsize=12,
                    framealpha=0.95,
                    handlelength=2.5,
                    handletextpad=0.6,
                    ncol=2)

    fig.suptitle(
        'Coverage Comparison for individual classifiers within TRE', fontsize=14)

    path_to_save = os.path.join(os.getcwd(), #'models_and_simulated_datasets', 'classifiers', 'TRE_full_trawl',
                                #'selected_models', 'per_classifier_coverage_check',
                                #'tre_classifiers_calibration_comparison.pdf'
                                'src','visualisations','Figure4_bottom.pdf')
    plt.savefig(path_to_save, bbox_inches='tight', pad_inches=0.05, dpi=900)
    plt.show()
