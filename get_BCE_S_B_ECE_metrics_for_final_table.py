import pandas as pd
import numpy as np
import os
from get_ecdf_statistics import load_NRE_within_TRE_ranks, compare_uncal_cal_ranks
from NRE_TRE_coverage_figures_and_ecdf_metrics import load_uncal_and_beta_cal_NRE_TRE_ranks


def get_BCE_S_B_ECE_classifier_metrics_for_individual_NRE_within_TRE(seq_len, calibration_type):

    # d = {1000: 9, 1500: 7, 2000: 5}
    d = {1000: 5, 1500: 5, 2000: 5}
    tre_types = ('acf', 'beta', 'mu', 'sigma')
    base_path = os.path.join(os.getcwd(), 'models_and_simulated_datasets', 'classifiers')

    # TRE_ paths
    df_list = []
    for tre_type, tre_type in zip(tre_types, tre_types):
        TRE_path = os.path.join(base_path, 'TRE_full_trawl', 'selected_models',
                                tre_type,  'validation_results')
        df = pd.read_excel(os.path.join(TRE_path, f'BCE_S_B_{seq_len}_{tre_type}.xlsx'),
                           header=0).set_index('Unnamed: 0')
        df.index.name = None
        df = df.rename(columns={calibration_type: 'cal'})

        # Then subset
        df_list.append(
            df.loc[['BCE', 'S', 'B', 'ACE_t'], ['uncal', 'cal']])

    return pd.concat(df_list, keys=tre_types, axis=1)


if __name__ == '__main__':

    seq_lengths = (1000, 1500, 2000)
    ecdf_metric_to_use = 'w1'
    tre_types = ('acf', 'beta', 'mu', 'sigma')
    N = 128
    calibration_type = 'beta'
    df_list_1 = []  # posterior ecdf deviation
    df_list_2 = []  # BCE S B ECE

    # index_NRE_TRE = [('NRE', 'uncal'),('NRE',   'cal'), ('TRE', 'uncal'),('TRE',   'cal')]
    # index_acf

    # deviations from uniiform cdf of posterior samples
    for seq_len in seq_lengths:

        # NRE TRE: uncal first
        NRE_ = compare_uncal_cal_ranks(
            *load_uncal_and_beta_cal_NRE_TRE_ranks('NRE', seq_len))[ecdf_metric_to_use].values
        TRE_ = compare_uncal_cal_ranks(
            *load_uncal_and_beta_cal_NRE_TRE_ranks('TRE', seq_len))[ecdf_metric_to_use].values

        df_ = pd.DataFrame(np.concatenate([NRE_, TRE_]).reshape(1, -1),
                           columns=pd.MultiIndex.from_product(
                               [['NRE', 'TRE'], ['uncal', 'cal']]),
                           index=[ecdf_metric_to_use])

        # individual NRES within TRE
        l = []

        for tre_type in tre_types:
            l.append(compare_uncal_cal_ranks(
                *load_NRE_within_TRE_ranks(tre_type, seq_len, N, calibration_type))[ecdf_metric_to_use].values)

        df__ = pd.DataFrame(np.concatenate(l).reshape(1, -1),
                            columns=pd.MultiIndex.from_product(
                                [tre_types,  ['uncal', 'cal']]),
                            index=[ecdf_metric_to_use])

        df_list_1.append(pd.concat([df_, df__], axis=1))

    # BCE S B ECE F
    for seq_len in seq_lengths:
        # BCE, S, B, ECE_f for NRE and TRE
        path_2 = os.path.join(os.getcwd(), #'models_and_simulated_datasets', 'classifiers',                            
                              'src','visualisations','aux_metrics',
                              f'BCE_S_B_for_NRE_and_TRE_{seq_len}_{calibration_type}.xlsx')
        df_ = pd.read_excel(path_2, header=[0, 1],  index_col=0)

        # BCE, S, B, ECE_f for individual NRES within TRE
        df__ = get_BCE_S_B_ECE_classifier_metrics_for_individual_NRE_within_TRE(
            seq_len, calibration_type)

        df_list_2.append(pd.concat([df_, df__], axis=1))

    df_list = []
    for i in (0, 1, 2):
        df_list.append(pd.concat([df_list_2[i], df_list_1[i]], axis=0))

    df = pd.concat(df_list, keys=seq_lengths, axis=0)

#    # Reorder columns
#    new_order = [('acf', 'uncal'), ('acf', 'cal'),
#                 ('beta', 'uncal'), ('beta', 'cal'),
#                 ('mu', 'uncal'), ('mu', 'cal'),
#                 ('sigma', 'uncal'), ('sigma', 'cal'),
#                 ('NRE', 'uncal'), ('NRE', 'cal'),
#                 ('TRE', 'uncal'), ('TRE', 'cal')]
#    df = df[new_order]

    # Reorder rows
    metric_order = ['BCE', 'S', 'B', ecdf_metric_to_use, 'ACE_t']
    df = df.reindex(metric_order, level=1)
    df = df.rename(columns={'ACE_t': 'ECE_t'}) #use adaptive discretisation for ece


    excel_save_path = os.path.join(os.getcwd(), #'models_and_simulated_datasets', 'classifiers',
                                   #f'final_table_{ecdf_metric_to_use}_{calibration_type}.xlsx'
                                   'src','visualisations','Table2.xlsx')
    tex_save_path = os.path.join(os.getcwd(), #'models_and_simulated_datasets', 'classifiers',
                                 #f'final_table_{ecdf_metric_to_use}_{calibration_type}.tex'
                                 'src','visualisations','Table2.tex')
    df.to_excel(excel_save_path)
    df.to_latex(tex_save_path, float_format='%.3f')
