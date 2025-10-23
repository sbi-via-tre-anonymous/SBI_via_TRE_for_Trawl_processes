import pandas as pd
import numpy as np
import os
import pickle


def get_tre_or_nre_results(nre, tre, seq_len, num_lags):

    assert (nre and not tre) or (tre and not nre)

    if tre:
        folder_path = os.path.join(
            os.getcwd(), 'TRE', f'TRE_results_seq_len_{seq_len}')

    elif nre:
        folder_path = os.path.join(
            os.getcwd(), 'NRE', f'NRE_results_seq_len_{seq_len}')

    else:
        raise ValueError

    acf_path = os.path.join(
        folder_path, f'MLE_acf_estimation_error_{seq_len}_{num_lags}.npy')
    mar_path = os.path.join(
        folder_path, f'MLE_marginal_estimation_error_{seq_len}.npy')

    acf = np.load(acf_path, allow_pickle=True).item()
    mar = np.load(mar_path, allow_pickle=True).item()

    return acf, mar


def get_GMM_results(seq_len, num_lags):

    folder_path = os.path.join(os.getcwd(), 'GMM', f'GMM_results_seq_len_{seq_len}')
    acf_path = os.path.join(
        folder_path, f'GMM_acf_estimation_error_{seq_len}_{num_lags}.npy')
    mar_path = os.path.join(
        folder_path, f'GMM_marginal_estimation_error_{seq_len}.npy')

    gmm_acf = np.load(acf_path, allow_pickle=True).item()
    gmm_mar = np.load(mar_path, allow_pickle=True).item()

    return gmm_acf, gmm_mar


def get_nbe_results(seq_len, num_lags):#, nbe_type):
    folder_path = os.path.join(
        os.getcwd(), 'NBE', f'NBE_results_seq_len_{seq_len}')# , nbe_type, f'NBE_results_seq_len_{seq_len}')

    acf_path = os.path.join(
        folder_path, f'NBE_acf_estimation_error_{seq_len}_{num_lags}.npy')
    mar_path = os.path.join(
        folder_path, f'NBE_marginal_estimation_error_{seq_len}.npy')

    acf = np.load(acf_path, allow_pickle=True).item()
    mar = np.load(mar_path, allow_pickle=True).item()

    return acf, mar


def process_dict(d_acf, d_mar, rev):

    data_acf = [d_acf['L1_acf'], d_acf['L2_acf']]
    data_mar = [item for pair in zip(
        d_mar['marginal_MAEs'], d_mar['marginal_MSEs']) for item in pair]

    if rev:
        return np.array(data_acf + data_mar + [d_mar['forward_kl'], d_mar['rev_kl']])

    else:
        return np.array(data_acf + data_mar + [d_mar['forward_kl']])


def get_point_estimators_sumary_statistics(seq_len, num_lags, rev):

    # get GMM data
    gmm_acf, gmm_mar = get_GMM_results(seq_len, num_lags)
    data_gmm = process_dict(gmm_acf, gmm_mar, rev)

    # get NRE data
    nre_acf, nre_mar = get_tre_or_nre_results(nre=True, tre=False,
                                              seq_len=seq_len, num_lags=num_lags)

    data_nre = process_dict(nre_acf, nre_mar, rev)

    # get TRE data
    tre_acf, tre_mar = get_tre_or_nre_results(nre=False, tre=True,
                                              seq_len=seq_len, num_lags=num_lags)

    data_tre = process_dict(tre_acf, tre_mar, rev)

    # get NBE data
    direct_nbe_acf, direct_nbe_mar = get_nbe_results(
        seq_len, num_lags)
    direct_data_nbe = process_dict(direct_nbe_acf, direct_nbe_mar, rev)
    
    # extra cod for supplementary material
    # not for the main body of the paper
    #
    #kl_nbe_acf, kl_nbe_mar = get_nbe_results(
    #    seq_len, num_lags, 'kl')
    #kl_data_nbe = process_dict(kl_nbe_acf, kl_nbe_mar, rev)
    #
    #rev_nbe_acf, rev_nbe_mar = get_nbe_results(
    #    seq_len, num_lags, 'rev')
    #rev_data_nbe = process_dict(rev_nbe_acf, rev_nbe_mar, rev)
    #
    #sym_nbe_acf, sym_nbe_mar = get_nbe_results(
    #    seq_len, num_lags, 'sym')
    #sym_data_nbe = process_dict(sym_nbe_acf, sym_nbe_mar, rev)

    # put the data together
    #
    #
    row_names = ['GMM', 'NRE', 'TRE', 'NBE_direct']#,
                 #'NBE_kl', 'NBE_rev', 'NBE_sym']
    columns_tuples = [
        ('acf', 'mean L1'),  # r'mean $$L^1$$'),
        ('acf', 'mean L2'),  # r'mean $$L^2$$'),
        ('marginal', 'MAE mu'),  # r'MAE $$\mu$$'),
        ('marginal', 'MSE mu'),  # r'MSE $$\mu$$'),
        ('marginal', 'MAE sigma'),  # r'MAE $$\sigma$$'),
        ('marginal', 'MSE sigma'),  # r'MSE $$\sigma$$'),
        ('marginal', 'MAE beta'),  # r'MAE $$\beta$$'),
        ('marginal', 'MSE beta'),  # r'MSE $$\beta$$'),
        ('marginal', 'mean KL'),  # Single-level column (KL) with empty top-level
    ]

    if rev:
        columns_tuples = columns_tuples + [('marginal', 'rev_KL')]

    columns = pd.MultiIndex.from_tuples(columns_tuples)

    # ,
    data = np.stack([data_gmm, data_nre, data_tre,  direct_data_nbe])
                    #, kl_data_nbe, rev_data_nbe, sym_data_nbe])

    df = pd.DataFrame(data=data, index=row_names, columns=columns)

    df.loc[:, ('acf', 'mean L1')] = df.loc[:, ('acf', 'mean L1')] * num_lags
    df.loc[:, ('acf', 'mean L2')] = df.loc[:,
                                           ('acf', 'mean L2')] * num_lags**0.5

    return df


if __name__ == '__main__':
    num_lags = 35
    rev = False

    df_2000 = get_point_estimators_sumary_statistics(2000, num_lags, rev)
    df_1500 = get_point_estimators_sumary_statistics(1500, num_lags, rev)
    df_1000 = get_point_estimators_sumary_statistics(1000, num_lags, rev)


    df = pd.concat([df_1000, df_1500, df_2000], keys=[1000, 1500, 2000])
    # df.to_latex(f"my_table_{num_lags}.tex", float_format="%.3f")
    #df.to_latex(f"my_table_{num_lags}_{rev}.tex",
    #            float_format=lambda x: f"{x:.3f}")
    df.to_csv(f"my_table_{num_lags}_{rev}.csv",
              float_format=lambda x: f"{x:.3f}")