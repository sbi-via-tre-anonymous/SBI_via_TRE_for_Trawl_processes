import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.stats import norminvgauss
import numpy as np
import scienceplots
matplotlib.rcParams['text.usetex'] = False
plt.style.use(['science', 'no-latex'])
# =====================
# Load the CSV
# =====================
df = pd.read_csv("all_years_at_once_electricity_data.csv")
df['period'] = pd.to_datetime(df['period'])
df = df.sort_values('period')


P_trend = 1 #degree of polynomial trend to remove    
K_harmonics = 2 # number of yearly harmonics to remove  
P_year = 365.25 # period of the harmonics
# PP = 2, KK = 2 also seems to work well

output_dir = f"OLS_results_p{P_trend}_k{K_harmonics}" 
os.makedirs(output_dir, exist_ok=True)

def deseasonalise_ols(series, p=P_trend, K=K_harmonics, P_year=365.25):
    y = series.values.astype(float); n = len(y); idx = series.index
    t = np.arange(n, dtype=float)
    doy = idx.dayofyear.values.astype(float); dow = idx.dayofweek.values
    cols = [(t / n) ** j for j in range(p + 1)]
    for k in range(1, K + 1):
        cols += [np.cos(2*np.pi*k*doy/P_year), np.sin(2*np.pi*k*doy/P_year)]
    cols += [(dow == m).astype(float) for m in range(1, 7)]
    X = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    n_tr, n_an = p + 1, 2 * K
    trend  = X[:, :n_tr] @ b[:n_tr]
    annual = X[:, n_tr:n_tr+n_an] @ b[n_tr:n_tr+n_an]
    weekly = X[:, n_tr+n_an:] @ b[n_tr+n_an:]
    resid  = y - X @ b
    wbar = weekly.mean()
    weekly = weekly - wbar
    trend = trend + wbar
    return pd.Series(trend, index=idx), pd.Series(weekly, index=idx), pd.Series(annual, index=idx), pd.Series(resid, index=idx)

all_results = []

for respondent in df['respondent'].unique():
    print(f"Processing respondent: {respondent}")
    
    # Subset
    subdf = df[df['respondent'] == respondent].copy()
    subdf = subdf.set_index('period').sort_index()
    series = subdf['value'].astype(float)
    original_series = series.copy()
    
    # =====================
    # Apply deseazonalization
    # =====================
    trend_, weekly_, annual_, resid_ = deseasonalise_ols(series, p = P_trend, K = K_harmonics, P_year = P_year)

    
    result_df = pd.DataFrame({
        'trend': trend_,
        'seasonal_7': weekly_,
        'seasonal_365': annual_,
        'resid': resid_
    }, index=series.index)
    result_df['respondent'] = respondent
    all_results.append(result_df.reset_index())
    
    respondent_dir = os.path.join(output_dir, respondent)
    os.makedirs(respondent_dir, exist_ok=True)
    
    result_df.to_csv(os.path.join(respondent_dir, f"{respondent}_OLS.csv"))
    
    # =====================
    # New plot: Original vs Trend+Seasonality and Residuals
    # =====================
    trend_seasonality = original_series - resid_
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.25))
    
    # Left plot: Original series and trend+seasonality
    ax1.plot(original_series.index, original_series.values, 'b-', alpha=0.7, label='Original', linewidth=0.8)
    ax1.plot(trend_seasonality.index, trend_seasonality.values, 'r-', alpha=0.75, label='Trend+Seasonality', linewidth=0.4)
    ax1.set_title("Original vs Trend+Seasonality Time Series",fontsize=16)
    ax1.set_xlabel("Date",fontsize=14)
    ax1.set_ylabel("Value",fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Residuals
    ax2.plot(resid_.index, resid_.values, 'b-', linewidth=0.8, label = 'Residuals')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_title("Residuals",fontsize=16)
    ax2.set_xlabel("Date",fontsize=14)
    ax2.set_ylabel("Residuals",fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    
    plt.tight_layout()
    plt.savefig(os.path.join(respondent_dir, f"{respondent}_comparison.pdf"), dpi=150, bbox_inches="tight")
    if respondent == 'BPAT':
        save_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)),'src','visualisations', "Figure4.pdf")
        plt.savefig(save_path, dpi=450, bbox_inches="tight")
    plt.close()
    

# =====================
# Combine all results
# =====================
final_df = pd.concat(all_results, ignore_index=True)
final_df.to_csv(os.path.join(output_dir, "all_respondents_OLS.csv"), index=False)
