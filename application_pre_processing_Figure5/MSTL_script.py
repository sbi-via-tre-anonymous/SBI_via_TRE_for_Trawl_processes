import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import MSTL
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

# three possibilitites we investigated
#MSTL double: first apply with periods 7 14 365, then apply again with periods 7 365
#MSTL: apply once with periods 7 365
#MSTL 14: first 14, then apply again with 7 14 365
output_dir = "MSTL_results_14" #"MSTL_results_double"
os.makedirs(output_dir, exist_ok=True)

all_results = []

for respondent in df['respondent'].unique():
    print(f"Processing respondent: {respondent}")
    
    # Subset
    subdf = df[df['respondent'] == respondent].copy()
    subdf = subdf.set_index('period').sort_index()
    series = subdf['value'].astype(float)
    original_series = series.copy()
    
    # =====================
    # Apply MSTL
    # =====================
    mstl = MSTL(series, periods=(7, 14, 365))
    res = mstl.fit()
    
    #series = pd.Series(res.resid.values, index = series.index)
    #
    #mstl = MSTL(series, periods=(7, 365))
    #res = mstl.fit()
    
    result_df = pd.DataFrame({
        'trend': res.trend,
        'seasonal_7': res.seasonal['seasonal_7'],
        'seasonal_365': res.seasonal['seasonal_365'],
        'resid': res.resid
    }, index=series.index)
    result_df['respondent'] = respondent
    all_results.append(result_df.reset_index())
    
    respondent_dir = os.path.join(output_dir, respondent)
    os.makedirs(respondent_dir, exist_ok=True)
    
    result_df.to_csv(os.path.join(respondent_dir, f"{respondent}_mstl.csv"))
    
    # =====================
    # Save MSTL plot
    # =====================
    fig = res.plot()
    fig.suptitle(f"MSTL decomposition for {respondent}", fontsize=14)
    plt.savefig(os.path.join(respondent_dir, f"{respondent}_mstl.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =====================
    # New plot: Original vs Trend+Seasonality and Residuals
    # =====================
    trend_seasonality = original_series - res.resid
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.25))
    
    # Left plot: Original series and trend+seasonality
    ax1.plot(original_series.index, original_series.values, 'b-', alpha=0.7, label='Original', linewidth=0.8)
    ax1.plot(trend_seasonality.index, trend_seasonality.values, 'r-', alpha=0.8, label='Trend+Seasonality', linewidth=1)
    ax1.set_title("Original vs Trend+Seasonality Time Series",fontsize=16)
    ax1.set_xlabel("Date",fontsize=14)
    ax1.set_ylabel("Value",fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Residuals
    ax2.plot(res.resid.index, res.resid.values, 'b-', linewidth=0.8, label = 'Residuals')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_title("Residuals",fontsize=16)
    ax2.set_xlabel("Date",fontsize=14)
    ax2.set_ylabel("Residuals",fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    
    plt.tight_layout()
    plt.savefig(os.path.join(respondent_dir, f"{respondent}_comparison.pdf"), dpi=150, bbox_inches="tight")
    if respondent == 'AZPS':
        save_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)),'src','visualisations', "Figure5.pdf")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    

# =====================
# Combine all results
# =====================
final_df = pd.concat(all_results, ignore_index=True)
final_df.to_csv(os.path.join(output_dir, "all_respondents_mstl.csv"), index=False)
