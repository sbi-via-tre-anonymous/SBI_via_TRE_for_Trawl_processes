import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import os
import pandas as pd
from matplotlib.ticker import FuncFormatter
plt.style.use(['science'])

tre_type_list = ('acf', 'beta', 'mu', 'sigma')
metric_name_list = ('BCE', 'S', 'acc', 'B')
data_path = os.path.join(os.getcwd(),'models_and_simulated_datasets','classifiers','TRE_full_trawl','metric_lots_during_training')

# Dict: {metric_name: {tre_type: (steps, values)}}
result_dict = {}

for metric_name in metric_name_list:
    inner_dict = {}

    for tre_type in tre_type_list:
        # Load CSV
        csv_path = os.path.join(data_path, f'{tre_type}', f'{tre_type}_{metric_name}.csv')
        df__ = pd.read_csv(csv_path)

        # Extract step and value arrays directly
        steps = df__['Step'].values
        values = df__.iloc[:, 1].values  

        # Store both in dict
        inner_dict[tre_type] = (steps, values)

    result_dict[metric_name] = inner_dict


############################## PLOTTING #######################################

def format_func_raw(value, tick_number):
    """Format axis ticks as regular integers with commas"""
    return f'{int(value):,}'


# Colors and LaTeX names
colors = {
    'acf': '#1f77b4',
    'beta': '#ff7f0e',
    'mu': '#2ca02c',
    'sigma': '#d62728'
}

tre_type_list_latex = dict(zip(tre_type_list, ('ACF', r'$\beta$', r'$\mu$', r'$\sigma$')))
metric_name_latex = dict(zip(metric_name_list, ('BCE loss', r'$\mathcal{S}$', 'Accuracy', r'$\mathcal{B}$')))

# Create figure
fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5.5), constrained_layout=True)

# Shared display ticks (for appearance only)
custom_ticks = [6000, 15000, 25000, 35000]

for idx, metric_name in enumerate(metric_name_list):
    ax = axes2[idx]
    metric_data = result_dict[metric_name]

    for tre_type in tre_type_list:
        steps, values = metric_data[tre_type]
        ax.plot(steps, values,
                label=tre_type_list_latex[tre_type],
                color=colors[tre_type],
                linewidth=1.5)

    ax.set_title(metric_name_latex[metric_name], fontsize=21)
    ax.grid(True, alpha=0.3)

    if idx == 0:
        ax.set_ylabel('Value', fontsize=21)

    # Cosmetic x-ticks
    ax.set_xticks(custom_ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(format_func_raw))
    ax.tick_params(axis='x', rotation=45, labelsize=17)
    ax.tick_params(axis='y', labelsize=17)

    # Add legend only to the rightmost plot
    if idx == len(metric_name_list) - 1:
        ax.legend(frameon=True, loc='lower right', title='classifier',
                  bbox_to_anchor=(1.02, -0.02),
                  fontsize=15.2, title_fontsize=16.7)

# Shared x-label
fig2.supxlabel('Step', fontsize=21)

plt.tight_layout()
plt.subplots_adjust(bottom=0.175)

# Save & show
save_path = os.path.join(os.getcwd(),'src','visualisations','Figure2.pdf')
plt.savefig(save_path, dpi=600, bbox_inches='tight')
plt.show()
