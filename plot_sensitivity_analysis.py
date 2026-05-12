"""
Plot sensitivity analysis results for HiFiEMS evaluation
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define paths
summary_file = r'HPP\Evaluations\HiFiEMS\Sensitivity_HiFiEMS\Sud_Atlantique_HiFiEMS_hifiems_sensitivity_summary_1982_2015.csv'
detail_file = r'HPP\Evaluations\HiFiEMS\Sensitivity_HiFiEMS\Sud_Atlantique_HiFiEMS_hifiems_sensitivity_detail_1982_2015.csv'



# Load data
summary_df = pd.read_csv(summary_file)
detail_df = pd.read_csv(detail_file)

# Function to rename parameters for display
def format_parameter_name(param_group):
    rename_map = {
        'electricity_price': 'Electricity Price',
        'grid_connection': 'POC Cost',
        'pv_capex': 'PV CAPEX',
        'solar_opex': 'PV OPEX',
        'wacc': 'WACC',
        'wind_capex': 'Wind CAPEX',
        'wind_opex': 'Wind OPEX'
    }
    return rename_map.get(param_group, param_group.replace('_', ' ').title())

# Extract unique parameter groups (excluding baseline)
param_groups = summary_df[summary_df['scenario_id'] != 'baseline']['parameter_group'].unique()

# Create tornado plot for NPV/CAPEX sensitivity
fig, ax = plt.subplots(figsize=(11, 6))

baseline_npv_capex = summary_df[summary_df['scenario_id'] == 'baseline']['NPV_over_CAPEX mean'].values[0]
baseline_npv_capex_std = summary_df[summary_df['scenario_id'] == 'baseline']['NPV_over_CAPEX std'].values[0]

tornado_data = []

for param_group in param_groups:
    group_data = summary_df[summary_df['parameter_group'] == param_group].copy()
    group_data = group_data.sort_values('level')
    
    # Get NPV/CAPEX ratios and std directly from the data
    ratios = group_data['NPV_over_CAPEX mean'].values
    ratio_stds = group_data['NPV_over_CAPEX std'].values
    
    # Get parameter range from the data
    param_levels = group_data['level'].values
    param_range = f"({param_levels.min():.2f} to {param_levels.max():.2f})"
    
    # Calculate delta from baseline (in absolute terms)
    ratio_min = ratios.min() - baseline_npv_capex
    ratio_max = ratios.max() - baseline_npv_capex
    
    # Get std for min and max scenarios
    min_idx = np.argmin(ratios)
    max_idx = np.argmax(ratios)
    min_std = ratio_stds[min_idx]
    max_std = ratio_stds[max_idx]
    
    tornado_data.append({
        'Parameter': f"{format_parameter_name(param_group)} {param_range}",
        'Min': ratio_min,
        'Max': ratio_max,
        'Min_std': min_std,
        'Max_std': max_std,
        'Range': ratio_max - ratio_min
    })

tornado_df = pd.DataFrame(tornado_data).sort_values('Max', ascending=True)

y_pos = np.arange(len(tornado_df))

# Plot min bar (red)
ax.barh(y_pos, tornado_df['Min'], left=0, height=0.5, color='#d62728', alpha=0.8, zorder=2)

# Plot max bar (green)
ax.barh(y_pos, tornado_df['Max'], left=0, height=0.5, color='#2ca02c', alpha=0.8, zorder=2)

ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(tornado_df['Parameter'], fontsize=9)
ax.set_xlabel('Delta NPV over CAPEX vs baseline', fontsize=10)
ax.set_title('Sensitivity Analysis on NPV over CAPEX - Sud Atlantique (FRw)', fontsize=11)
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')

plt.tight_layout()
plt.savefig('tornado_plot_npv.png', dpi=300, bbox_inches='tight')
print("Saved: tornado_plot_npv.png")
plt.show()

# Create box plots for financial metrics by parameter group
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

metrics_for_box = [
    ('NPV [MEuro] mean', 'NPV [MEuro]'),
    ('IRR mean', 'IRR [-]'),
    ('LCOE [Euro/MWh] mean', 'LCOE [Euro/MWh]')
]

for ax_idx, (metric_col, metric_label) in enumerate(metrics_for_box):
    ax = axes[ax_idx]
    
    box_data = []
    labels = []
    
    for param_group in param_groups:
        group_data = summary_df[summary_df['parameter_group'] == param_group][metric_col].values
        box_data.append(group_data)
        labels.append(format_parameter_name(param_group))
    
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
    
    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('#87CEEB')
    
    ax.set_ylabel(metric_label, fontsize=9)
    ax.set_title(f'{metric_label}', fontsize=10)
    ax.grid(False)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

plt.tight_layout()
plt.savefig('sensitivity_analysis_distributions.png', dpi=300, bbox_inches='tight')
print("Saved: sensitivity_analysis_distributions.png")
plt.show()

print("\nSensitivity Analysis Summary:")
print(f"Parameter groups analyzed: {', '.join(param_groups)}")
print(f"Baseline NPV/CAPEX: {baseline_npv_capex:.4f}")
