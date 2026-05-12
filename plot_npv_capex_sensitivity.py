"""
Plot NPV over CAPEX comparison with wind and solar generation sensitivity analysis.

This script extends the NPV/CAPEX comparison to include sensitivity bands showing
the impact of ±20% variations in wind and solar generation (in 5% steps).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Define paths
eval_dir = Path(r'HPP\Evaluations\HiFiEMS\P25')
sensitivity_dir = Path(r'HPP\Evaluations\HiFiEMS\Sensitivity_HiFiEMS')

# Find all HiFiEMS and Offshore HiFiEMS file pairs
hifiems_files = {}
offshore_files = {}

for file in eval_dir.glob('*HiFiEMS_eval_*.csv'):
    filename = file.name
    
    # Skip normalized and variant versions (Wind, Solar)
    if 'Norm' in filename or 'Wind' in filename or 'Solar' in filename:
        continue
    
    if 'Offshore' in filename:
        # Extract site name
        site = filename.replace('_Offshore_HiFiEMS_eval_1982_2015_p25.0.csv', '')
        offshore_files[site] = file
    else:
        # Extract site name
        site = filename.replace('_HiFiEMS_eval_1982_2015_p25.0.csv', '')
        hifiems_files[site] = file

# Find common sites
common_sites = sorted(set(hifiems_files.keys()) & set(offshore_files.keys()))

print(f"Found {len(common_sites)} sites with both HiFiEMS and Offshore variants:")
print(f"  {', '.join(common_sites)}\n")

# Load data and calculate NPV/CAPEX for each site with sensitivity ranges
results = {
    'site': [],
    'HiFiEMS': [],
    'HiFiEMS_wind_min': [],
    'HiFiEMS_wind_max': [],
    'HiFiEMS_solar_min': [],
    'HiFiEMS_solar_max': [],
    'Offshore': [],
    'Offshore_wind_min': [],
    'Offshore_wind_max': [],
    'Offshore_solar_min': [],
    'Offshore_solar_max': [],
}

for site in common_sites:
    # Load baseline HiFiEMS data
    hifi_df = pd.read_csv(hifiems_files[site])
    hifi_npv_capex = hifi_df['NPV_over_CAPEX'].mean()
    
    # Load baseline Offshore HiFiEMS data
    offshore_df = pd.read_csv(offshore_files[site])
    offshore_npv_capex = offshore_df['NPV_over_CAPEX'].mean()
    
    # Initialize sensitivity bounds to baseline values
    hifi_wind_min = hifi_npv_capex
    hifi_wind_max = hifi_npv_capex
    hifi_solar_min = hifi_npv_capex
    hifi_solar_max = hifi_npv_capex
    offshore_wind_min = offshore_npv_capex
    offshore_wind_max = offshore_npv_capex
    offshore_solar_min = offshore_npv_capex
    offshore_solar_max = offshore_npv_capex
    
    # Try to load sensitivity analysis results for this site
    # Look for sensitivity summary files
    base_site_name = site.replace('_Offshore', '')
    sensitivity_pattern = f"*{base_site_name}_hifiems_sensitivity_summary_*.csv"
    
    sensitivity_files = list(sensitivity_dir.glob(sensitivity_pattern))
    
    for sens_file in sensitivity_files:
        try:
            sens_df = pd.read_csv(sens_file)
            
            # Extract wind generation sensitivity (±20%, steps of 5%)
            wind_gen_scenarios = sens_df[sens_df['parameter_group'] == 'wind_generation'].copy()
            if not wind_gen_scenarios.empty:
                wind_levels = wind_gen_scenarios['level'].values
                wind_means = wind_gen_scenarios['NPV_over_CAPEX mean'].values
                
                if len(wind_means) > 0:
                    # Determine if this is HiFiEMS or Offshore based on site name in file
                    if 'Offshore' in sens_file.name or 'Offshore' in site:
                        offshore_wind_min = wind_means.min()
                        offshore_wind_max = wind_means.max()
                    else:
                        hifi_wind_min = wind_means.min()
                        hifi_wind_max = wind_means.max()
            
            # Extract solar generation sensitivity (±20%, steps of 5%)
            solar_gen_scenarios = sens_df[sens_df['parameter_group'] == 'solar_generation'].copy()
            if not solar_gen_scenarios.empty:
                solar_levels = solar_gen_scenarios['level'].values
                solar_means = solar_gen_scenarios['NPV_over_CAPEX mean'].values
                
                if len(solar_means) > 0:
                    # Determine if this is HiFiEMS or Offshore based on site name in file
                    if 'Offshore' in sens_file.name or 'Offshore' in site:
                        offshore_solar_min = solar_means.min()
                        offshore_solar_max = solar_means.max()
                    else:
                        hifi_solar_min = solar_means.min()
                        hifi_solar_max = solar_means.max()
        except Exception as e:
            print(f"Warning: Could not load sensitivity data from {sens_file}: {e}")
    
    # Calculate percentage difference
    pct_diff = ((offshore_npv_capex - hifi_npv_capex) / abs(hifi_npv_capex) * 100) if hifi_npv_capex != 0 else 0
    
    results['site'].append(site)
    results['HiFiEMS'].append(hifi_npv_capex)
    results['HiFiEMS_wind_min'].append(hifi_wind_min)
    results['HiFiEMS_wind_max'].append(hifi_wind_max)
    results['HiFiEMS_solar_min'].append(hifi_solar_min)
    results['HiFiEMS_solar_max'].append(hifi_solar_max)
    results['Offshore'].append(offshore_npv_capex)
    results['Offshore_wind_min'].append(offshore_wind_min)
    results['Offshore_wind_max'].append(offshore_wind_max)
    results['Offshore_solar_min'].append(offshore_solar_min)
    results['Offshore_solar_max'].append(offshore_solar_max)
    
    print(f"{site:25} | HiFiEMS: {hifi_npv_capex:8.4f} | Offshore: {offshore_npv_capex:8.4f} | Difference: {pct_diff:6.2f}%")

# Create DataFrame for plotting
results_df = pd.DataFrame(results)

# Create figure with subplots for HiFiEMS and Offshore
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

x = np.arange(len(common_sites))
width = 0.35
bar_width = 0.15  # Width for sensitivity bands visualization

# Function to plot bars with sensitivity bands
def plot_with_sensitivity(ax, x, baseline_vals, wind_mins, wind_maxs, solar_mins, solar_maxs, 
                         label, color_idx=0):
    colors = ['#1f77b4', '#ff7f0e']  # blue, orange
    color = colors[color_idx]
    
    # Plot baseline bars
    bars = ax.bar(x, baseline_vals, bar_width, label=label, alpha=0.8, color=color)
    
    # Plot wind generation sensitivity as error bars
    wind_errors = np.array([
        np.array(baseline_vals) - np.array(wind_mins),
        np.array(wind_maxs) - np.array(baseline_vals)
    ])
    
    # Plot solar generation sensitivity as error bars
    solar_errors = np.array([
        np.array(baseline_vals) - np.array(solar_mins),
        np.array(solar_maxs) - np.array(baseline_vals)
    ])
    
    # Add shaded regions for wind sensitivity
    for i, site in enumerate(common_sites):
        if wind_mins[i] != baseline_vals[i] or wind_maxs[i] != baseline_vals[i]:
            ax.fill_between([x[i] - bar_width/2, x[i] + bar_width/2], 
                           wind_mins[i], wind_maxs[i], 
                           alpha=0.2, color='green', label='Wind Gen. Range (±20%)' if i == 0 else '')
    
    # Add shaded regions for solar sensitivity  
    for i, site in enumerate(common_sites):
        if solar_mins[i] != baseline_vals[i] or solar_maxs[i] != baseline_vals[i]:
            ax.fill_between([x[i] - bar_width/2, x[i] + bar_width/2], 
                           solar_mins[i], solar_maxs[i], 
                           alpha=0.2, color='red', label='Solar Gen. Range (±20%)' if i == 0 else '')
    
    return bars

# Plot HiFiEMS data
bars1 = plot_with_sensitivity(
    ax1, x, 
    results_df['HiFiEMS'].values,
    results_df['HiFiEMS_wind_min'].values,
    results_df['HiFiEMS_wind_max'].values,
    results_df['HiFiEMS_solar_min'].values,
    results_df['HiFiEMS_solar_max'].values,
    'HiFiEMS', color_idx=0
)

ax1.set_xlabel('Site', fontsize=12, fontweight='bold')
ax1.set_ylabel('NPV / CAPEX Ratio', fontsize=12, fontweight='bold')
ax1.set_title('HiFiEMS with Generation Sensitivity (±20%, 5% steps)', 
             fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(results_df['site'], rotation=45, ha='right')
ax1.legend(fontsize=9, loc='best')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.axhline(y=0, color='black', linewidth=0.8)

# Add value labels on HiFiEMS bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=8)

# Plot Offshore HiFiEMS data
bars2 = plot_with_sensitivity(
    ax2, x, 
    results_df['Offshore'].values,
    results_df['Offshore_wind_min'].values,
    results_df['Offshore_wind_max'].values,
    results_df['Offshore_solar_min'].values,
    results_df['Offshore_solar_max'].values,
    'Offshore HiFiEMS', color_idx=1
)

ax2.set_xlabel('Site', fontsize=12, fontweight='bold')
ax2.set_ylabel('NPV / CAPEX Ratio', fontsize=12, fontweight='bold')
ax2.set_title('Offshore HiFiEMS with Generation Sensitivity (±20%, 5% steps)', 
             fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(results_df['site'], rotation=45, ha='right')
ax2.legend(fontsize=9, loc='best')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.axhline(y=0, color='black', linewidth=0.8)

# Add value labels on Offshore bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=8)

# Add overall title
fig.suptitle('NPV over CAPEX Comparison with Wind and Solar Generation Sensitivity Analysis\n(P25 - 1982-2015)', 
             fontsize=14, fontweight='bold', y=1.00)

# Adjust layout and save
plt.tight_layout()

# Save figure
output_path = Path(r'HPP\Evaluations\HiFiEMS\P25\Plots\NPV_CAPEX_Sensitivity_Analysis.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")

# Also create a summary figure showing sensitivity ranges
fig2, ax3 = plt.subplots(figsize=(14, 8))

# Prepare data for sensitivity comparison
param_groups = ['Wind Generation', 'Solar Generation']
site_names = results_df['site'].values

# Calculate ranges for each site and parameter
y_pos = np.arange(len(common_sites) * 2)
sensitivity_ranges = []
site_labels = []

for site in common_sites:
    site_idx = results_df[results_df['site'] == site].index[0]
    
    # Wind generation range
    wind_range = results_df.loc[site_idx, 'HiFiEMS_wind_max'] - results_df.loc[site_idx, 'HiFiEMS_wind_min']
    sensitivity_ranges.append(wind_range)
    site_labels.append(f"{site}\n(Wind)")
    
    # Solar generation range
    solar_range = results_df.loc[site_idx, 'HiFiEMS_solar_max'] - results_df.loc[site_idx, 'HiFiEMS_solar_min']
    sensitivity_ranges.append(solar_range)
    site_labels.append(f"{site}\n(Solar)")

colors_sens = ['green' if 'Wind' in label else 'red' for label in site_labels]
bars = ax3.barh(y_pos, sensitivity_ranges, color=colors_sens, alpha=0.7)

ax3.set_yticks(y_pos)
ax3.set_yticklabels(site_labels, fontsize=10)
ax3.set_xlabel('NPV/CAPEX Sensitivity Range (±20%)', fontsize=12, fontweight='bold')
ax3.set_title('Wind and Solar Generation Sensitivity Ranges for HiFiEMS Configurations', 
             fontsize=13, fontweight='bold')
ax3.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, sensitivity_ranges)):
    ax3.text(val, bar.get_y() + bar.get_height()/2., f'{val:.4f}',
            ha='left', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()

# Save sensitivity ranges figure
output_path2 = Path(r'HPP\Evaluations\HiFiEMS\P25\Plots\Generation_Sensitivity_Ranges.png')
output_path2.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"Sensitivity ranges figure saved to: {output_path2}")

plt.show()
