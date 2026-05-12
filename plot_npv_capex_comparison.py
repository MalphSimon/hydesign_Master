"""
Plot NPV over CAPEX comparison between HiFi EMS and Offshore HiFi EMS for all sites
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Define path to evaluations
eval_dir = Path(r'HPP\Evaluations\HiFiEMS\P25')

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

# Find common sites (those that have both HiFiEMS and Offshore variants)
common_sites = sorted(set(hifiems_files.keys()) & set(offshore_files.keys()))

print(f"Found {len(common_sites)} sites with both HiFiEMS and Offshore variants:")
print(f"  {', '.join(common_sites)}\n")

# Load data and calculate mean NPV/CAPEX for each site
results = {
    'site': [],
    'HiFiEMS': [],
    'Offshore': []
}

for site in common_sites:
    # Load HiFiEMS data
    hifi_df = pd.read_csv(hifiems_files[site])
    hifi_npv_capex = hifi_df['NPV_over_CAPEX'].mean()
    
    # Load Offshore HiFiEMS data
    offshore_df = pd.read_csv(offshore_files[site])
    offshore_npv_capex = offshore_df['NPV_over_CAPEX'].mean()
    
    # Calculate percentage difference
    pct_diff = ((offshore_npv_capex - hifi_npv_capex) / abs(hifi_npv_capex) * 100) if hifi_npv_capex != 0 else 0
    
    results['site'].append(site)
    results['HiFiEMS'].append(hifi_npv_capex)
    results['Offshore'].append(offshore_npv_capex)
    
    print(f"{site:25} | HiFiEMS: {hifi_npv_capex:8.4f} | Offshore: {offshore_npv_capex:8.4f} | Difference: {pct_diff:6.2f}%")

# Create DataFrame for plotting
results_df = pd.DataFrame(results)

# Create bar chart
fig, ax = plt.subplots(figsize=(12, 7))

# Set up bar positions
x = np.arange(len(common_sites))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, results_df['HiFiEMS'], width, label='HiFiEMS', alpha=0.8)
bars2 = ax.bar(x + width/2, results_df['Offshore'], width, label='Offshore HiFiEMS', alpha=0.8)

# Customize plot
ax.set_xlabel('Site', fontsize=12, fontweight='bold')
ax.set_ylabel('NPV / CAPEX Ratio', fontsize=12, fontweight='bold')
ax.set_title('NPV over CAPEX Comparison: HiFi EMS vs Offshore HiFi EMS (P25 - 1982-2015)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(results_df['site'], rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=0, color='black', linewidth=0.8)

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

add_value_labels(bars1)
add_value_labels(bars2)

# Adjust layout and save
plt.tight_layout()

# Save figure
output_path = Path(r'HPP\Evaluations\HiFiEMS\P25\Plots\NPV_CAPEX_Comparison.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")

# plt.show()
