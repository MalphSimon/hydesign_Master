
# Compare the yearly evaluations of the HPP in mean with std dev.
#
# Usage example (at bottom):
# compare_yearly_evaluations([
#     "Golfe_du_Lion",
#     "NordsoenMidt",
#     "SicilySouth",
#     "Sud_Atlantique",
#     "Thetys",
#     "Vestavind"
# ])

from doctest import Example
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compare_yearly_evaluations(site_names, eval_dir=None):
	"""
	Compare yearly evaluations for selected sites, plotting mean ± std for NPV, NPV/CAPEX, Revenue, IRR.
	Args:
		site_names (list of str): List of site names (e.g., ["Golfe_du_Lion", ...])
		eval_dir (str, optional): Directory containing yearly eval CSVs. Defaults to Evaluations/ in this script's folder.
	"""
	if eval_dir is None:
		eval_dir = os.path.join(os.path.dirname(__file__), "Evaluations")

	# Data containers
	stats = {s: {} for s in site_names}

	for site in site_names:
		csv_path = os.path.join(eval_dir, f"{site}_yearly_eval_1982_2015_life25.csv")
		if not os.path.exists(csv_path):
			print(f"Warning: {csv_path} not found. Skipping {site}.")
			continue
		df = pd.read_csv(csv_path)

		# Try to find columns (case-insensitive, allow some flexibility)
		def find_col(possibles):
			for p in possibles:
				for c in df.columns:
					if p.lower() == c.lower():
						return c
			return None

		npv_col = find_col(["NPV [MEuro]", "NPV", "npv"])
		capex_col = find_col(["CAPEX [MEuro]", "CAPEX", "capex"])
		revenue_col = find_col(["Revenues [MEuro]", "Revenue", "revenue"])
		irr_col = find_col(["IRR", "irr"])
		npv_over_capex_col = find_col(["NPV_over_CAPEX", "NPV/CAPEX", "NPV over CAPEX"])

		# Helper to check if column exists and is non-empty
		def valid_col(col):
			return col is not None and col in df.columns and df[col].notna().sum() > 0

		# Compute stats with checks
		if valid_col(npv_col):
			npv = df[npv_col]
			stats[site]["NPV_mean"] = np.nanmean(npv)
			stats[site]["NPV_std"] = np.nanstd(npv)
		else:
			print(f"Warning: NPV column missing or empty for {site}.")
			stats[site]["NPV_mean"] = np.nan
			stats[site]["NPV_std"] = np.nan

		if valid_col(revenue_col):
			revenue = df[revenue_col]
			stats[site]["Revenue_mean"] = np.nanmean(revenue)
			stats[site]["Revenue_std"] = np.nanstd(revenue)
		else:
			print(f"Warning: Revenue column missing or empty for {site}.")
			stats[site]["Revenue_mean"] = np.nan
			stats[site]["Revenue_std"] = np.nan

		if valid_col(irr_col):
			irr = df[irr_col]
			stats[site]["IRR_mean"] = np.nanmean(irr)
			stats[site]["IRR_std"] = np.nanstd(irr)
		else:
			print(f"Warning: IRR column missing or empty for {site}.")
			stats[site]["IRR_mean"] = np.nan
			stats[site]["IRR_std"] = np.nan

		# Prefer direct NPV_over_CAPEX column if present
		if valid_col(npv_over_capex_col):
			npv_over_capex = df[npv_over_capex_col]
			stats[site]["NPV_CAPEX_mean"] = np.nanmean(npv_over_capex)
			stats[site]["NPV_CAPEX_std"] = np.nanstd(npv_over_capex)
		elif valid_col(npv_col) and valid_col(capex_col):
			capex = df[capex_col]
			capex_nonzero = capex.replace(0, np.nan)
			npv_capex = df[npv_col] / capex_nonzero * 100
			if npv_capex.notna().sum() > 0:
				stats[site]["NPV_CAPEX_mean"] = np.nanmean(npv_capex)
				stats[site]["NPV_CAPEX_std"] = np.nanstd(npv_capex)
			else:
				print(f"Warning: NPV/CAPEX calculation invalid for {site} (all NaN or zero CAPEX).")
				stats[site]["NPV_CAPEX_mean"] = np.nan
				stats[site]["NPV_CAPEX_std"] = np.nan
		else:
			print(f"Warning: NPV_over_CAPEX, or NPV or CAPEX column missing or empty for {site}.")
			stats[site]["NPV_CAPEX_mean"] = np.nan
			stats[site]["NPV_CAPEX_std"] = np.nan

	# Prepare for plotting
	sites = list(stats.keys())
	npv_means = [stats[s]["NPV_mean"] for s in sites]
	npv_stds = [stats[s]["NPV_std"] for s in sites]
	npv_capex_means = [stats[s]["NPV_CAPEX_mean"] for s in sites]
	npv_capex_stds = [stats[s]["NPV_CAPEX_std"] for s in sites]
	revenue_means = [stats[s]["Revenue_mean"] for s in sites]
	revenue_stds = [stats[s]["Revenue_std"] for s in sites]
	irr_means = [stats[s]["IRR_mean"] for s in sites]
	irr_stds = [stats[s]["IRR_std"] for s in sites]

	fig, axs = plt.subplots(2, 2, figsize=(16, 9))
	fig.suptitle("All-Site Summary Statistics", fontsize=16)

	# NPV
	axs[0, 0].bar(sites, npv_means, yerr=npv_stds, color="#3498db", capsize=6)
	axs[0, 0].set_title("NPV Mean +/- Std")
	axs[0, 0].set_ylabel("M EUR")
	axs[0, 0].tick_params(axis='x', rotation=25)

	# NPV/CAPEX
	axs[0, 1].bar(sites, npv_capex_means, yerr=npv_capex_stds, color="#ff9800", capsize=6)
	axs[0, 1].set_title("NPV/CAPEX Mean +/- Std")
	axs[0, 1].set_ylabel("%")
	axs[0, 1].tick_params(axis='x', rotation=25)

	# Revenue
	axs[1, 0].bar(sites, revenue_means, yerr=revenue_stds, color="#e74c3c", capsize=6)
	axs[1, 0].set_title("Revenue Mean +/- Std")
	axs[1, 0].set_ylabel("M EUR")
	axs[1, 0].tick_params(axis='x', rotation=25)

	# IRR
	axs[1, 1].bar(sites, irr_means, yerr=irr_stds, color="#27ae60", capsize=6)
	axs[1, 1].set_title("IRR Mean +/- Std")
	axs[1, 1].set_ylabel("%")
	axs[1, 1].tick_params(axis='x', rotation=25)

	plt.tight_layout(rect=[0, 0.03, 1, 0.95])

	# Save the plot
	save_path = r"C:\Users\malth\HPP\hydesign\HPP\HPPCompares\HPPEvalCompare.png"
	plt.savefig(save_path, dpi=300)
	print(f"Plot saved to {save_path}")
	

# Example usage:
compare_yearly_evaluations([
     "Golfe_du_Lion",
     "NordsoenMidt",
	 "SicilySouth",
	 "Sud_Atlantique",
	 "Thetys",
	 "Vestavind"
 ])