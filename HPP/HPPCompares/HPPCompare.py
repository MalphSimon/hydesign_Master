"""
Statistical Comparison Tool for HPP Sites and Portfolios.

This script aggregates results from different simulation runs (individual HPPs 
and portfolios) to compare their financial and bankability performance. 
It visualizes the Mean and Standard Deviation (volatility) using bar charts 
with error bars, allowing for a direct comparison of risk and return.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. NAME MAPPING CONFIGURATION
# ---------------------------------------------------------------------------
# Key: The actual name used in the filename (e.g., "Golfe_du_Lion")
# Value: The name you want to see on the plot (e.g., "France - Mediterranean")

SITE_DISPLAY_MAP = {
    "Golfe_du_Lion": "Golfe du Lion (FRs)",
    "NordsoenMidt": "Nordsøen Midt (DK)",
    "Vestavind": "Vestavind (NO)",
    "SicilySouth": "Sicily South (IT)",
    "Sud_Atlantique": "Sud Atlantique (FRw)",
    "Thetys": "Thetys (NL)",
    "All_Sites": "Total Portfolio",
    "NordsoenMidt_Golfe_du_Lion": "NSM + GDL Joint"
}

# ---------------------------------------------------------------------------
# 2. METRIC CONFIGURATIONS
# ---------------------------------------------------------------------------

FINANCIAL_METRICS = [
    {"key": "npv", "title": "NPV Mean", "ylabel": "M EUR", "color": "#3498db", "candidates": ["NPV [MEuro]", "NPV"]},
    {"key": "npv_capex", "title": "NPV/CAPEX Mean", "ylabel": "%", "color": "#ff9800", "candidates": ["NPV_over_CAPEX", "NPV/CAPEX"]},
    {"key": "revenue", "title": "Revenue Mean", "ylabel": "M EUR", "color": "#e74c3c", "candidates": ["Revenues [MEuro]", "Revenue"]},
    {"key": "irr", "title": "IRR Mean", "ylabel": "%", "color": "#27ae60", "candidates": ["IRR"]},
]

BANKABILITY_METRICS = [
    {"key": "dscr", "title": "DSCR Mean", "ylabel": "[-]", "color": "#3498db", "candidates": ["DSCR [-]"]},
    {"key": "llcr", "title": "LLCR P50 Mean", "ylabel": "[-]", "color": "#ff9800", "candidates": ["LLCR P50 [-]", "LLCR P50", "LLCR [-]"]},
    {"key": "dscr_breach_years", "title": "DSCR Breach Years Mean", "ylabel": "Years", "color": "#e74c3c", "candidates": ["DSCR Breach Years"]},
    {"key": "debt_headroom", "title": "Debt Headroom Mean", "ylabel": "M EUR", "color": "#27ae60", "candidates": ["Debt Headroom [MEuro]"]},
]

# ---------------------------------------------------------------------------
# 3. CORE LOGIC
# ---------------------------------------------------------------------------

def compare_yearly_evaluations(names, site_dir, portfolio_dir, save_path, metrics, title):
    """
    Computes statistics for sites/portfolios and generates comparison plots.
    
    Args:
        names (list): List of site/portfolio IDs (filenaming keys).
        site_dir (str): Folder path for individual HPP results.
        portfolio_dir (str): Folder path for Portfolio results.
        save_path (str): Full path to save the .png.
        metrics (list): Metric configuration dictionary.
        title (str): Figure title.
    """
    stats = {n: {} for n in names}

    for name in names:
        # File discovery: Site dir naming vs Portfolio dir naming
        site_path = os.path.join(site_dir, f"{name}_yearly_eval_1982_2015_life25_p30.csv")
        port_path = os.path.join(portfolio_dir, f"{name}_yearly.csv")

        if os.path.exists(site_path):
            csv_path, is_portfolio = site_path, False
        elif os.path.exists(port_path):
            csv_path, is_portfolio = port_path, True
        else:
            print(f"  [Warning] Skipping '{name}': No file found.")
            continue

        df = pd.read_csv(csv_path)
        prefix = "portfolio_" if is_portfolio else ""

        # Column Finder Logic
        def find_col(possibles):
            for p in possibles:
                for search_str in [f"{prefix}{p}", p]:
                    for c in df.columns:
                        if search_str.lower() == c.lower():
                            return c
            return None

        # Statistics Computation
        for cfg in metrics:
            key, col = cfg["key"], find_col(cfg["candidates"])
            if col and col in df.columns:
                data = pd.to_numeric(df[col], errors='coerce').dropna()
                stats[name][f"{key}_mean"] = np.mean(data)
                stats[name][f"{key}_std"] = np.std(data)
            else:
                stats[name][f"{key}_mean"] = stats[name][f"{key}_std"] = 0

    # Plotting Logic
    valid_names = [n for n in names if stats[n]]
    if not valid_names:
        print("No valid data to plot.")
        return

    # Convert Technical names to Display names for the X-Axis
    plot_labels = [SITE_DISPLAY_MAP.get(n, n) for n in valid_names]

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title, fontsize=18)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for cfg, pos in zip(metrics, positions):
        key = cfg["key"]
        means = [stats[n].get(f"{key}_mean", 0) for n in valid_names]
        stds = [stats[n].get(f"{key}_std", 0) for n in valid_names]
        
        # Plotting bars using mapped labels
        axs[pos].bar(plot_labels, means, yerr=stds, color=cfg["color"], capsize=8, alpha=0.8)
        axs[pos].set_title(cfg["title"], fontweight='bold')
        axs[pos].set_ylabel(cfg["ylabel"])
        axs[pos].grid(axis='y', linestyle='--', alpha=0.6)
        plt.setp(axs[pos].get_xticklabels(), rotation=30, ha='right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Successfully saved plot: {os.path.basename(save_path)}")


def compare_financial_and_bankability(
    names,
    site_dir,
    portfolio_dir,
    financial_save_path=None,
    bankability_save_path=None,
    f_save=None,
    b_save=None,
):
    """Orchestrates both financial and bankability comparison plots."""
    # Keep backward compatibility with legacy parameter names.
    financial_save_path = financial_save_path or f_save
    bankability_save_path = bankability_save_path or b_save

    if not financial_save_path or not bankability_save_path:
        raise ValueError(
            "Both financial and bankability save paths must be provided."
        )

    # Financial Comparison
    compare_yearly_evaluations(
        names, site_dir, portfolio_dir, financial_save_path, FINANCIAL_METRICS,
        "HPP vs Portfolio Financial Summary Statistics (Mean ± Std)"
    )
    # Bankability Comparison
    compare_yearly_evaluations(
        names, site_dir, portfolio_dir, bankability_save_path, BANKABILITY_METRICS,
        "HPP vs Portfolio Bankability Summary Statistics (Mean ± Std)"
    )

# ---------------------------------------------------------------------------
# 4. EXECUTION
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Local Paths ---
    SITE_DIR = r"C:\Users\malth\HPP\hydesign\HPP\Evaluations\P30"
    PORT_DIR = r"C:\Users\malth\HPP\hydesign\HPP\Portfolio\Outputs"
    SAVE_FINANCE = r"C:\Users\malth\HPP\hydesign\HPP\HPPCompares\HPPEvalCompareP300finance.png"
    SAVE_BANK = r"C:\Users\malth\HPP\hydesign\HPP\HPPCompares\HPPEvalCompareP300bankability.png"

    # --- Site List (Use the Filename Keys here) ---
    COMPARE_LIST = [
        "Golfe_du_Lion", 
        "NordsoenMidt", 
        "Vestavind",
        "SicilySouth",
        "Sud_Atlantique",
        "Thetys",
        "NordsoenMidt_Golfe_du_Lion" # Example portfolio entry
    ]

    compare_financial_and_bankability(
        names=COMPARE_LIST,
        site_dir=SITE_DIR,
        portfolio_dir=PORT_DIR,
        financial_save_path=SAVE_FINANCE,
        bankability_save_path=SAVE_BANK,
    )