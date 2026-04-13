import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. NAME MAPPING & METRIC CONFIGURATION
# ---------------------------------------------------------------------------
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

# --- Original Metric Lists ---
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

# --- New NPV-Specific List ---
NPV_ONLY_METRICS = [
    {"key": "npv", "title": "NPV Mean", "ylabel": "M EUR", "color": "#3498db", "candidates": ["NPV [MEuro]", "NPV"]},
    {"key": "npv_capex", "title": "NPV/CAPEX Mean", "ylabel": "%", "color": "#ff9800", "candidates": ["NPV_over_CAPEX", "NPV/CAPEX"]},
]

# ---------------------------------------------------------------------------
# 2. UPDATED DYNAMIC CORE LOGIC
# ---------------------------------------------------------------------------

def compare_yearly_evaluations(names, site_dir, portfolio_dir, save_path, metrics, title):
    """
    Computes statistics for sites/portfolios and generates comparison plots.
    Automatically handles any number of metrics by adjusting subplot grid.
    """
    stats = {n: {} for n in names}

    for name in names:
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

        def find_col(possibles):
            for p in possibles:
                for search_str in [f"{prefix}{p}", p]:
                    for c in df.columns:
                        if search_str.lower() == c.lower():
                            return c
            return None

        for cfg in metrics:
            key, col = cfg["key"], find_col(cfg["candidates"])
            if col and col in df.columns:
                data = pd.to_numeric(df[col], errors='coerce').dropna()
                stats[name][f"{key}_mean"] = np.mean(data)
                stats[name][f"{key}_std"] = np.std(data)
            else:
                stats[name][f"{key}_mean"] = stats[name][f"{key}_std"] = 0

    valid_names = [n for n in names if stats[n]]
    if not valid_names:
        print("No valid data to plot.")
        return

    plot_labels = [SITE_DISPLAY_MAP.get(n, n) for n in valid_names]

    # --- DYNAMIC GRID CALCULATION ---
    num_metrics = len(metrics)
    ncols = 2 if num_metrics > 1 else 1
    nrows = (num_metrics + 1) // 2
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=18)
    axs_flat = axs.flatten()

    for i, cfg in enumerate(metrics):
        key = cfg["key"]
        means = [stats[n].get(f"{key}_mean", 0) for n in valid_names]
        stds = [stats[n].get(f"{key}_std", 0) for n in valid_names]
        
        ax = axs_flat[i]
        ax.bar(plot_labels, means, yerr=stds, color=cfg["color"], capsize=8, alpha=0.8)
        ax.set_title(cfg["title"], fontweight='bold')
        ax.set_ylabel(cfg["ylabel"])
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # Remove empty subplots if number of metrics is odd
    for j in range(i + 1, len(axs_flat)):
        fig.delaxes(axs_flat[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Successfully saved plot: {os.path.basename(save_path)}")


def compare_financial_and_bankability(names, site_dir, portfolio_dir, finance_path, bank_path, npv_path):
    """Orchestrates all three comparison plots."""
    
    # 1. Original Financial Comparison (4 Plots)
    compare_yearly_evaluations(
        names, site_dir, portfolio_dir, finance_path, FINANCIAL_METRICS,
        "HPP vs Portfolio Financial Summary Statistics (Mean ± Std)"
    )
    
    # 2. Original Bankability Comparison (4 Plots)
    compare_yearly_evaluations(
        names, site_dir, portfolio_dir, bank_path, BANKABILITY_METRICS,
        "HPP vs Portfolio Bankability Summary Statistics (Mean ± Std)"
    )

    # 3. NEW NPV & NPV/CAPEX Comparison (Only 2 Plots)
    compare_yearly_evaluations(
        names, site_dir, portfolio_dir, npv_path, NPV_ONLY_METRICS,
        "HPP vs Portfolio: NPV & Efficiency Comparison"
    )

# ---------------------------------------------------------------------------
# 3. EXECUTION
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Local Paths ---
    SITE_DIR = r"C:\Users\malth\HPP\hydesign\HPP\Evaluations\P30"
    PORT_DIR = r"C:\Users\malth\HPP\hydesign\HPP\Portfolio\Outputs"
    
    SAVE_FINANCE = r"C:\Users\malth\HPP\hydesign\HPP\HPPCompares\HPPEvalCompareP300finance.png"
    SAVE_BANK = r"C:\Users\malth\HPP\hydesign\HPP\HPPCompares\HPPEvalCompareP300bankability.png"
    SAVE_NPV_ONLY = r"C:\Users\malth\HPP\hydesign\HPP\HPPCompares\NPV_CAPEX_Comparison.png"

    # --- Site List ---
    COMPARE_LIST = [
        "Golfe_du_Lion", 
        "NordsoenMidt", 
        "Vestavind",
        "SicilySouth",
        "Sud_Atlantique",
        "Thetys",
        "NordsoenMidt_Golfe_du_Lion"
    ]

    compare_financial_and_bankability(
        names=COMPARE_LIST,
        site_dir=SITE_DIR,
        portfolio_dir=PORT_DIR,
        finance_path=SAVE_FINANCE,
        bank_path=SAVE_BANK,
        npv_path=SAVE_NPV_ONLY
    )