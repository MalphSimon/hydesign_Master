import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FINANCIAL_METRICS = [
    {
        "key": "npv",
        "title": "NPV Mean",
        "ylabel": "M EUR",
        "color": "#3498db",
        "candidates": ["NPV [MEuro]", "NPV"],
    },
    {
        "key": "npv_capex",
        "title": "NPV/CAPEX Mean",
        "ylabel": "%",
        "color": "#ff9800",
        "candidates": ["NPV_over_CAPEX", "NPV/CAPEX"],
    },
    {
        "key": "revenue",
        "title": "Revenue Mean",
        "ylabel": "M EUR",
        "color": "#e74c3c",
        "candidates": ["Revenues [MEuro]", "Revenue"],
    },
    {
        "key": "irr",
        "title": "IRR Mean",
        "ylabel": "%",
        "color": "#27ae60",
        "candidates": ["IRR"],
    },
]

BANKABILITY_METRICS = [
    {
        "key": "dscr",
        "title": "DSCR Mean",
        "ylabel": "[-]",
        "color": "#3498db",
        "candidates": ["DSCR [-]"],
    },
    {
        "key": "llcr",
        "title": "LLCR P50 Mean",
        "ylabel": "[-]",
        "color": "#ff9800",
        "candidates": ["LLCR P50 [-]", "LLCR P50", "LLCR [-]"],
    },
    {
        "key": "dscr_breach_years",
        "title": "DSCR Breach Years Mean",
        "ylabel": "Years",
        "color": "#e74c3c",
        "candidates": ["DSCR Breach Years"],
    },
    {
        "key": "debt_headroom",
        "title": "Debt Headroom Mean",
        "ylabel": "M EUR",
        "color": "#27ae60",
        "candidates": ["Debt Headroom [MEuro]"],
    },
]

def compare_yearly_evaluations(names, site_dir, portfolio_dir, save_path, metrics, title):
    """
    Compare yearly evaluations for sites and portfolios.
    Args:
        names (list): Names of sites or portfolios to compare.
        site_dir (str): Directory for individual HPP files.
        portfolio_dir (str): Directory for Portfolio files.
        save_path (str): Full path to save the resulting .png.
    """
    stats = {n: {} for n in names}

    for name in names:
        # 1. Determine file path (Check Site dir first, then Portfolio dir)
        # Individual HPP naming convention
        site_path = os.path.join(site_dir, f"{name}_yearly_eval_1982_2015_life25_p30.csv")
        # Portfolio naming convention
        port_path = os.path.join(portfolio_dir, f"{name}_yearly.csv")

        if os.path.exists(site_path):
            csv_path = site_path
            is_portfolio = False
        elif os.path.exists(port_path):
            csv_path = port_path
            is_portfolio = True
        else:
            print(f"Warning: No file found for '{name}' in sites or portfolios. Skipping.")
            continue

        df = pd.read_csv(csv_path)
        prefix = "portfolio_" if is_portfolio else ""

        # 2. Flexible column finder (handles the portfolio_ prefix)
        def find_col(possibles):
            for p in possibles:
                # Check with and without prefix
                for search_str in [f"{prefix}{p}", p]:
                    for c in df.columns:
                        if search_str.lower() == c.lower():
                            return c
            return None

        # Define targets
        cols = {cfg["key"]: find_col(cfg["candidates"]) for cfg in metrics}

        # 3. Compute Statistics
        for key, col in cols.items():
            if col and col in df.columns:
                data = pd.to_numeric(df[col], errors='coerce').dropna()
                stats[name][f"{key}_mean"] = np.mean(data)
                stats[name][f"{key}_std"] = np.std(data)
            else:
                stats[name][f"{key}_mean"] = 0
                stats[name][f"{key}_std"] = 0

    # 4. Plotting Logic
    valid_names = [n for n in names if stats[n]]
    if not valid_names:
        print("No valid data to plot.")
        return

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title, fontsize=18)

    plot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for cfg, pos in zip(metrics, plot_positions):
        key = cfg["key"]
        ylabel = cfg["ylabel"]
        color = cfg["color"]
        means = [stats[n].get(f"{key}_mean", 0) for n in valid_names]
        stds = [stats[n].get(f"{key}_std", 0) for n in valid_names]
        
        axs[pos].bar(valid_names, means, yerr=stds, color=color, capsize=8, alpha=0.8)
        axs[pos].set_title(cfg["title"], fontweight='bold')
        axs[pos].set_ylabel(ylabel)
        axs[pos].grid(axis='y', linestyle='--', alpha=0.6)
        plt.setp(axs[pos].get_xticklabels(), rotation=30, ha='right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Successfully saved comparison plot to: {save_path}")


def compare_financial_and_bankability(names, site_dir, portfolio_dir, financial_save_path, bankability_save_path):
    compare_yearly_evaluations(
        names=names,
        site_dir=site_dir,
        portfolio_dir=portfolio_dir,
        save_path=financial_save_path,
        metrics=FINANCIAL_METRICS,
        title="HPP vs Portfolio Financial Summary Statistics (Mean ± Std)",
    )
    compare_yearly_evaluations(
        names=names,
        site_dir=site_dir,
        portfolio_dir=portfolio_dir,
        save_path=bankability_save_path,
        metrics=BANKABILITY_METRICS,
        title="HPP vs Portfolio Bankability Summary Statistics (Mean ± Std)",
    )

# --- Execute ---
SITE_DIR = r"C:\Users\malth\HPP\hydesign\HPP\Evaluations\P30"
PORT_DIR = r"C:\Users\malth\HPP\hydesign\HPP\Portfolio\Outputs"
SAVE_TO_FINANCIAL = r"C:\Users\malth\HPP\hydesign\HPP\HPPCompares\HPPEvalCompareP30finance.png"
SAVE_TO_BANKABILITY = r"C:\Users\malth\HPP\hydesign\HPP\HPPCompares\HPPEvalCompareP30_bankability.png"

# Mix and match sites and portfolios here:
compare_financial_and_bankability(
    names=[
        "Golfe_du_Lion", 
        "NordsoenMidt", 
        "Vestavind",
        "SicilySouth",
        "Sud_Atlantique",
        "Thetys", 
    ],
    site_dir=SITE_DIR,
    portfolio_dir=PORT_DIR,
    financial_save_path=SAVE_TO_FINANCIAL,
    bankability_save_path=SAVE_TO_BANKABILITY,
)