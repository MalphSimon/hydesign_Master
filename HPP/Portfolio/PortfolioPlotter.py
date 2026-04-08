"""
Portfolio Visualization Tool for Hybrid Power Plants (HPP).

This script generates standardized visual reports for aggregated portfolio data.
It produces two distinct sets of 2x2 grids for each portfolio:
1. Financial Metrics: NPV, NPV/CAPEX, Revenue, and IRR.
2. Bankability Metrics: DSCR, LLCR, Breach Years, and Debt Headroom.
"""

import argparse
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Metric Configurations
# ---------------------------------------------------------------------------
# These lists define the mapping between CSV column names (candidates)
# and the visual styling used in the Matplotlib plots.

METRICS = [
    {
        "key": "npv",
        "title": "NPV",
        "ylabel": "M EUR",
        "color": "tab:orange",
        "style": "line",
        "candidates": ["portfolio_NPV [MEuro]", "portfolio_NPV"],
    },
    {
        "key": "npv_capex",
        "title": "NPV/CAPEX",
        "ylabel": "%",
        "color": "tab:blue",
        "style": "line",
        "candidates": ["portfolio_NPV_over_CAPEX", "portfolio_NPV/CAPEX"],
    },
    {
        "key": "revenue",
        "title": "Revenue",
        "ylabel": "M EUR",
        "color": "tab:red",
        "style": "bar",
        "candidates": ["portfolio_Revenues [MEuro]", "portfolio_Revenue"],
    },
    {
        "key": "irr",
        "title": "IRR",
        "ylabel": "%",
        "color": "tab:green",
        "style": "line",
        "candidates": ["portfolio_IRR"],
    },
]

BANKABILITY_METRICS = [
    {
        "key": "dscr",
        "title": "DSCR",
        "ylabel": "[-]",
        "color": "tab:blue",
        "style": "line",
        "candidates": ["portfolio_DSCR [-]", "portfolio_DSCR Avg [-]"],
    },
    {
        "key": "llcr",
        "title": "LLCR",
        "ylabel": "[-]",
        "color": "tab:orange",
        "style": "line",
        "candidates": ["portfolio_LLCR [-]"],
    },
    {
        "key": "dscr_breach_years",
        "title": "DSCR Breach Years",
        "ylabel": "Years",
        "color": "tab:red",
        "style": "bar",
        "candidates": ["portfolio_DSCR Breach Years"],
    },
    {
        "key": "debt_headroom",
        "title": "Debt Headroom",
        "ylabel": "M EUR",
        "color": "tab:green",
        "style": "line",
        "candidates": ["portfolio_Debt Headroom [MEuro]"],
    },
]

# --- Default Paths ---
INPUT_DIR_DEFAULT = os.path.join("HPP", "Portfolio", "Outputs") 
OUTPUT_DIR_DEFAULT = os.path.join("HPP", "Portfolio", "plots")

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _find_column(df, candidates):
    """
    Identifies which candidate column exists in the provided DataFrame.
    
    Args:
        df (pd.DataFrame): The portfolio data.
        candidates (list): List of possible column names.
        
    Returns:
        str or None: The first matching column name found.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _infer_portfolio_name(csv_path):
    """
    Extracts the portfolio ID from the file name.
    
    Assumes naming convention: 'PortfolioName_yearly.csv'.
    """
    base = os.path.basename(csv_path)
    return base.replace("_yearly.csv", "")


def _prepare_year_axis(df):
    """
    Creates X-axis positions and extracts year labels.
    
    Returns:
        tuple: (array of x-positions, numeric year labels)
    """
    if "weather_year" in df.columns:
        years = pd.to_numeric(df["weather_year"], errors="coerce")
    else:
        # Fallback to index-based range if no year column exists
        years = pd.Series(np.arange(len(df)), index=df.index, dtype=float)
    return np.arange(len(df)), years


def _plot_metric(ax, x, labels, y, cfg):
    """
    Handles the low-level Matplotlib plotting for a single subplot.
    
    Args:
        ax: Matplotlib axis.
        x: Array of x-positions.
        labels: Original years for labeling.
        y: Data values.
        cfg (dict): Configuration dictionary for styling.
    """
    if cfg["style"] == "bar":
        ax.bar(x, y, color=cfg["color"], alpha=0.85)
    else:
        ax.plot(x, y, marker="o", linewidth=1.8, color=cfg["color"])

    ax.set_title(cfg["title"])
    ax.set_ylabel(cfg["ylabel"])
    ax.grid(True, alpha=0.25)
    ax.set_xticks(x)
    
    # Format labels: 1982 becomes '82 to prevent overlapping on the X-axis
    year_short = [f"{int(val) % 100:02d}" if pd.notna(val) else "" for val in labels]
    ax.set_xticklabels(year_short, rotation=35)


def plot_portfolio_file(csv_path, output_dir, metrics, metric_type):
    """
    Generates and saves a 2x2 grid of plots for a specific portfolio CSV.
    
    Args:
        csv_path (str): Path to the portfolio_yearly.csv.
        output_dir (str): Where to save the resulting .png.
        metrics (list): The list of metric configs to plot.
        metric_type (str): Label for the file name ('financial' or 'bankability').
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    port_name = _infer_portfolio_name(csv_path)
    x, year_labels = _prepare_year_axis(df)

    # Initialize a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    axes = axes.flatten()

    # Iterate through each subplot and its corresponding config
    for ax, cfg in zip(axes, metrics):
        col = _find_column(df, cfg["candidates"])
        if col is None:
            # Visual feedback in the plot if data is missing
            ax.text(0.5, 0.5, f"Missing:\n{cfg['candidates'][0]}", 
                    ha="center", va="center", color='red')
            ax.set_title(cfg["title"])
            continue

        y = pd.to_numeric(df[col], errors="coerce").to_numpy()
        _plot_metric(ax, x, year_labels, y, cfg)

    # Label bottom axes only
    axes[2].set_xlabel("Year")
    axes[3].set_xlabel("Year")

    suffix = "bankability" if metric_type == "bankability" else "financial"
    fig.suptitle(f"Portfolio {suffix.title()} Metrics - {port_name}", fontsize=16)

    # Persistence
    os.makedirs(output_dir, exist_ok=True)
    out_fn = os.path.join(output_dir, f"{port_name}_{suffix}_metrics.png")
    fig.savefig(out_fn, dpi=160)
    plt.close(fig)
    return out_fn

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    """CLI entry point to batch-process portfolio plots."""
    parser = argparse.ArgumentParser(description="Generate plots for HPP Portfolios.")
    parser.add_argument("--input-dir", default=INPUT_DIR_DEFAULT, 
                        help="Directory containing portfolio yearly CSVs.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT, 
                        help="Directory to save plots.")
    args = parser.parse_args()

    # Target the specific yearly aggregation files
    pattern = os.path.join(args.input_dir, "*_yearly.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"No portfolio yearly files found in {args.input_dir}")
        return

    print(f"Found {len(csv_files)} portfolios. Generating plots...")
    
    for csv_path in csv_files:
        # Generate Financial Grid (NPV, Revenue, etc.)
        f_plot = plot_portfolio_file(csv_path, args.output_dir, METRICS, "financial")
        if f_plot: 
            print(f"  [OK] Saved Financial: {os.path.basename(f_plot)}")
        
        # Generate Bankability Grid (DSCR, LLCR, etc.)
        b_plot = plot_portfolio_file(csv_path, args.output_dir, BANKABILITY_METRICS, "bankability")
        if b_plot: 
            print(f"  [OK] Saved Bankability: {os.path.basename(b_plot)}")


if __name__ == "__main__":
    main()