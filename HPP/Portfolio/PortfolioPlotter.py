import argparse
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Metric Configurations (Updated with portfolio_ prefixes)
# ---------------------------------------------------------------------------
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

# Default directories - Point these to your portfolio OUTPUT folder
INPUT_DIR_DEFAULT = os.path.join("HPP", "Portfolio", "Outputs") 
OUTPUT_DIR_DEFAULT = os.path.join("HPP", "Portfolio", "plots")

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None

def _infer_portfolio_name(csv_path):
    # Portfolio files are named 'Name_yearly.csv'
    base = os.path.basename(csv_path)
    return base.replace("_yearly.csv", "")

def _prepare_year_axis(df):
    if "weather_year" in df.columns:
        years = pd.to_numeric(df["weather_year"], errors="coerce")
    else:
        years = pd.Series(np.arange(len(df)), index=df.index, dtype=float)
    return np.arange(len(df)), years

def _plot_metric(ax, x, labels, y, cfg):
    if cfg["style"] == "bar":
        ax.bar(x, y, color=cfg["color"], alpha=0.85)
    else:
        ax.plot(x, y, marker="o", linewidth=1.8, color=cfg["color"])

    ax.set_title(cfg["title"])
    ax.set_ylabel(cfg["ylabel"])
    ax.grid(True, alpha=0.25)
    ax.set_xticks(x)
    
    # Format labels to show last two digits of year (e.g., '82, '83)
    year_short = [f"{int(val) % 100:02d}" if pd.notna(val) else "" for val in labels]
    ax.set_xticklabels(year_short, rotation=35)

def plot_portfolio_file(csv_path, output_dir, metrics, metric_type):
    df = pd.read_csv(csv_path)
    if df.empty: return None

    port_name = _infer_portfolio_name(csv_path)
    x, year_labels = _prepare_year_axis(df)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    axes = axes.flatten()

    for ax, cfg in zip(axes, metrics):
        col = _find_column(df, cfg["candidates"])
        if col is None:
            ax.text(0.5, 0.5, f"Missing:\n{cfg['candidates'][0]}", ha="center", va="center")
            ax.set_title(cfg["title"])
            continue

        y = pd.to_numeric(df[col], errors="coerce").to_numpy()
        _plot_metric(ax, x, year_labels, y, cfg)

    axes[2].set_xlabel("Year")
    axes[3].set_xlabel("Year")

    suffix = "bankability" if metric_type == "bankability" else "financial"
    fig.suptitle(f"Portfolio {suffix.title()} Metrics - {port_name}", fontsize=16)

    os.makedirs(output_dir, exist_ok=True)
    out_fn = os.path.join(output_dir, f"{port_name}_{suffix}_metrics.png")
    fig.savefig(out_fn, dpi=160)
    plt.close(fig)
    return out_fn

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=INPUT_DIR_DEFAULT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    args = parser.parse_args()

    # Look specifically for the portfolio yearly files
    pattern = os.path.join(args.input_dir, "*_yearly.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"No portfolio yearly files found in {args.input_dir}")
        return

    print(f"Found {len(csv_files)} portfolios. Generating plots...")
    
    for csv_path in csv_files:
        # Financial Plot
        f_plot = plot_portfolio_file(csv_path, args.output_dir, METRICS, "financial")
        if f_plot: print(f"Saved: {f_plot}")
        
        # Bankability Plot
        b_plot = plot_portfolio_file(csv_path, args.output_dir, BANKABILITY_METRICS, "bankability")
        if b_plot: print(f"Saved: {b_plot}")

if __name__ == "__main__":
    main()