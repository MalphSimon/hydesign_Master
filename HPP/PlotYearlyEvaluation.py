import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    {
        "key": "npv",
        "title": "NPV",
        "ylabel": "M EUR",
        "color": "tab:orange",
        "style": "line",
        "candidates": ["NPV [MEuro]", "NPV"],
    },
    {
        "key": "npv_capex",
        "title": "NPV/CAPEX",
        "ylabel": "%",
        "color": "tab:blue",
        "style": "line",
        "candidates": ["NPV_over_CAPEX", "NPV/CAPEX"],
    },
    {
        "key": "revenue",
        "title": "Revenue",
        "ylabel": "M EUR",
        "color": "tab:red",
        "style": "bar",
        "candidates": ["Revenues [MEuro]", "Revenue [MEuro]", "Revenue"],
    },
    {
        "key": "irr",
        "title": "IRR",
        "ylabel": "%",
        "color": "tab:green",
        "style": "line",
        "candidates": ["IRR", "IRR [%]"],
    },
]

BANKABILITY_METRICS = [
    {
        "key": "dscr",
        "title": "DSCR",
        "ylabel": "[-]",
        "color": "tab:blue",
        "style": "line",
        "candidates": ["DSCR [-]"],
    },
    {
        "key": "llcr",
        "title": "LLCR",
        "ylabel": "[-]",
        "color": "tab:orange",
        "style": "line",
        "candidates": ["LLCR [-]"],
    },
    {
        "key": "dscr_breach_years",
        "title": "DSCR Breach Years",
        "ylabel": "Years",
        "color": "tab:red",
        "style": "bar",
        "candidates": ["DSCR Breach Years"],
    },
    {
        "key": "debt_headroom",
        "title": "Debt Headroom",
        "ylabel": "M EUR",
        "color": "tab:green",
        "style": "line",
        "candidates": ["Debt Headroom [MEuro]"],
    },
]

# User settings for Run button usage (no terminal arguments required).
# Edit SITES_TO_PLOT to switch site(s):
# - ["SicilySouth"] for one site
# - ["SicilySouth", "NordsoenMidt"] for multiple sites
# - [] for all sites found in INPUT_DIR_DEFAULT
SITES_TO_PLOT = ["Golfe_du_Lion", "SicilySouth", "NordsoenMidt", "Sud_Atlantique", "Thetys", "Vestavind"]  # Example: plot only Golfe_du_Lion site. Set to [] to plot all sites in input dir.
INPUT_DIR_DEFAULT = os.path.join("hydesign", "HPP", "Evaluations")
OUTPUT_DIR_DEFAULT = os.path.join("hydesign", "HPP", "Evaluations", "plots")


def _find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _infer_site_name(df, csv_path):
    if "site" in df.columns:
        site_vals = df["site"].dropna().astype(str)
        if not site_vals.empty:
            return site_vals.iloc[0]

    base = os.path.basename(csv_path)
    if "_yearly_eval_" in base:
        return base.split("_yearly_eval_")[0]
    return os.path.splitext(base)[0]


def _prepare_year_axis(df):
    if "weather_year" in df.columns:
        years = pd.to_numeric(df["weather_year"], errors="coerce")
    else:
        years = pd.Series(np.arange(len(df)), index=df.index, dtype=float)

    year_labels = years.copy()
    years_for_plot = np.arange(len(df))

    return years_for_plot, year_labels


def _plot_metric(ax, x, labels, y, cfg):
    if cfg["style"] == "bar":
        ax.bar(x, y, color=cfg["color"], alpha=0.85)
    else:
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            color=cfg["color"],
        )

    ax.set_title(cfg["title"])
    ax.set_ylabel(cfg["ylabel"])
    ax.grid(True, alpha=0.25)

    ax.set_xticks(x)
    year_num = pd.to_numeric(labels, errors="coerce")
    year_short = []
    for val in year_num:
        if pd.notna(val):
            year_short.append(f"{int(val) % 100:02d}")
        else:
            year_short.append("")
    ax.set_xticklabels(year_short, rotation=35)


def plot_site_file(csv_path, output_dir, metrics=None, metric_type="yearly"):
    if metrics is None:
        metrics = METRICS
    
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Input CSV has no rows: {csv_path}")

    site_name = _infer_site_name(df, csv_path)
    x, year_labels = _prepare_year_axis(df)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    axes = axes.flatten()

    for ax, cfg in zip(axes, metrics):
        col = _find_column(df, cfg["candidates"])
        if col is None:
            ax.text(0.5, 0.5, "Column not found", ha="center", va="center")
            ax.set_title(cfg["title"])
            ax.set_axis_off()
            continue

        y = pd.to_numeric(df[col], errors="coerce").to_numpy()
        _plot_metric(ax, x, year_labels, y, cfg)

    axes[2].set_xlabel("Year")
    axes[3].set_xlabel("Year")

    if metric_type == "bankability":
        title = f"Bankability Metrics - {site_name}"
        out_suffix = "bankability_metrics"
    else:
        title = f"Yearly Evaluation Metrics - {site_name}"
        out_suffix = "yearly_metrics"

    fig.suptitle(title, fontsize=16)

    os.makedirs(output_dir, exist_ok=True)
    out_fn = os.path.join(output_dir, f"{site_name}_{out_suffix}.png")
    fig.savefig(out_fn, dpi=160)
    plt.close(fig)
    return out_fn


def _resolve_input_files(input_csv, input_dir):
    if input_csv:
        return input_csv

    pattern = os.path.join(input_dir, "*_yearly_eval_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No yearly evaluation CSV files found with: {pattern}"
        )
    return files


def _filter_files_by_sites(csv_files, sites_to_plot):
    if not sites_to_plot:
        return csv_files

    wanted = {str(site).strip().lower() for site in sites_to_plot}
    filtered = []
    for csv_path in csv_files:
        base = os.path.basename(csv_path)
        if "_yearly_eval_" in base:
            site_name = base.split("_yearly_eval_")[0]
        else:
            site_name = ""
        if site_name.strip().lower() in wanted:
            filtered.append(csv_path)
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot yearly evaluation and/or bankability metrics in 2x2 subplots "
            "for each site CSV."
        )
    )
    parser.add_argument(
        "--input-csv",
        nargs="+",
        default=None,
        help="Optional one or more yearly evaluation CSV files.",
    )
    parser.add_argument(
        "--input-dir",
        default=INPUT_DIR_DEFAULT,
        help="Directory to search for yearly evaluation CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR_DEFAULT,
        help="Directory where output figures are saved.",
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        default=None,
        help="Optional site names to plot (overrides SITES_TO_PLOT).",
    )
    parser.add_argument(
        "--plot-type",
        choices=["financial", "bankability", "both"],
        default="both",
        help="Which metrics to plot: 'financial' (NPV, Revenue, etc.), "
             "'bankability' (DSCR, LLCR, etc.), or 'both'.",
    )
    args = parser.parse_args()

    csv_files = _resolve_input_files(args.input_csv, args.input_dir)
    sites_to_plot = args.sites if args.sites is not None else SITES_TO_PLOT
    csv_files = _filter_files_by_sites(csv_files, sites_to_plot)

    if not csv_files:
        raise FileNotFoundError(
            "No matching yearly evaluation CSV files found for "
            "selected site(s). "
            "Update SITES_TO_PLOT in this script or pass --sites."
        )

    print("Generating plots...")
    for csv_path in csv_files:
        if args.plot_type in ("financial", "both"):
            out_fn = plot_site_file(csv_path, args.output_dir, metrics=METRICS, metric_type="yearly")
            print(f"Saved: {out_fn}")
        if args.plot_type in ("bankability", "both"):
            out_fn = plot_site_file(csv_path, args.output_dir, metrics=BANKABILITY_METRICS, metric_type="bankability")
            print(f"Saved: {out_fn}")


if __name__ == "__main__":
    main()
