import argparse
import glob
import os
import math

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
        "key": "break_even_ppa",
        "title": "Break-even PPA",
        "ylabel": "EUR/MWh",
        "color": "tab:orange",
        "style": "line",
        "candidates": [
            "Break-even PPA price [Euro/MWh]",
            "Break-even PPA",
        ],
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
SITES_TO_PLOT = []  # Example: plot only Golfe_du_Lion site. Set to [] to plot all sites in input dir.
INPUT_DIR_DEFAULT = os.path.join("HPP", "Evaluations", "P20")  # Change to "P50" if you want to plot P50 evaluations instead of P20.
OUTPUT_DIR_DEFAULT = os.path.join("HPP", "Evaluations", "P20", "plots")


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


def _plot_bankability_dual_axis(df, site_name, x, year_labels, output_dir):
    dscr_col = _find_column(df, ["DSCR [-]", "DSCR"])
    headroom_col = _find_column(
        df,
        ["Debt Headroom [MEuro]", "Debt Headroom"],
    )

    fig, ax_left = plt.subplots(figsize=(14, 6), constrained_layout=True)
    ax_right = ax_left.twinx()

    line_handles = []
    line_labels = []
    headroom_min = None
    headroom_max = None

    if dscr_col is not None:
        dscr = pd.to_numeric(df[dscr_col], errors="coerce").to_numpy()
        left_line = ax_left.plot(
            x,
            dscr,
            marker="o",
            linewidth=2.0,
            color="tab:blue",
            label="DSCR",
        )[0]
        line_handles.append(left_line)
        line_labels.append("DSCR")
    else:
        ax_left.text(0.5, 0.5, "DSCR column not found", ha="center", va="center")

    if headroom_col is not None:
        headroom = pd.to_numeric(df[headroom_col], errors="coerce").to_numpy()
        finite_headroom = headroom[np.isfinite(headroom)]
        if finite_headroom.size > 0:
            headroom_min = float(np.min(finite_headroom))
            headroom_max = float(np.max(finite_headroom))
        right_bars = ax_right.bar(
            x,
            headroom,
            color="tab:green",
            alpha=0.35,
            width=0.55,
            label="Debt Headroom",
        )
        line_handles.append(right_bars[0])
        line_labels.append("Debt Headroom")
    else:
        ax_right.text(
            0.5,
            0.5,
            "Debt Headroom column not found",
            ha="center",
            va="center",
            transform=ax_right.transAxes,
        )

    ax_left.set_ylabel("DSCR [-]", color="tab:blue")
    ax_right.set_ylabel("Debt Headroom [MEuro]", color="tab:green")
    ax_left.set_xlabel("Year")
    ax_left.set_title(f"Bankability - DSCR and Debt Headroom - {site_name}")
    ax_left.grid(True, alpha=0.25)

    if (
        headroom_min is not None
        and headroom_max is not None
        and headroom_min < 0
    ):
        step = 50.0
        y_min = step * math.floor(headroom_min / step)
        y_max = step * math.ceil(headroom_max / step)
        if y_max <= y_min:
            y_max = y_min + step
        ax_right.set_ylim(y_min, y_max)
        ax_right.set_yticks(np.arange(y_min, y_max + step, step))
    elif headroom_max is not None and headroom_max > 0:
        step = 50.0
        y_min = 0.0
        y_max = step * math.ceil(headroom_max / step)
        if y_max <= y_min:
            y_max = y_min + step
        ax_right.set_ylim(y_min, y_max)
        ax_right.set_yticks(np.arange(y_min, y_max + step, step))

    ax_left.set_xticks(x)
    year_num = pd.to_numeric(year_labels, errors="coerce")
    year_short = []
    for val in year_num:
        if pd.notna(val):
            year_short.append(f"{int(val) % 100:02d}")
        else:
            year_short.append("")
    ax_left.set_xticklabels(year_short, rotation=35)

    if line_handles:
        ax_left.legend(line_handles, line_labels, loc="best")

    os.makedirs(output_dir, exist_ok=True)
    out_fn = os.path.join(output_dir, f"{site_name}_bankability_metrics.png")
    fig.savefig(out_fn, dpi=160)
    plt.close(fig)
    return out_fn


def plot_site_file(csv_path, output_dir, metrics=None, metric_type="yearly"):
    if metrics is None:
        metrics = METRICS

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Input CSV has no rows: {csv_path}")

    site_name = _infer_site_name(df, csv_path)
    x, year_labels = _prepare_year_axis(df)

    if metric_type == "bankability":
        return _plot_bankability_dual_axis(
            df=df,
            site_name=site_name,
            x=x,
            year_labels=year_labels,
            output_dir=output_dir,
        )

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
