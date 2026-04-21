# File for plotting and tabulating input data.

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import seaborn as sns



if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from HPP.HelperFunctions.DataLoader import load_examples_sites, load_input_timeseries


def _resolve_column(df: pd.DataFrame, column_name: str) -> str:
    """Resolve column name case-insensitively and raise if not found."""
    exact_matches = [c for c in df.columns if c == column_name]
    if exact_matches:
        return exact_matches[0]

    lower_matches = [c for c in df.columns if c.lower() == column_name.lower()]
    if lower_matches:
        return lower_matches[0]

    raise KeyError(f"Column '{column_name}' not found in input time-series data")


def _format_site_name(name: str) -> str:
    """Convert site names to title-style labels for table columns."""
    return name.replace("_", " ").strip()


def _safe_stat(series: pd.Series, stat: str) -> float:
    """Return safe summary statistic as float from a numeric pandas Series."""
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return float("nan")

    if stat == "mean":
        return float(series.mean())
    if stat == "max":
        return float(series.max())
    if stat == "std":
        return float(series.std())

    raise ValueError(f"Unsupported stat '{stat}'")


def _capacity_factor_percent(series: pd.Series) -> float:
    """Convert a 0-1 power/irradiance ratio series to percentage mean."""
    return 100.0 * _safe_stat(series, "mean")


def build_summary_table(
    input_ts_by_site: Dict[int, pd.DataFrame],
    examples_sites: pd.DataFrame,
    wind_col: str = "WS_150",
    solar_col: str = "ghi",
    price_col: str = "price",
    site_ids: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Build summary table for meteorological and price data."""
    if site_ids is None:
        selected_ids = list(input_ts_by_site)
    else:
        selected_ids = list(site_ids)

    table_rows = {
        "Total Years": {},
        "Wind Speed - Mean (m/s)": {},
        "Wind Speed - Max (m/s)": {},
        "Wind Speed - Std Dev (m/s)": {},
        "Solar GHI - Mean (W/m^2)": {},
        "Solar GHI - Max (W/m^2)": {},
        "Solar GHI - Std Dev (W/m^2)": {},
        "Price - Mean": {},
        "Price - Max": {},
        "Price - Std Dev": {},
    }

    include_wind_cf = True
    include_solar_cf = True

    for site_id in selected_ids:
        site_df = input_ts_by_site[site_id]
        site_name = _format_site_name(str(examples_sites.loc[site_id, "name"]))

        wind_col_res = _resolve_column(site_df, wind_col)
        solar_col_res = _resolve_column(site_df, solar_col)
        price_col_res = _resolve_column(site_df, price_col)

        table_rows["Total Years"][site_name] = len(site_df) / (24.0 * 365.0)
        table_rows["Wind Speed - Mean (m/s)"][site_name] = _safe_stat(
            site_df[wind_col_res], "mean"
        )
        table_rows["Wind Speed - Max (m/s)"][site_name] = _safe_stat(
            site_df[wind_col_res], "max"
        )
        table_rows["Wind Speed - Std Dev (m/s)"][site_name] = _safe_stat(
            site_df[wind_col_res], "std"
        )

        table_rows["Solar GHI - Mean (W/m^2)"][site_name] = _safe_stat(
            site_df[solar_col_res], "mean"
        )
        table_rows["Solar GHI - Max (W/m^2)"][site_name] = _safe_stat(
            site_df[solar_col_res], "max"
        )
        table_rows["Solar GHI - Std Dev (W/m^2)"][site_name] = _safe_stat(
            site_df[solar_col_res], "std"
        )

        table_rows["Price - Mean"][site_name] = _safe_stat(
            site_df[price_col_res], "mean"
        )
        table_rows["Price - Max"][site_name] = _safe_stat(
            site_df[price_col_res], "max"
        )
        table_rows["Price - Std Dev"][site_name] = _safe_stat(
            site_df[price_col_res], "std"
        )

        if "WP_150" not in site_df.columns:
            include_wind_cf = False
        if "SP" not in site_df.columns:
            include_solar_cf = False

    if include_wind_cf:
        table_rows["Wind Capacity Factor (%)"] = {}
        for site_id in selected_ids:
            site_name = _format_site_name(str(examples_sites.loc[site_id, "name"]))
            table_rows["Wind Capacity Factor (%)"][site_name] = _capacity_factor_percent(
                input_ts_by_site[site_id]["WP_150"]
            )

    if include_solar_cf:
        table_rows["Solar Capacity Factor (%)"] = {}
        for site_id in selected_ids:
            site_name = _format_site_name(str(examples_sites.loc[site_id, "name"]))
            table_rows["Solar Capacity Factor (%)"][site_name] = _capacity_factor_percent(
                input_ts_by_site[site_id]["SP"]
            )

    table = pd.DataFrame.from_dict(table_rows, orient="index")
    return table.round(2)


def load_and_build_summary_table(
    wind_col: str = "WS_150",
    solar_col: str = "ghi",
    price_col: str = "price",
    site_ids: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Load repository defaults and build the summary table."""
    examples_sites = load_examples_sites()
    input_ts_by_site = load_input_timeseries()
    return build_summary_table(
        input_ts_by_site=input_ts_by_site,
        examples_sites=examples_sites,
        wind_col=wind_col,
        solar_col=solar_col,
        price_col=price_col,
        site_ids=site_ids,
    )


def plot_wind_speed_distribution(
    site_ids: Optional[Iterable[int]] = None,
    wind_col: str = "WS_150",
    figsize: Optional[tuple] = None,
    title: str = "Wind Speed Distribution",
    bins: int = 30,
    save_path: Optional[str] = None,
    manual_site_names: Optional[Iterable[str]] = None, # Added manual control
) -> tuple:
    """
    Create a figure with histograms of wind speed distributions with 
    manual site naming and top-right legends for mean values.
    """
    # Load data
    examples_sites = load_examples_sites()
    input_ts_by_site = load_input_timeseries()

    # Default to first 3 sites if none specified
    if site_ids is None:
        site_ids = [0, 5, 3]

    selected_ids = list(site_ids)
    num_sites = len(selected_ids)

    # Validation for manual names
    if manual_site_names is not None:
        manual_names = list(manual_site_names)
        if len(manual_names) != num_sites:
            raise ValueError("Number of manual names must match number of site IDs.")

    # Set figure size
    if figsize is None:
        figsize = (4 * num_sites, 4)

    fig, axes = plt.subplots(1, num_sites, figsize=figsize)

    # Handle single site (axes won't be an array)
    if num_sites == 1:
        axes = [axes]

    for i, (site_id, ax) in enumerate(zip(selected_ids, axes)):
        if site_id not in input_ts_by_site:
            raise ValueError(f"Site ID {site_id} not found in loaded data")

        site_df = input_ts_by_site[site_id]
        
        # Site name logic: Manual vs Automatic
        if manual_site_names is not None:
            site_name = manual_names[i]
        else:
            site_name = _format_site_name(str(examples_sites.loc[site_id, "name"]))

        # Resolve wind column case-insensitively
        wind_col_res = _resolve_column(site_df, wind_col)
        wind_data = pd.to_numeric(site_df[wind_col_res], errors="coerce").dropna()

        # Create histogram
        ax.hist(wind_data, bins=bins, color="steelblue", edgecolor="black", alpha=0.7)

        # Add mean line and label for the legend
        mean_val = wind_data.mean()
        ax.axvline(
            mean_val, 
            color="red", 
            linestyle="--", 
            linewidth=2,
            label=f"Mean: {mean_val:.2f} m/s"
        )

        # Formatting
        ax.set_xlabel("Wind Speed (m/s)", fontsize=10)
        ax.set_ylabel("Frequency (hours)", fontsize=10)
        ax.set_title(f"Site: {site_name}", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Force legend to top right
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    # Save plot if save_path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename_base = os.path.join(save_path, "wind_speed_distribution")
        fig.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight")
        print(f"Plot saved to:\n  {filename_base}.png")

    return fig, axes

def plot_wind_speed_distribution(
    site_ids: Optional[Iterable[int]] = None,
    wind_col: str = "WS_150",
    figsize: Optional[tuple] = None,
    title: str = "Wind Speed Distribution",
    bins: int = 30,
    save_path: Optional[str] = None,
    manual_site_names: Optional[Iterable[str]] = None,
) -> tuple:
    """
    Create a figure with histograms of wind speed distributions with 
    standardized x-axis (0-40, steps of 5) and top-right legends.
    """
    # Load data using your project's data loaders
    examples_sites = load_examples_sites()
    input_ts_by_site = load_input_timeseries()

    if site_ids is None:
        site_ids = [0, 5, 3]

    selected_ids = list(site_ids)
    num_sites = len(selected_ids)

    # Validate manual names if provided
    if manual_site_names is not None:
        manual_names = list(manual_site_names)
        if len(manual_names) != num_sites:
            raise ValueError("Number of manual names must match number of site IDs.")

    if figsize is None:
        figsize = (4 * num_sites, 4)

    fig, axes = plt.subplots(1, num_sites, figsize=figsize)
    if num_sites == 1:
        axes = [axes]

    for i, (site_id, ax) in enumerate(zip(selected_ids, axes)):
        if site_id not in input_ts_by_site:
            raise ValueError(f"Site ID {site_id} not found in loaded data")

        site_df = input_ts_by_site[site_id]
        
        # Site name logic
        if manual_site_names is not None:
            site_name = manual_names[i]
        else:
            # Fallback to internal formatter if no manual names
            site_name = _format_site_name(str(examples_sites.loc[site_id, "name"]))

        # Resolve column and clean data
        wind_col_res = _resolve_column(site_df, wind_col)
        wind_data = pd.to_numeric(site_df[wind_col_res], errors="coerce").dropna()

        # Create histogram
        ax.hist(wind_data, bins=bins, color="steelblue", edgecolor="black", alpha=0.7)

        # Plot mean line
        mean_val = wind_data.mean()
        ax.axvline(
            mean_val, 
            color="red", 
            linestyle="--", 
            linewidth=2, 
            label=f"Mean: {mean_val:.2f} m/s"
        )

        # --- X-AXIS STANDARDIZATION ---
        ax.set_xlim(0, 40)  # Forces same scale for all subplots
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5)) # Ticks at 0, 5, 10, etc.
        # ------------------------------

        # Formatting
        ax.set_xlabel("Wind Speed at 150m (m/s)", fontsize=10)
        ax.set_ylabel("Frequency (hours)", fontsize=10)
        ax.set_title(f"Site: {site_name}", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        
        # Legend at top right
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename_base = os.path.join(save_path, "wind_speed_distribution")
        fig.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {filename_base}.png")

    return fig, axes


def plot_solar_irradiance_distribution(
    site_ids: Optional[Iterable[int]] = None,
    solar_col: str = "ghi",
    figsize: Optional[tuple] = None,
    title: str = "Solar Irradiance (GHI) Distribution",
    bins: int = 30,
    save_path: Optional[str] = None,
    manual_site_names: Optional[Iterable[str]] = None,
) -> tuple:
    """
    Create a figure with histograms of solar irradiance distributions 
    with mean values shown in a top-right legend.
    """
    examples_sites = load_examples_sites()
    input_ts_by_site = load_input_timeseries()

    if site_ids is None:
        site_ids = [0, 5, 3]

    selected_ids = list(site_ids)
    num_sites = len(selected_ids)

    if manual_site_names is not None:
        manual_names = list(manual_site_names)
        if len(manual_names) != num_sites:
            raise ValueError("Number of manual names must match number of site IDs.")

    if figsize is None:
        figsize = (4 * num_sites, 4)

    fig, axes = plt.subplots(1, num_sites, figsize=figsize)
    if num_sites == 1:
        axes = [axes]

    for i, (site_id, ax) in enumerate(zip(selected_ids, axes)):
        if site_id not in input_ts_by_site:
            raise ValueError(f"Site ID {site_id} not found in loaded data")

        site_df = input_ts_by_site[site_id]
        
        # Site name logic
        if manual_site_names is not None:
            site_name = manual_names[i]
        else:
            site_name = _format_site_name(str(examples_sites.loc[site_id, "name"]))

        solar_col_res = _resolve_column(site_df, solar_col)
        solar_data = pd.to_numeric(site_df[solar_col_res], errors="coerce").dropna()
        solar_data_visual = solar_data[solar_data > 0]
        if solar_data_visual.empty:
            solar_data_visual = solar_data

        ax.hist(
            solar_data_visual,
            bins=bins,
            color="darkorange",
            edgecolor="black",
            alpha=0.7,
        )

        # Plot mean line and add label for the legend
        mean_val = solar_data.mean()
        ax.axvline(
            mean_val, 
            color="red", 
            linestyle="--", 
            linewidth=2, 
            label=f"Mean: {mean_val:.2f} W/m^2"
        )

        # Formatting
        ax.set_xlabel("Global Horizontal Irradiance (W/m^2)", fontsize=10)
        ax.set_ylabel("Frequency (hours)", fontsize=10)
        ax.set_title(f"Site: {site_name}", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Force legend to top right
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename_base = os.path.join(save_path, "solar_irradiance_distribution")
        fig.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight")

    return fig, axes


def plot_daily_mean_wind_and_solar(
    site_ids: Optional[Iterable[int]] = None,
    wind_col: str = "WS_150",
    solar_col: str = "ghi",
    figsize: Optional[tuple] = None,
    target_year: int = 2012,
    title_prefix: str = "Daily Mean Wind Speed and Solar Irradiance",
    save_path: Optional[str] = None,
    manual_site_names: Optional[Iterable[str]] = None,  # Added for manual control
) -> tuple:
    """
    Create stacked time-series plots with custom site names and top-right legends.
    """
    examples_sites = load_examples_sites()
    input_ts_by_site = load_input_timeseries()

    if site_ids is None:
        site_ids = [0, 5, 3]

    selected_ids = list(site_ids)
    num_sites = len(selected_ids)

    # Convert manual_site_names to a list if provided
    if manual_site_names is not None:
        manual_names = list(manual_site_names)
        if len(manual_names) != num_sites:
            raise ValueError("Number of manual names must match number of site IDs.")

    if figsize is None:
        figsize = (12, 2.4 * num_sites)

    fig, axes = plt.subplots(num_sites, 1, figsize=figsize, sharex=True)
    if num_sites == 1:
        axes = [axes]

    wind_color = "#5b9bd5"
    solar_color = "orange"

    for i, (site_id, ax1) in enumerate(zip(selected_ids, axes)):
        if site_id not in input_ts_by_site:
            raise ValueError(f"Site ID {site_id} not found in loaded data")

        site_df = input_ts_by_site[site_id].copy()
        
        # Logic to choose between manual and automatic names
        if manual_site_names is not None:
            site_name = manual_names[i]
        else:
            site_name = _format_site_name(str(examples_sites.loc[site_id, "name"]))

        wind_col_res = _resolve_column(site_df, wind_col)
        solar_col_res = _resolve_column(site_df, solar_col)

        if not isinstance(site_df.index, pd.DatetimeIndex):
            site_df.index = pd.to_datetime(
                site_df.index,
                errors="coerce",
                dayfirst=True,
            )
        site_df = site_df[site_df.index.notna()]

        daily = site_df[[wind_col_res, solar_col_res]].copy()
        daily[wind_col_res] = pd.to_numeric(daily[wind_col_res], errors="coerce")
        daily[solar_col_res] = pd.to_numeric(daily[solar_col_res], errors="coerce")
        daily = daily.resample("D").mean().dropna(how="all")

        daily_year = daily[daily.index.year == target_year].sort_index()
        if daily_year.empty:
            raise ValueError(
                f"No daily data available for year {target_year} at site '{site_name}'"
            )

        ax1.plot(
            daily_year.index,
            daily_year[wind_col_res],
            color=wind_color,
            linewidth=0.8,
            label="Wind Speed",
        )
        ax1.set_ylabel("Daily Mean Wind Speed (m/s)", color=wind_color, fontsize=8)
        ax1.tick_params(axis="y", labelcolor=wind_color, labelsize=7)
        ax1.grid(True, linestyle="-", linewidth=0.4, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(
            daily_year.index,
            daily_year[solar_col_res],
            color=solar_color,
            linewidth=0.8,
            label="GHI",
        )
        ax2.set_ylabel(
            "Daily Mean GHI (W/m^2)",
            color=solar_color,
            fontsize=8,
        )
        ax2.tick_params(axis="y", labelcolor=solar_color, labelsize=7)

        # Forced legend to top right
        line1 = ax1.get_lines()[0]
        line2 = ax2.get_lines()[0]
        ax1.legend(
            [line1, line2], 
            ["Wind Speed", "GHI"], 
            fontsize=6,
            loc="upper right"
        )

        ax1.set_title(
            f"{title_prefix} | Site: {site_name}",
            fontsize=8,
            fontweight="bold",
        )

        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax1.tick_params(axis="x", labelsize=7)

        if i < num_sites - 1:
            ax1.tick_params(axis="x", labelbottom=True)

    axes[-1].set_xlabel("Month", fontsize=8, fontweight="bold")
    fig.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename_base = os.path.join(
            save_path,
            "daily_mean_wind_speed_and_solar_irradiance",
        )
        fig.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight")
        print(f"Plot saved to:\n  {filename_base}.png")

    return fig, axes


def plot_daily_mean_wind_and_solar_overlay(
    site_ids: Optional[Iterable[int]] = None,
    wind_col: str = "WS_150",
    solar_col: str = "ghi",
    figsize: Optional[tuple] = None,
    target_year: int = 2012,
    title: str = "Daily Mean Wind Speed and Solar Irradiance (Overlay)",
    save_path: Optional[str] = None,
) -> tuple:
    """
    Plot daily mean wind speed and solar irradiance for two sites on a single plot (no subplots).
    Default: Thetys and Sud Atlantique.
    """
    import matplotlib.dates as mdates
    examples_sites = load_examples_sites()
    input_ts_by_site = load_input_timeseries()

    # Default to Thetys (1) and Sud Atlantique (5) if not specified
    if site_ids is None:
        site_ids = [1, 5]
    if len(site_ids) != 2:
        raise ValueError("Exactly two site IDs must be provided.")

    if figsize is None:
        figsize = (12, 5)

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    colors = ["#5b9bd5", "#2ca02c"]  # blue, green for wind
    solar_colors = ["orange", "red"]  # orange, red for solar
    labels = []

    for idx, site_id in enumerate(site_ids):
        if site_id not in input_ts_by_site:
            raise ValueError(f"Site ID {site_id} not found in loaded data")
        site_df = input_ts_by_site[site_id].copy()
        site_name = _format_site_name(str(examples_sites.loc[site_id, "name"]))

        wind_col_res = _resolve_column(site_df, wind_col)
        solar_col_res = _resolve_column(site_df, solar_col)

        if not isinstance(site_df.index, pd.DatetimeIndex):
            site_df.index = pd.to_datetime(
                site_df.index,
                errors="coerce",
                dayfirst=True,
            )
        site_df = site_df[site_df.index.notna()]

        daily = site_df[[wind_col_res, solar_col_res]].copy()
        daily[wind_col_res] = pd.to_numeric(daily[wind_col_res], errors="coerce")
        daily[solar_col_res] = pd.to_numeric(daily[solar_col_res], errors="coerce")
        daily = daily.resample("D").mean().dropna(how="all")
        daily_year = daily[daily.index.year == target_year].sort_index()
        if daily_year.empty:
            raise ValueError(
                f"No daily data available for year {target_year} at site '{site_name}'"
            )

        # Wind on left axis
        l1, = ax1.plot(
            daily_year.index,
            daily_year[wind_col_res],
            color=colors[idx],
            linewidth=1.2,
            label=f"{site_name} Wind Speed",
        )
        # Solar on right axis
        l2, = ax2.plot(
            daily_year.index,
            daily_year[solar_col_res],
            color=solar_colors[idx],
            linewidth=1.2,
            linestyle="--",
            label=f"{site_name} Solar Irradiance",
        )
        labels.append((l1, l2))

    ax1.set_ylabel("Daily Mean Wind Speed (m/s)", color="#5b9bd5", fontsize=10)
    ax2.set_ylabel("Daily Mean GHI (W/m^2)", color="orange", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="#5b9bd5", labelsize=9)
    ax2.tick_params(axis="y", labelcolor="orange", labelsize=9)
    ax1.grid(True, linestyle="-", linewidth=0.4, alpha=0.3)

    ax1.set_title(title + f" ({target_year})", fontsize=12, fontweight="bold")
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax1.tick_params(axis="x", labelsize=9)
    ax1.set_xlabel("Month", fontsize=10, fontweight="bold")

    # Build legend
    handles = []
    legend_labels = []
    for l1, l2 in labels:
        handles.extend([l1, l2])
        legend_labels.extend([l1.get_label(), l2.get_label()])
    ax1.legend(handles, legend_labels, fontsize=9, loc="upper left")

    fig.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename_base = os.path.join(
            save_path,
            "daily_mean_wind_speed_and_solar_irradiance_overlay",
        )
        fig.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight")
        print(f"Plot saved to:\n  {filename_base}.png")

    return fig, (ax1, ax2)


def _plot_daily_mean_pair(
    site_ids: Optional[Iterable[int]],
    left_col: str,
    right_col: str,
    left_label: str,
    right_label: str,
    left_color: str,
    right_color: str,
    left_ylabel: str,
    right_ylabel: str,
    figsize: Optional[tuple],
    target_year: int,
    title_prefix: str,
    save_path: Optional[str],
    save_filename: str,
) -> tuple:
    """Create stacked time-series plots of daily mean values for a variable pair."""
    examples_sites = load_examples_sites()
    input_ts_by_site = load_input_timeseries()

    if site_ids is None:
        site_ids = [0, 5, 3]

    selected_ids = list(site_ids)
    num_sites = len(selected_ids)

    if figsize is None:
        figsize = (12, 2.4 * num_sites)

    fig, axes = plt.subplots(num_sites, 1, figsize=figsize, sharex=True)
    if num_sites == 1:
        axes = [axes]

    for i, (site_id, ax1) in enumerate(zip(selected_ids, axes)):
        if site_id not in input_ts_by_site:
            raise ValueError(f"Site ID {site_id} not found in loaded data")

        site_df = input_ts_by_site[site_id].copy()
        site_name = _format_site_name(str(examples_sites.loc[site_id, "name"]))

        left_col_res = _resolve_column(site_df, left_col)
        right_col_res = _resolve_column(site_df, right_col)

        if not isinstance(site_df.index, pd.DatetimeIndex):
            site_df.index = pd.to_datetime(
                site_df.index,
                errors="coerce",
                dayfirst=True,
            )
        site_df = site_df[site_df.index.notna()]

        daily = site_df[[left_col_res, right_col_res]].copy()
        daily[left_col_res] = pd.to_numeric(daily[left_col_res], errors="coerce")
        daily[right_col_res] = pd.to_numeric(daily[right_col_res], errors="coerce")
        daily = daily.resample("D").mean().dropna(how="all")

        daily_year = daily[daily.index.year == target_year].sort_index()
        if daily_year.empty:
            raise ValueError(
                f"No daily data available for year {target_year} at site '{site_name}'"
            )

        ax1.plot(
            daily_year.index,
            daily_year[left_col_res],
            color=left_color,
            linewidth=0.8,
            label=left_label,
        )
        ax1.set_ylabel(left_ylabel, color=left_color, fontsize=8)
        ax1.tick_params(axis="y", labelcolor=left_color, labelsize=7)
        ax1.grid(True, linestyle="-", linewidth=0.4, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(
            daily_year.index,
            daily_year[right_col_res],
            color=right_color,
            linewidth=0.8,
            label=right_label,
        )
        ax2.set_ylabel(right_ylabel, color=right_color, fontsize=8)
        ax2.tick_params(axis="y", labelcolor=right_color, labelsize=7)

        line1 = ax1.get_lines()[0]
        line2 = ax2.get_lines()[0]
        ax1.legend([line1, line2], [left_label, right_label], fontsize=6)

        ax1.set_title(
            f"{title_prefix} ({target_year}) | Site: {site_name}",
            fontsize=8,
            fontweight="bold",
        )

        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax1.tick_params(axis="x", labelsize=7)

        if i < num_sites - 1:
            ax1.tick_params(axis="x", labelbottom=True)

    axes[-1].set_xlabel("Month", fontsize=8, fontweight="bold")
    fig.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename_base = os.path.join(save_path, save_filename)
        fig.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight")
        print(f"Plot saved to:\n  {filename_base}.png")

    return fig, axes


def plot_daily_mean_wind_and_price(
    site_ids: Optional[Iterable[int]] = None,
    wind_col: str = "WS_150",
    price_col: str = "price",
    figsize: Optional[tuple] = None,
    target_year: int = 2012,
    title_prefix: str = "Daily Mean Wind Speed and Electricity Price",
    save_path: Optional[str] = None,
) -> tuple:
    """Create stacked plots of daily mean wind speed and electricity price."""
    return _plot_daily_mean_pair(
        site_ids=site_ids,
        left_col=wind_col,
        right_col=price_col,
        left_label="Wind Speed",
        right_label="Price",
        left_color="#5b9bd5",
        right_color="seagreen",
        left_ylabel="Daily Mean Wind Speed (m/s)",
        right_ylabel="Daily Mean Price (EUR/MW)",
        figsize=figsize,
        target_year=target_year,
        title_prefix=title_prefix,
        save_path=save_path,
        save_filename="daily_mean_wind_speed_and_price",
    )


def plot_daily_mean_solar_and_price(
    site_ids: Optional[Iterable[int]] = None,
    solar_col: str = "ghi",
    price_col: str = "price",
    figsize: Optional[tuple] = None,
    target_year: int = 2012,
    title_prefix: str = "Daily Mean Solar Irradiance and Electricity Price",
    save_path: Optional[str] = None,
) -> tuple:
    """Create stacked plots of daily mean solar irradiance and electricity price."""
    return _plot_daily_mean_pair(
        site_ids=site_ids,
        left_col=solar_col,
        right_col=price_col,
        left_label="Solar Irradiance",
        right_label="Price",
        left_color="orange",
        right_color="seagreen",
        left_ylabel="Daily Mean GHI (W/m^2)",
        right_ylabel="Daily Mean Price (EUR/MW)",
        figsize=figsize,
        target_year=target_year,
        title_prefix=title_prefix,
        save_path=save_path,
        save_filename="daily_mean_solar_irradiance_and_price",
    )


import os
from typing import Iterable, Optional, Dict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Assuming these helper functions are defined elsewhere, as in previous code
# from HPP.HelperFunctions.DataLoader import load_examples_sites, load_input_timeseries
# def _resolve_column(df, col): ...
# def _format_site_name(name): ...

def plot_mean_hourly_resource_and_price(
    site_ids: Optional[Iterable[int]] = None,
    wind_col: str = "WS_150",
    solar_col: str = "ghi",
    price_col: str = "price",
    figsize: Optional[tuple] = None,
    target_year: Optional[int] = None,
    save_path: Optional[str] = None,
    manual_site_names: Optional[Iterable[str]] = None,
) -> tuple:
    """
    Create side-by-side hourly mean plots with custom academic styling.
    Colors matched to provided sample image, markers removed, and refined labels.
    """
    examples_sites = load_examples_sites()
    input_ts_by_site = load_input_timeseries()

    if site_ids is None:
        site_ids = [0, 5, 3]

    selected_ids = list(site_ids)
    num_sites = len(selected_ids)

    if manual_site_names is not None:
        manual_names = list(manual_site_names)
        if len(manual_names) != num_sites:
            raise ValueError("Number of manual names must match number of site IDs.")

    if figsize is None:
        figsize = (13, 5.5)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax_wind, ax_solar = axes[0], axes[1]
    ax_wind_price = ax_wind.twinx()
    ax_solar_price = ax_solar.twinx()

    # CUSTOM COLORS FROM SAMPLE IMAGE
    # Light Teal/Cyan, Mid Blue, Muted Brown/Copper
    site_colors = ["#43D1D9", "#4B86C2", "#9C7667"]
    
    hours = list(range(24))
    site_handles = []

    for i, site_id in enumerate(selected_ids):
        if site_id not in input_ts_by_site:
            continue
            
        color = site_colors[i % len(site_colors)]
        site_df = input_ts_by_site[site_id].copy()
        
        # Site name logic
        site_name = manual_names[i] if manual_site_names else _format_site_name(str(examples_sites.loc[site_id, "name"]))

        # Data processing
        wind_col_res = _resolve_column(site_df, wind_col)
        solar_col_res = _resolve_column(site_df, solar_col)
        price_col_res = _resolve_column(site_df, price_col)

        if not isinstance(site_df.index, pd.DatetimeIndex):
            site_df.index = pd.to_datetime(site_df.index, errors="coerce", dayfirst=True)
        
        period_df = site_df[site_df.index.year == target_year].copy() if target_year else site_df
        hourly = period_df[[wind_col_res, solar_col_res, price_col_res]].apply(pd.to_numeric, errors='coerce').groupby(period_df.index.hour).mean().reindex(hours)

        # Plot Wind & Solar (Solid lines, NO markers)
        ax_wind.plot(hours, hourly[wind_col_res], color=color, linewidth=2.0, marker=None)
        ax_solar.plot(hours, hourly[solar_col_res], color=color, linewidth=2.0, marker=None)

        # Plot Price (Dotted/Dashed lines, NO markers)
        ax_wind_price.plot(hours, hourly[price_col_res], color=color, linestyle='--', linewidth=1.5, alpha=0.7, marker=None)
        ax_solar_price.plot(hours, hourly[price_col_res], color=color, linestyle='--', linewidth=1.5, alpha=0.7, marker=None)

        site_handles.append(mlines.Line2D([], [], color=color, linewidth=2, label=site_name))

    # --- ACADEMIC FORMATTING ---
    ax_wind.set_ylabel("Mean Wind Speed at 150m [m/s]", fontweight="bold", fontsize=9)
    ax_solar.set_ylabel("Mean GHI [W/m²]", fontweight="bold", fontsize=9)
    ax_solar_price.set_ylabel("Mean Electricity Price [€/MWh]", fontweight="bold", fontsize=9)
    
    # ACADEMIC TITLES
    ax_wind.set_title("Diurnal Profiles of Wind Speed and Price", fontweight="bold", fontsize=10)
    ax_solar.set_title("Diurnal Profiles of Solar Irradiance and Price", fontweight="bold", fontsize=10)

    for ax in [ax_wind, ax_solar]:
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
        ax.set_xlabel("Hour of Day", fontweight="bold", fontsize=9)
        
        # SHOW EVERY SECOND HOUR (0, 2, 4...)
        ax.set_xticks(hours[::2]) 
        ax.set_xlim(0, 23)

    # --- LEGEND CONSTRUCTION ---
    resource_proxy = mlines.Line2D([], [], color='#555555', linestyle='-', label='Resource (Wind/Solar)')
    price_proxy = mlines.Line2D([], [], color='#555555', linestyle='--', label='Electricity Price')
    
    fig.subplots_adjust(bottom=0.25, wspace=0.35)
    
    all_handles = site_handles + [resource_proxy, price_proxy]
    
    fig.legend(
        handles=all_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=len(site_handles) + 1, 
        frameon=False,
        fontsize=8
    )

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, "academic_hourly_profiles.png"), dpi=300, bbox_inches="tight")

    return fig, axes



def plot_mean_hourly_resource_and_price(
    site_ids: Optional[Iterable[int]] = None,
    wind_col: str = "WS_150",
    solar_col: str = "ghi",
    price_col: str = "price",
    figsize: Optional[tuple] = None,
    target_year: Optional[int] = None,
    save_path: Optional[str] = None,
    manual_site_names: Optional[Iterable[str]] = None,
) -> tuple:
    """
    Academic Diurnal Profile Plot.
    - Solid lines for resources, dashed for price.
    - No markers.
    - Colors matched to teal, blue, and copper palette.
    - Every second hour labeled on X-axis.
    """
    # Load data
    examples_sites = load_examples_sites()
    input_ts_by_site = load_input_timeseries()

    if site_ids is None:
        site_ids = [0, 5, 3]

    selected_ids = list(site_ids)
    num_sites = len(selected_ids)

    # Handle manual names passed in the function call
    if manual_site_names is not None:
        manual_names = list(manual_site_names)
        if len(manual_names) != num_sites:
            raise ValueError("Number of manual names must match number of site IDs.")

    if figsize is None:
        figsize = (13, 5.5)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax_wind, ax_solar = axes[0], axes[1]
    ax_wind_price = ax_wind.twinx()
    ax_solar_price = ax_solar.twinx()

    # ACADEMIC COLOR PALETTE (Teal, Blue, Copper)
    site_colors = ["#43D1D9", "#4B86C2", "#9C7667"]
    hours = list(range(24))
    site_handles = []

    for i, site_id in enumerate(selected_ids):
        if site_id not in input_ts_by_site:
            continue
            
        color = site_colors[i % len(site_colors)]
        site_df = input_ts_by_site[site_id].copy()
        
        # Site name logic
        if manual_site_names is not None:
            site_name = manual_names[i]
        else:
            site_name = _format_site_name(str(examples_sites.loc[site_id, "name"]))

        # Column resolution
        wind_col_res = _resolve_column(site_df, wind_col)
        solar_col_res = _resolve_column(site_df, solar_col)
        price_col_res = _resolve_column(site_df, price_col)

        if not isinstance(site_df.index, pd.DatetimeIndex):
            site_df.index = pd.to_datetime(site_df.index, errors="coerce", dayfirst=True)
        
        # Data Filtering and Resampling
        period_df = site_df[site_df.index.year == target_year].copy() if target_year else site_df
        hourly = period_df[[wind_col_res, solar_col_res, price_col_res]].apply(pd.to_numeric, errors='coerce').groupby(period_df.index.hour).mean().reindex(hours)

        # Plot Wind & Solar (Solid lines, No markers)
        ax_wind.plot(hours, hourly[wind_col_res], color=color, linewidth=2.0, marker=None)
        ax_solar.plot(hours, hourly[solar_col_res], color=color, linewidth=2.0, marker=None)

        # Plot Price (Dashed lines, No markers)
        ax_wind_price.plot(hours, hourly[price_col_res], color=color, linestyle='--', linewidth=1.5, alpha=0.7, marker=None)
        ax_solar_price.plot(hours, hourly[price_col_res], color=color, linestyle='--', linewidth=1.5, alpha=0.7, marker=None)

        # Create handles for the site legend
        site_handles.append(mlines.Line2D([], [], color=color, linewidth=2, label=site_name))

    # --- ACADEMIC FORMATTING ---
    ax_wind.set_ylabel("Mean Wind Speed at 150m [m/s]", fontweight="bold", fontsize=9)
    ax_solar.set_ylabel("Mean GHI [W/m²]", fontweight="bold", fontsize=9)
    ax_solar_price.set_ylabel("Mean Electricity Price [€/MWh]", fontweight="bold", fontsize=9)
    
    # ACADEMIC TITLES
    ax_wind.set_title("Diurnal Profiles of Wind Speed and Electricity Price", fontweight="bold", fontsize=10)
    ax_solar.set_title("Diurnal Profiles of Solar Irradiance and Electricity Price", fontweight="bold", fontsize=10)

    for ax in [ax_wind, ax_solar]:
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
        ax.set_xlabel("Hour of Day", fontweight="bold", fontsize=9)
        
        # Labels for every second hour: 0, 2, 4, ..., 22
        ax.set_xticks(hours[::2]) 
        ax.set_xlim(0, 23)

    # --- SINGLE LINE LEGEND ---
    resource_proxy = mlines.Line2D([], [], color='#555555', linestyle='-', label='Resource (Wind/Solar)')
    price_proxy = mlines.Line2D([], [], color='#555555', linestyle='--', label='Price (Electricity)')
    
    all_handles = site_handles + [resource_proxy, price_proxy]
    
    fig.subplots_adjust(bottom=0.2, wspace=0.35)
    
    fig.legend(
        handles=all_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(all_handles), # This forces everything onto one line
        frameon=False,
        fontsize=8,
        columnspacing=1.0 # Adjusts space between the items to fit nicely
    )

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, "diurnal_resource_price_profiles.png")
        fig.savefig(save_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_file}")

    return fig, axes


def scatterhist_resource_vs_price(
    df: pd.DataFrame,
    resource_col: str,
    price_col: str = "price",
    site_name: str = "",
    bins: int = 30,
    figsize: tuple = (7, 7),
    color: str = "tab:blue",
    alpha: float = 0.5,
    title: str = None,
):
    """
    Create a scatter plot with marginal histograms for resource vs. price.
    """
    x = pd.to_numeric(df[resource_col], errors="coerce")
    y = pd.to_numeric(df[price_col], errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]


    g = sns.JointGrid(x=x, y=y, height=figsize[0])
    g.plot_joint(sns.scatterplot, color=color, alpha=alpha)
    g.plot_marginals(sns.histplot, color=color, bins=bins, kde=True, alpha=0.7)

    g.set_axis_labels(resource_col, price_col)
    # Set the title if provided, else use site_name if available
    if title:
        plt.suptitle(title, y=1.02)
    elif site_name:
        plt.suptitle(f"{site_name}: {resource_col} vs. {price_col}", y=1.02)
    # Call tight_layout after suptitle to avoid title being cut off
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    
def plot_all_sites_scatterhist_resource_vs_price(
    input_ts_by_site,
    examples_sites,
    resource_col: str,
    price_col: str = "price",
    figsize: tuple = (14, 10),
    color: str = "tab:blue",
    alpha: float = 0.5,
    resource_label: str = None,
    price_label: str = None,
    save_path: str = None,
    plot_type: str = "wind",  # "wind" or "solar"
):
    """
    Plot 3x2 subplots for all 6 sites: resource vs price scatterhist for each site.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    site_ids = [0, 1, 2, 3, 4, 5]
    site_names = [_format_site_name(str(examples_sites.loc[site_id, "name"])) for site_id in site_ids]
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    axes = axes.flatten()

    for i, site_id in enumerate(site_ids):
        ax = axes[i]
        site_df = input_ts_by_site[site_id].copy()
        site_name = site_names[i]
        # Resolve columns case-insensitively
        res_col = _resolve_column(site_df, resource_col)
        prc_col = _resolve_column(site_df, price_col)
        x = pd.to_numeric(site_df[res_col], errors="coerce")
        y = pd.to_numeric(site_df[prc_col], errors="coerce")
        mask = x.notna() & y.notna()
        x = x[mask]
        y = y[mask]
        # Scatter
        ax.scatter(x, y, color=color, alpha=alpha, s=10)
        # Marginal histograms
        ax_histx = ax.inset_axes([0, 1.05, 1, 0.18], sharex=ax)
        ax_histy = ax.inset_axes([1.05, 0, 0.18, 1], sharey=ax)
        ax_histx.hist(x, bins=30, color=color, alpha=0.7)
        ax_histy.hist(y, bins=30, color=color, alpha=0.7, orientation="horizontal")
        ax_histx.axis("off")
        ax_histy.axis("off")
        # Labels
        ax.set_xlabel(resource_label or resource_col)
        ax.set_ylabel(price_label or price_col)
        ax.set_title(site_name, fontsize=10)

    # Remove empty axes if any
    for j in range(len(site_ids), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{plot_type.capitalize()} vs Price for All Sites", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename_base = os.path.join(
            save_path,
            "Wind_vs_Price_ScatterHist_All_Sites" if plot_type == "wind" else "Solar_vs_Price_ScatterHist_All_Sites",
        )
        fig.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight")
        print(f"Plot saved to:\n  {filename_base}.png")
    
    return fig, axes


def plot_monthly_market_prices(
    site_ids: Optional[Iterable[int]] = None,
    price_col: str = "price",
    figsize: Optional[tuple] = (12, 6),
    save_path: Optional[str] = None,
    manual_site_names: Optional[Iterable[str]] = None,
) -> tuple:
    """
    Create a monthly mean price plot with error bars.
    Matches the academic styling of the diurnal profile plots.
    """
    # 1. Load data
    examples_sites = load_examples_sites()
    input_ts_by_site = load_input_timeseries()

    if site_ids is None:
        site_ids = [0, 5, 3]

    selected_ids = list(site_ids)
    num_sites = len(selected_ids)

    if manual_site_names is not None:
        manual_names = list(manual_site_names)
        if len(manual_names) != num_sites:
            raise ValueError("Number of manual names must match number of site IDs.")

    fig, ax = plt.subplots(figsize=figsize)

    # ACADEMIC COLOR PALETTE (Teal, Blue, Copper)
    site_colors = ["#43D1D9", "#4B86C2", "#9C7667"]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    months = np.arange(1, 13)
    
    site_handles = []

    for i, site_id in enumerate(selected_ids):
        if site_id not in input_ts_by_site:
            continue
            
        color = site_colors[i % len(site_colors)]
        site_df = input_ts_by_site[site_id].copy()
        
        # Site name logic
        site_name = manual_names[i] if manual_site_names else _format_site_name(str(examples_sites.loc[site_id, "name"]))

        # Column resolution and numeric cleaning
        price_col_res = _resolve_column(site_df, price_col)
        site_df[price_col_res] = pd.to_numeric(site_df[price_col_res], errors="coerce")

        if not isinstance(site_df.index, pd.DatetimeIndex):
            site_df.index = pd.to_datetime(site_df.index, errors="coerce", dayfirst=True)
        
        # Calculate Monthly Mean and Standard Deviation (for error bars)
        monthly_stats = site_df.groupby(site_df.index.month)[price_col_res].agg(['mean', 'std']).reindex(months)
        
        # Plot Line with Error Bars
        ax.errorbar(
            months, monthly_stats['mean'], yerr=monthly_stats['std'],
            label=site_name, color=color, fmt='-o', markersize=6, 
            linewidth=2, capsize=4, elinewidth=1, alpha=0.9
        )

        # Store mean for legend display (matching your image's legend style)
        overall_mean = site_df[price_col_res].mean()
        site_handles.append(mlines.Line2D([], [], color=color, marker='o', 
                           linewidth=2, label=f"{site_name} (Mean: {overall_mean:.2f} €/MWh)"))

    # --- ACADEMIC FORMATTING ---
    ax.set_ylabel("Day-Ahead Market Price [€/MWh]", fontweight="bold", fontsize=10)
    ax.set_xlabel("Month", fontweight="bold", fontsize=10)
    ax.set_title("Monthly Variations in Long-Term Mean Day-Ahead Market Prices", 
                 fontweight="bold", fontsize=12, pad=15)

    # Set X-Ticks to Months
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.set_xlim(0.5, 12.5)

    # Dotted Grid
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

# --- SINGLE LINE LEGEND ---
    fig.subplots_adjust(bottom=0.2) # Make room at bottom
    ax.legend(
        handles=site_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(site_handles), # Forces all sites onto one line
        frameon=False,
        fontsize=8.5,
        columnspacing=1.0 # Adjust spacing between items
    )

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, "monthly_market_prices.png"), dpi=300, bbox_inches="tight")

    return fig, ax


def plot_monthly_mean_price_timeseries(
    site_ids: Optional[Iterable[int]] = None,
    price_col: str = "price",
    figsize: Optional[tuple] = (12, 10),
    save_path: Optional[str] = None,
    manual_site_names: Optional[Iterable[str]] = None,
) -> tuple:
    """
    Historical Monthly Mean Price Plot.
    - Centered titles.
    - Colors: Teal (#43D1D9), Blue (#4B86C2), Copper (#9C7667).
    """
    examples_sites = load_examples_sites()
    input_ts_by_site = load_input_timeseries()

    if site_ids is None:
        site_ids = [0, 5, 3]

    selected_ids = list(site_ids)
    num_sites = len(selected_ids)

    if manual_site_names is not None:
        manual_names = list(manual_site_names)
        if len(manual_names) != num_sites:
            raise ValueError("Number of manual names must match number of site IDs.")

    fig, axes = plt.subplots(num_sites, 1, figsize=figsize, sharex=True)
    if num_sites == 1:
        axes = [axes]

    # THE COLORS FROM YOUR IMAGE
    site_colors = ["#43D1D9", "#4B86C2", "#9C7667"]

    for i, (site_id, ax) in enumerate(zip(selected_ids, axes)):
        if site_id not in input_ts_by_site:
            continue
            
        color = site_colors[i % len(site_colors)]
        site_df = input_ts_by_site[site_id].copy()
        
        # Manual name logic
        site_name = manual_names[i] if manual_site_names else _format_site_name(str(examples_sites.loc[site_id, "name"]))

        price_col_res = _resolve_column(site_df, price_col)
        site_df[price_col_res] = pd.to_numeric(site_df[price_col_res], errors="coerce")

        if not isinstance(site_df.index, pd.DatetimeIndex):
            site_df.index = pd.to_datetime(site_df.index, errors="coerce", dayfirst=True)
        
        # Monthly Resampling
        monthly_df = site_df[price_col_res].resample('MS').mean()
        overall_mean = monthly_df.mean()
        overall_std = monthly_df.std()

        # Plot line
        ax.plot(monthly_df.index, monthly_df, color=color, linewidth=1.5, alpha=0.9)
        
        # Add shaded area for volatility (std dev)
        ax.fill_between(monthly_df.index, monthly_df - overall_std, monthly_df + overall_std, 
                        color=color, alpha=0.15)

        # --- CENTERED TITLE ---
        ax.set_title(f"{site_name} Mean: {overall_mean:.2f} €/MWh, Std Dev: {overall_std:.2f} €/MWh", 
                     fontsize=10, fontweight="bold", loc='center', pad=10)

        # Formatting
        ax.set_ylabel("Price [€/MWh]", fontweight="bold", fontsize=9)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    axes[-1].set_xlabel("Year", fontweight="bold", fontsize=10)
    fig.suptitle("Monthly Mean Day-Ahead Market Price Variations", 
                 fontsize=12, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, "monthly_mean_price_timeseries.png")
        fig.savefig(save_file, dpi=300, bbox_inches="tight")

    return fig, axes

def plot_normalized_annual_resource_availability(
    site_ids: Optional[Iterable[int]] = None,
    wind_col: str = "WS_150",
    solar_col: str = "ghi",
    price_col: str = "price",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
    manual_site_names: Optional[Iterable[str]] = None,
) -> tuple:
    """
    Plots normalized annual mean wind, solar, and price.
    Colors and styling matched to reference images.
    """
    examples_sites = load_examples_sites()
    input_ts_by_site = load_input_timeseries()

    if site_ids is None:
        site_ids = [12]
    selected_ids = list(site_ids)
    
    if manual_site_names is not None:
        manual_names = list(manual_site_names)
    else:
        manual_names = [_format_site_name(str(examples_sites.loc[sid, "name"])) for sid in selected_ids]

    fig, ax = plt.subplots(figsize=figsize)

    # EXACT COLORS FROM YOUR REFERENCE
    # Teal: #61d9d9 (HPP/Combined)
    # Blue: #6091cf (Solar)
    # Brown: #a68b7c (Wind)
    # Price: #2e7d32 (Academic Green)
    
    color_wind = "#43D1D9"
    color_solar = "#4B86C2"
    color_price = "#9C7667" 

    for i, site_id in enumerate(selected_ids):
        if site_id not in input_ts_by_site: continue
            
        site_df = input_ts_by_site[site_id].copy()
        site_name = manual_names[i]

        if not isinstance(site_df.index, pd.DatetimeIndex):
            site_df.index = pd.to_datetime(site_df.index, errors="coerce", dayfirst=True)
        
        # Column Resolution
        w_res = _resolve_column(site_df, wind_col)
        s_res = _resolve_column(site_df, solar_col)
        p_res = _resolve_column(site_df, price_col)

        # Annual Resampling
        cols = [w_res, s_res, p_res]
        annual = site_df[cols].apply(pd.to_numeric, errors='coerce').resample('YE').mean()
        
        # Normalization (Value / Mean)
        norm = annual / annual.mean()

        # PLOTTING WITH MATCHED STYLES
        # Wind (Brown, solid, circle)
        ax.plot(norm.index.year, norm[w_res], color=color_wind, 
                linestyle='-', marker='o', markersize=6, linewidth=1.5, label='Wind')
        
        # Solar (Blue, solid, square - adjusted from reference)
        ax.plot(norm.index.year, norm[s_res], color=color_solar, 
                linestyle='-', marker='o', markersize=5, linewidth=1.5, label='Solar')
        
        # Price (Green, dotted, triangle)
        ax.plot(norm.index.year, norm[p_res], color=color_price, 
                linestyle='-', marker='o', markersize=6, linewidth=1.2, alpha=0.8, label='Price')

    # --- ACADEMIC FORMATTING ---
    ax.axhline(1.0, color='#333333', lw=1.0, linestyle='-', alpha=0.5)
    
    ax.set_ylabel("Normalized Annual Index", fontweight="bold", fontsize=10)
    ax.set_xlabel("Scenario Year", fontweight="bold", fontsize=10)
    ax.set_title(f"Annual Resource & Price Variability: {manual_names[0]}", 
                 fontweight="bold", fontsize=14, pad=20, color="#2c3e50")
    
    # Show every year on the x-axis
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    # 2. Format labels to show only the last two digits (e.g., 1982 -> 82)
    def format_year(x, pos):
        return f"{int(x) % 100:02d}"
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_year))
    ax.set_xlim(1981.5, 2015.5)

    
    # Optional: Rotate labels if they overlap
    plt.xticks(rotation=45, fontsize=8) 

    ax.tick_params(axis='x', labelsize=8.5)
    ax.set_xlabel("Scenario Year", fontweight="bold", fontsize=10)
      

    # Matching the grid style from reference
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, color='#d3d3d3')
    ax.set_facecolor('#fafafa')

    # --- SINGLE LINE LEGEND ---
    # Proxies for the legend to ensure clean labels
    wind_p = mlines.Line2D([], [], color=color_wind, marker='o', label='Wind Speed')
    solar_p = mlines.Line2D([], [], color=color_solar, marker='o', label='Solar Irradiance')
    price_p = mlines.Line2D([], [], color=color_price, ls='-', marker='o', label='Electricity Price')
    
    ax.legend(handles=[wind_p, solar_p, price_p], loc='lower center', 
              bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, "annual_indexed_variability_SudAtlantique.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {filename}")

    return fig, ax


def plot_normalized_annual_wind_solar_availability(
    site_ids: Optional[Iterable[int]] = None,
    wind_col: str = "WS_150",
    solar_col: str = "ghi",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
    manual_site_names: Optional[Iterable[str]] = None,
) -> tuple:
    """
    Plots normalized annual mean wind and solar resource availability only.
    Colors and styling matched to reference images.
    """
    examples_sites = load_examples_sites()
    input_ts_by_site = load_input_timeseries()

    if site_ids is None:
        site_ids = [12]
    selected_ids = list(site_ids)
    
    if manual_site_names is not None:
        manual_names = list(manual_site_names)
    else:
        manual_names = [_format_site_name(str(examples_sites.loc[sid, "name"])) for sid in selected_ids]

    fig, ax = plt.subplots(figsize=figsize)

    # Academic Palette
    color_wind = "#43D1D9"   # Teal
    color_solar = "#4B86C2"  # Blue

    for i, site_id in enumerate(selected_ids):
        if site_id not in input_ts_by_site: continue
            
        site_df = input_ts_by_site[site_id].copy()
        
        if not isinstance(site_df.index, pd.DatetimeIndex):
            site_df.index = pd.to_datetime(site_df.index, errors="coerce", dayfirst=True)
        
        # Column Resolution
        w_res = _resolve_column(site_df, wind_col)
        s_res = _resolve_column(site_df, solar_col)

        # Annual Resampling (Wind and Solar only)
        cols = [w_res, s_res]
        annual = site_df[cols].apply(pd.to_numeric, errors='coerce').resample('YE').mean()
        
        # Normalization (Value / Mean)
        norm = annual / annual.mean()

        # PLOTTING
        # Wind (Teal, solid, circle)
        ax.plot(norm.index.year, norm[w_res], color=color_wind, 
                linestyle='-', marker='o', markersize=6, linewidth=1.5, label='Wind')
        
        # Solar (Blue, solid, circle)
        ax.plot(norm.index.year, norm[s_res], color=color_solar, 
                linestyle='-', marker='o', markersize=6, linewidth=1.5, label='Solar')

    # --- ACADEMIC FORMATTING ---
    ax.axhline(1.0, color='#333333', lw=1.0, linestyle='-', alpha=0.5)
    
    ax.set_ylabel("Normalized Annual Index", fontweight="bold", fontsize=10)
    ax.set_title(f"Annual Resource Variability: {manual_names[0]}", 
                 fontweight="bold", fontsize=14, pad=20, color="#2c3e50")
    
    # X-Axis Formatting: Every year, two digits
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    
    def format_year(x, pos):
        return f"{int(x) % 100:02d}"
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_year))
    ax.set_xlim(1981.5, 2015.5)
    
    plt.xticks(rotation=45, fontsize=8.5) 
    ax.set_xlabel("Scenario Year", fontweight="bold", fontsize=10)

    # Grid and background
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, color='#d3d3d3')
    ax.set_facecolor('#fafafa')

    # --- SINGLE LINE LEGEND ---
    wind_p = mlines.Line2D([], [], color=color_wind, marker='o', label='Wind Speed')
    solar_p = mlines.Line2D([], [], color=color_solar, marker='o', label='Solar Irradiance')
    
    ax.legend(handles=[wind_p, solar_p], loc='lower center', 
              bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False, fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, "annual_indexed_wind_solar_variability_SudAtlantique.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {filename}")

    return fig, ax

def plot_combined_annual_variability_final(
    site_ids: Optional[Iterable[int]] = None,
    wind_col: str = "WS_150",
    solar_col: str = "ghi",
    price_col: str = "price",
    figsize: tuple = (12, 10),
    save_path: Optional[str] = None,
    manual_site_names: Optional[Iterable[str]] = None,
) -> tuple:
    """
    Two-row subplot:
    Top: Wind & Solar
    Bottom: Wind, Solar & Price
    Single legend at the very bottom.
    """
    examples_sites = load_examples_sites()
    input_ts_by_site = load_input_timeseries()

    if site_ids is None:
        site_ids = [12]
    selected_ids = list(site_ids)
    
    if manual_site_names is not None:
        site_name = list(manual_site_names)[0]
    else:
        site_name = _format_site_name(str(examples_sites.loc[selected_ids[0], "name"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    color_wind = "#43D1D9"   # Teal
    color_solar = "#4B86C2"  # Blue
    color_price = "#9C7667"  # Copper/Brown

    def format_year(x, pos):
        return f"{int(x) % 100:02d}"

    for site_id in selected_ids:
        if site_id not in input_ts_by_site: continue
            
        site_df = input_ts_by_site[site_id].copy()
        if not isinstance(site_df.index, pd.DatetimeIndex):
            site_df.index = pd.to_datetime(site_df.index, errors="coerce", dayfirst=True)
        
        w_res = _resolve_column(site_df, wind_col)
        s_res = _resolve_column(site_df, solar_col)
        p_res = _resolve_column(site_df, price_col)

        annual = site_df[[w_res, s_res, p_res]].apply(pd.to_numeric, errors='coerce').resample('YE').mean()
        norm = annual / annual.mean()

        # TOP PLOT (Resources Only) - Labels suppressed for internal legend
        ax1.plot(norm.index.year, norm[w_res], color=color_wind, marker='o')
        ax1.plot(norm.index.year, norm[s_res], color=color_solar, marker='o')
        
        # BOTTOM PLOT (Resources & Price)
        ax2.plot(norm.index.year, norm[w_res], color=color_wind, marker='o')
        ax2.plot(norm.index.year, norm[s_res], color=color_solar, marker='o')
        ax2.plot(norm.index.year, norm[p_res], color=color_price, marker='o')

    # Formatting
    for ax in [ax1, ax2]:
        ax.axhline(1.0, color='#333333', lw=1.0, alpha=0.5)
        ax.set_ylabel("Normalized Index", fontweight="bold", fontsize=10)
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_year))
        ax.set_xlim(1981.5, 2015.5)

    ax1.tick_params(labelbottom=False) 
    ax2.set_xlabel("Scenario Year", fontweight="bold", fontsize=10)
    plt.setp(ax2.get_xticklabels(), rotation=45, fontsize=8.5)

    # SINGLE MAIN TITLE - Adjusted 'y' to be closer to the plot
    fig.suptitle(f"Annual Resource and Price Variability: {site_name}", 
                 fontweight="bold", fontsize=14, y=0.97)

    # SINGLE LEGEND AT THE VERY BOTTOM - Adjusted 'bbox_to_anchor'
    w_p = mlines.Line2D([], [], color=color_wind, marker='o', label='Wind Speed')
    s_p = mlines.Line2D([], [], color=color_solar, marker='o', label='Solar Irradiance')
    p_p = mlines.Line2D([], [], color=color_price, marker='o', label='Electricity Price')
    
    fig.legend(handles=[w_p, s_p, p_p], loc='lower center', 
               bbox_to_anchor=(0.5, 0.04), # Moved up slightly
               ncol=3, frameon=False, fontsize=10)

    # ADJUST LAYOUT - Reduced padding in 'rect' and reduced 'hspace'
    # rect=[left, bottom, right, top]
    plt.tight_layout(rect=[0, 0.07, 1, 0.98]) 
    plt.subplots_adjust(hspace=0.1) 

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, "annual_indexed_wind_solar_variability_stacked_Thetys.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {filename}")

    return fig, (ax1, ax2)

if __name__ == "__main__":
    # Storage path for plots
    PLOT_STORAGE_PATH = r"C:\Users\malth\HPP\hydesign\HPP\DataStoreage"

    # Print summary table
    #summary_table = load_and_build_summary_table(
    #    wind_col="WS_150",
    #    solar_col="ghi",
    #    price_col="price",
    #)
    #print(summary_table.to_string())
    #print("\n" + "="*80 + "\n")

    # Plot wind speed distribution for default sites (0, 5, 3) and save
    #fig, axes = plot_wind_speed_distribution(
    #    manual_site_names=["Nordsøen Midt (DK)", "Sud Atlantique (FRw)", "Sicily South (IT)"],
    #    site_ids=[7, 12, 10],
    #    save_path=PLOT_STORAGE_PATH,
    #)
    #plt.close(fig)

    # Plot solar irradiance distribution for default sites and save
    #fig, axes = plot_solar_irradiance_distribution(
    #    manual_site_names=["Nordsøen Midt (DK)", "Sud Atlantique (FRw)", "Sicily South (IT)"],
    #    site_ids=[7, 12, 10],
    #    save_path=PLOT_STORAGE_PATH,
    #)
    #plt.close(fig)

    # Plot price distribution for default sites and save
    #fig, axes = plot_price_distribution(
    #    site_ids=[0, 5, 3],
    #    save_path=PLOT_STORAGE_PATH,
    #)
    #plt.close(fig)

    # Plot daily mean wind speed and solar irradiance for default sites and save
    #fig, axes = plot_daily_mean_wind_and_solar(
    #    site_ids=[7, 12, 10],
    #    manual_site_names=["Nordsøen Midt (DK)", "Sud Atlantique (FRw)", "Sicily South (IT)"],
    #    target_year=2012,
    #    save_path=PLOT_STORAGE_PATH,
    #)
    #plt.close(fig)

    # Plot daily mean wind speed and price for default sites and save
    #fig, axes = plot_daily_mean_wind_and_price(
    #    site_ids=[0, 5, 3],
    #    target_year=2012,
    #    save_path=PLOT_STORAGE_PATH,
    #)
    #plt.close(fig)

    # Plot daily mean solar irradiance and price for default sites and save
    #fig, axes = plot_daily_mean_solar_and_price(
    #    site_ids=[0, 5, 3],
    #    target_year=2012,
    #    save_path=PLOT_STORAGE_PATH,
    #)
    #plt.close(fig)

    # Plot mean hourly wind-price and solar-price for default sites and save
    #fig, axes = plot_mean_hourly_resource_and_price(
    #    manual_site_names=["Nordsøen Midt (DK)", "Sud Atlantique (FRw)", "Sicily South (IT)"],
    #    site_ids=[7, 12, 10],
    #     save_path=PLOT_STORAGE_PATH,
    #)
    #plt.close(fig)

    # Plot daily mean wind speed and solar irradiance overlay for Thetys and Sud Atlantique and save
    #fig, (ax1, ax2) = plot_daily_mean_wind_and_solar_overlay(
    #    target_year=2012,
    #    save_path=PLOT_STORAGE_PATH,
    #)
    #plt.close(fig)

    # Plot monthly mean resource and price for default sites and save
    #fig, axes = plot_mean_monthly_resource_and_price(
    #    site_ids=[1, 4, 5],
    #    save_path=PLOT_STORAGE_PATH,
    #)
    #plt.close(fig)

    # Plot Monthly Variations in Long-Term Mean Day-Ahead Market Prices
    #fig, axes = plot_monthly_market_prices(
    #    site_ids=[7, 12, 10],
    #    manual_site_names=["Nordsøen Midt (DK)", "Sud Atlantique (FRw)", "Sicily South (IT)"],
    #    save_path=PLOT_STORAGE_PATH,
    #)

    # Plot Historical Day-Ahead Market Price Time-Series and Volatility
    #fig, axes = plot_monthly_mean_price_timeseries(
    #    site_ids=[7, 12, 10],
    #    manual_site_names=["Nordsøen Midt (DK)", "Sud Atlantique (FRw)", "Sicily South (IT)"],
    #    save_path=PLOT_STORAGE_PATH,
    #)
    #plt.close(fig)

    # Plot indexed wind, solar and price. 
    fig, axes = plot_combined_annual_variability_final(
        site_ids=[8],
        manual_site_names=["Thetys (NL)"],
        save_path=PLOT_STORAGE_PATH
    )   
    plt.close(fig)

    # Plot 3x2 wind vs price for all sites
    #examples_sites = load_examples_sites()
    #input_ts_by_site = load_input_timeseries()
    #plot_all_sites_scatterhist_resource_vs_price(
    #    input_ts_by_site=input_ts_by_site,
    #    examples_sites=examples_sites,
    #    resource_col="WS_150",
    #    price_col="Price",
    #    color="tab:blue",
    #    alpha=0.5,
    #    resource_label="Wind Speed (m/s)",
    #    price_label="Price (EUR/MW)",
    #    figsize=(14, 10),
    #    plot_type="wind",
    #    save_path=PLOT_STORAGE_PATH,
    #)

    # Plot 3x2 solar vs price for all sites
    #plot_all_sites_scatterhist_resource_vs_price(
    #    input_ts_by_site=input_ts_by_site,
    #    examples_sites=examples_sites,
    #    resource_col="GHI",
    #    price_col="Price",
    #    color="darkorange",
    #    alpha=0.5,
    #    resource_label="Solar Irradiance (W/m^2)",
    #    price_label="Price (EUR/MW)",
    #    figsize=(14, 10),
    #    plot_type="solar",
    #    save_path=PLOT_STORAGE_PATH,
    #)

    # Uncomment to plot different site combinations:
    # fig, axes = plot_wind_speed_distribution(site_ids=[0, 1, 2], save_path=PLOT_STORAGE_PATH)  # First 3 sites
    # fig, axes = plot_wind_speed_distribution(site_ids=range(6), save_path=PLOT_STORAGE_PATH)    # All 6 sites
    # fig, axes = plot_wind_speed_distribution(site_ids=[0, 5], save_path=PLOT_STORAGE_PATH)      # Just 2 sites
    # plt.show()




