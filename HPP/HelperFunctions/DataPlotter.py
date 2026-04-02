# File for plotting and tabulating input data.

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
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
    title: str = "Wind speed distrubution",
    bins: int = 30,
    save_path: Optional[str] = None,
) -> tuple:
    """
    Create a figure with histograms of wind speed distributions for selected sites.

    Parameters
    ----------
    site_ids : Optional[Iterable[int]]
        List of site indices to include. If None, includes first 3 sites by default.
        Available sites: 0-5 (NordsoenMidt, Thetys, Vestavind, SicilySouth, 
        Golfe_du_Lion, Sud_Atlantique)
    wind_col : str
        Name of the wind speed column in the input data. Default: "WS_150"
    figsize : Optional[tuple]
        Figure size as (width, height). If None, auto-sized based on number of sites.
    title : str
        Title for the figure.
    bins : int
        Number of bins for the histograms.
    save_path : Optional[str]
        Directory path to save the figure. If provided, figure is saved as PNG.
        Directory is created if it doesn't exist.

    Returns
    -------
    tuple
        (fig, axes) - matplotlib Figure and Axes objects for further customization.

    Example
    -------
    # Plot default 3 sites
    fig, axes = plot_wind_speed_distribution(site_ids=[0, 5, 3])
    plt.show()

    # Plot specific sites
    fig, axes = plot_wind_speed_distribution(site_ids=[0, 1, 5])
    plt.show()

    # Plot and save to folder
    fig, axes = plot_wind_speed_distribution(
        site_ids=[0, 5, 3],
        save_path="C:/Users/malth/HPP/hydesign/HPP/DataStoreage"
    )
    # Plot all 6 sites
    fig, axes = plot_wind_speed_distribution(site_ids=range(6))
    plt.show()
    """
    # Load data
    examples_sites = load_examples_sites()
    input_ts_by_site = load_input_timeseries()

    # Default to first 3 sites if none specified
    if site_ids is None:
        site_ids = [0, 5, 3]  # NordsoenMidt, Sud_Atlantique, SicilySouth

    selected_ids = list(site_ids)
    num_sites = len(selected_ids)

    # Set figure size
    if figsize is None:
        figsize = (4 * num_sites, 4)

    fig, axes = plt.subplots(1, num_sites, figsize=figsize)

    # Handle single site (axes won't be an array)
    if num_sites == 1:
        axes = [axes]

    for idx, (site_id, ax) in enumerate(zip(selected_ids, axes)):
        if site_id not in input_ts_by_site:
            raise ValueError(f"Site ID {site_id} not found in loaded data")

        site_df = input_ts_by_site[site_id]
        site_name = _format_site_name(str(examples_sites.loc[site_id, "name"]))

        # Resolve wind column case-insensitively
        wind_col_res = _resolve_column(site_df, wind_col)
        wind_data = pd.to_numeric(site_df[wind_col_res], errors="coerce").dropna()

        # Create histogram
        ax.hist(wind_data, bins=bins, color="steelblue", edgecolor="black", alpha=0.7)

        # Add mean line
        mean_val = wind_data.mean()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=2)

        # Formatting
        ax.set_xlabel("Wind Speed (m/s)", fontsize=10)
        ax.set_ylabel("Frequency (hours)", fontsize=10)
        ax.set_title(f"{site_name} | Mean: {mean_val:.2f} m/s", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    # Save plot if save_path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename_base = os.path.join(save_path, "wind_speed_distribution")
        fig.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight")
        print(f"Plot saved to:\n  {filename_base}.png")

    return fig, axes


def plot_solar_irradiance_distribution(
    site_ids: Optional[Iterable[int]] = None,
    solar_col: str = "ghi",
    figsize: Optional[tuple] = None,
    title: str = "Solar irradiance distrubution",
    bins: int = 30,
    save_path: Optional[str] = None,
) -> tuple:
    """
    Create a figure with histograms of solar irradiance distributions.
    """
    examples_sites = load_examples_sites()
    input_ts_by_site = load_input_timeseries()

    if site_ids is None:
        site_ids = [0, 5, 3]

    selected_ids = list(site_ids)
    num_sites = len(selected_ids)

    if figsize is None:
        figsize = (4 * num_sites, 4)

    fig, axes = plt.subplots(1, num_sites, figsize=figsize)

    if num_sites == 1:
        axes = [axes]

    for site_id, ax in zip(selected_ids, axes):
        if site_id not in input_ts_by_site:
            raise ValueError(f"Site ID {site_id} not found in loaded data")

        site_df = input_ts_by_site[site_id]
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

        mean_val = solar_data.mean()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=2)

        ax.set_xlabel("Solar Irradiance (W/m^2)", fontsize=10)
        ax.set_ylabel("Frequency (hours)", fontsize=10)
        ax.set_title(f"{site_name} | Mean: {mean_val:.2f} W/m^2", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename_base = os.path.join(save_path, "solar_irradiance_distribution")
        fig.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight")
        print(f"Plot saved to:\n  {filename_base}.png")

    return fig, axes


def plot_price_distribution(
    site_ids: Optional[Iterable[int]] = None,
    price_col: str = "price",
    figsize: Optional[tuple] = None,
    title: str = "Price distrubution",
    bins: int = 30,
    save_path: Optional[str] = None,
) -> tuple:
    """
    Create a figure with histograms of electricity price distributions.
    """
    examples_sites = load_examples_sites()
    input_ts_by_site = load_input_timeseries()

    if site_ids is None:
        site_ids = [0, 5, 3]

    selected_ids = list(site_ids)
    num_sites = len(selected_ids)

    if figsize is None:
        figsize = (4 * num_sites, 4)

    fig, axes = plt.subplots(1, num_sites, figsize=figsize)

    if num_sites == 1:
        axes = [axes]

    for site_id, ax in zip(selected_ids, axes):
        if site_id not in input_ts_by_site:
            raise ValueError(f"Site ID {site_id} not found in loaded data")

        site_df = input_ts_by_site[site_id]
        site_name = _format_site_name(str(examples_sites.loc[site_id, "name"]))

        price_col_res = _resolve_column(site_df, price_col)
        price_data = pd.to_numeric(site_df[price_col_res], errors="coerce").dropna()

        ax.hist(
            price_data,
            bins=bins,
            color="seagreen",
            edgecolor="black",
            alpha=0.7,
        )

        mean_val = price_data.mean()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=2)

        ax.set_xlabel("Price (EUR/MW)", fontsize=10)
        ax.set_ylabel("Frequency (hours)", fontsize=10)
        ax.set_title(f"{site_name} | Mean: {mean_val:.2f} EUR/MW", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename_base = os.path.join(save_path, "price_distribution")
        fig.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight")
        print(f"Plot saved to:\n  {filename_base}.png")

    return fig, axes


def plot_daily_mean_wind_and_solar(
    site_ids: Optional[Iterable[int]] = None,
    wind_col: str = "WS_150",
    solar_col: str = "ghi",
    figsize: Optional[tuple] = None,
    target_year: int = 2012,
    title_prefix: str = "Daily Mean Wind Speed and Solar Irradiance",
    save_path: Optional[str] = None,
) -> tuple:
    """
    Create stacked time-series plots of daily mean wind speed and solar irradiance.
    """
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

    wind_color = "#5b9bd5"
    solar_color = "orange"

    for i, (site_id, ax1) in enumerate(zip(selected_ids, axes)):
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
            label="Solar Irradiance",
        )
        ax2.set_ylabel(
            "Daily Mean GHI (W/m^2)",
            color=solar_color,
            fontsize=8,
        )
        ax2.tick_params(axis="y", labelcolor=solar_color, labelsize=7)

        line1 = ax1.get_lines()[0]
        line2 = ax2.get_lines()[0]
        ax1.legend([line1, line2], ["Wind Speed", "Solar Irradiance"], fontsize=6)

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


def plot_mean_hourly_resource_and_price(
    site_ids: Optional[Iterable[int]] = None,
    wind_col: str = "WS_150",
    solar_col: str = "ghi",
    price_col: str = "price",
    figsize: Optional[tuple] = None,
    target_year: Optional[int] = None,
    save_path: Optional[str] = None,
) -> tuple:
    """Create side-by-side hourly mean wind-price and solar-price subplots."""
    examples_sites = load_examples_sites()
    input_ts_by_site = load_input_timeseries()

    if site_ids is None:
        site_ids = [0, 5, 3]

    selected_ids = list(site_ids)
    if figsize is None:
        figsize = (13, 4.8)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    ax_wind = axes[0]
    ax_solar = axes[1]
    ax_wind_price = ax_wind.twinx()
    ax_solar_price = ax_solar.twinx()

    site_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    hours = list(range(24))

    wind_handles = []
    wind_labels = []
    solar_handles = []
    solar_labels = []

    for i, site_id in enumerate(selected_ids):
        if site_id not in input_ts_by_site:
            raise ValueError(f"Site ID {site_id} not found in loaded data")

        color = site_colors[i % len(site_colors)]
        site_df = input_ts_by_site[site_id].copy()
        site_name = _format_site_name(str(examples_sites.loc[site_id, "name"]))

        wind_col_res = _resolve_column(site_df, wind_col)
        solar_col_res = _resolve_column(site_df, solar_col)
        price_col_res = _resolve_column(site_df, price_col)

        if not isinstance(site_df.index, pd.DatetimeIndex):
            site_df.index = pd.to_datetime(
                site_df.index,
                errors="coerce",
                dayfirst=True,
            )
        site_df = site_df[site_df.index.notna()]

        if target_year is None:
            period_df = site_df.copy()
        else:
            period_df = site_df[site_df.index.year == target_year].copy()
            if period_df.empty:
                raise ValueError(
                    f"No data available for year {target_year} at site '{site_name}'"
                )

        for col in [wind_col_res, solar_col_res, price_col_res]:
            period_df[col] = pd.to_numeric(period_df[col], errors="coerce")

        hourly = period_df[[wind_col_res, solar_col_res, price_col_res]].groupby(
            period_df.index.hour
        ).mean()
        hourly = hourly.reindex(hours)

        wind_line = ax_wind.plot(
            hours,
            hourly[wind_col_res],
            color=color,
            linewidth=1.8,
            label=f"{site_name} Wind",
        )[0]
        wind_price_line = ax_wind_price.plot(
            hours,
            hourly[price_col_res],
            color=color,
            linestyle="--",
            linewidth=1.4,
            alpha=0.8,
            label=f"{site_name} Price",
        )[0]

        solar_line = ax_solar.plot(
            hours,
            hourly[solar_col_res],
            color=color,
            linewidth=1.8,
            label=f"{site_name} Solar",
        )[0]
        solar_price_line = ax_solar_price.plot(
            hours,
            hourly[price_col_res],
            color=color,
            linestyle="--",
            linewidth=1.4,
            alpha=0.8,
            label=f"{site_name} Price",
        )[0]

        wind_handles.extend([wind_line, wind_price_line])
        wind_labels.extend([f"{site_name} Wind", f"{site_name} Price"])
        solar_handles.extend([solar_line, solar_price_line])
        solar_labels.extend([f"{site_name} Solar", f"{site_name} Price"])

    title_suffix = f" ({target_year})" if target_year is not None else ""
    ax_wind.set_title(
        f"Mean Hourly Wind Speed and Mean Hourly Price {title_suffix}",
        fontsize=10,
        fontweight="bold",
    )
    ax_wind.set_ylabel("Wind Speed (m/s)", fontsize=9)
    ax_wind_price.set_ylabel("Price (EUR/MW)", fontsize=9, color="seagreen")
    ax_wind_price.tick_params(axis="y", labelcolor="seagreen")
    ax_wind.grid(True, linestyle="-", linewidth=0.4, alpha=0.3)
    ax_solar.set_title(
        f"Mean Hourly Solar Irradiance and Mean Hourly Price {title_suffix}",
        fontsize=10,
        fontweight="bold",
    )
    ax_solar.set_ylabel("Solar Irradiance (W/m^2)", fontsize=9)
    ax_solar_price.set_ylabel("Price (EUR/MW)", fontsize=9, color="seagreen")
    ax_solar_price.tick_params(axis="y", labelcolor="seagreen")
    ax_solar.grid(True, linestyle="-", linewidth=0.4, alpha=0.3)
    combined_handles = wind_handles + solar_handles
    combined_labels = wind_labels + solar_labels
    unique_handles = []
    unique_labels = []
    seen_labels = set()
    for handle, label in zip(combined_handles, combined_labels):
        if label in seen_labels:
            continue
        seen_labels.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)

    fig.legend(
        unique_handles,
        unique_labels,
        fontsize=7,
        ncol=min(4, len(unique_labels)),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
    )

    ax_wind.set_xlabel("Hour of Day", fontsize=9, fontweight="bold")
    ax_solar.set_xlabel("Hour of Day", fontsize=9, fontweight="bold")
    ax_solar.set_xticks(hours)
    ax_solar.set_xlim(0, 23)
    ax_wind.set_xticks(hours)
    ax_wind.set_xlim(0, 23)
    fig.subplots_adjust(bottom=0.30, wspace=0.28)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename_base = os.path.join(
            save_path,
            "mean_hourly_wind_price_and_solar_price_by_site",
        )
        fig.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight")
        print(f"Plot saved to:\n  {filename_base}.png")

    return fig, axes

def plot_mean_monthly_resource_and_price(
    site_ids: Optional[Iterable[int]] = None,
    wind_col: str = "WS_150",
    solar_col: str = "ghi",
    price_col: str = "price",
    figsize: Optional[tuple] = None,
    target_year: Optional[int] = None,
    save_path: Optional[str] = None,
) -> tuple:
    """Create side-by-side monthly mean wind-price and solar-price subplots."""
    examples_sites = load_examples_sites()
    input_ts_by_site = load_input_timeseries()

    if site_ids is None:
        site_ids = [0, 5, 3]

    selected_ids = list(site_ids)
    if figsize is None:
        figsize = (13, 4.8)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    ax_wind = axes[0]
    ax_solar = axes[1]
    ax_wind_price = ax_wind.twinx()
    ax_solar_price = ax_solar.twinx()

    site_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    months = list(range(1, 13))

    wind_handles = []
    wind_labels = []
    solar_handles = []
    solar_labels = []

    for i, site_id in enumerate(selected_ids):
        if site_id not in input_ts_by_site:
            raise ValueError(f"Site ID {site_id} not found in loaded data")

        color = site_colors[i % len(site_colors)]
        site_df = input_ts_by_site[site_id].copy()
        site_name = _format_site_name(str(examples_sites.loc[site_id, "name"]))

        wind_col_res = _resolve_column(site_df, wind_col)
        solar_col_res = _resolve_column(site_df, solar_col)
        price_col_res = _resolve_column(site_df, price_col)

        if not isinstance(site_df.index, pd.DatetimeIndex):
            site_df.index = pd.to_datetime(
                site_df.index,
                errors="coerce",
                dayfirst=True,
            )
        site_df = site_df[site_df.index.notna()]

        if target_year is None:
            period_df = site_df.copy()
        else:
            period_df = site_df[site_df.index.year == target_year].copy()
            if period_df.empty:
                raise ValueError(
                    f"No data available for year {target_year} at site '{site_name}'"
                )

        for col in [wind_col_res, solar_col_res, price_col_res]:
            period_df[col] = pd.to_numeric(period_df[col], errors="coerce")

        monthly = period_df[[wind_col_res, solar_col_res, price_col_res]].groupby(
            period_df.index.month
        ).mean()
        monthly = monthly.reindex(months)

        wind_line = ax_wind.plot(
            months,
            monthly[wind_col_res],
            color=color,
            linewidth=1.8,
            label=f"{site_name} Wind",
        )[0]
        wind_price_line = ax_wind_price.plot(
            months,
            monthly[price_col_res],
            color=color,
            linestyle="--",
            linewidth=1.4,
            alpha=0.8,
            label=f"{site_name} Price",
        )[0]

        solar_line = ax_solar.plot(
            months,
            monthly[solar_col_res],
            color=color,
            linewidth=1.8,
            label=f"{site_name} Solar",
        )[0]
        solar_price_line = ax_solar_price.plot(
            months,
            monthly[price_col_res],
            color=color,
            linestyle="--",
            linewidth=1.4,
            alpha=0.8,
            label=f"{site_name} Price",
        )[0]

        wind_handles.extend([wind_line, wind_price_line])
        wind_labels.extend([f"{site_name} Wind", f"{site_name} Price"])
        solar_handles.extend([solar_line, solar_price_line])
        solar_labels.extend([f"{site_name} Solar", f"{site_name} Price"])

    title_suffix = f" ({target_year})" if target_year is not None else ""
    ax_wind.set_title(
        f"Mean Monthly Wind Speed and Mean Monthly Price{title_suffix}",
        fontsize=10,
        fontweight="bold",
    )
    ax_wind.set_ylabel("Wind Speed (m/s)", fontsize=9)
    ax_wind_price.set_ylabel("Price (EUR/MW)", fontsize=9, color="seagreen")
    ax_wind_price.tick_params(axis="y", labelcolor="seagreen")
    ax_wind.grid(True, linestyle="-", linewidth=0.4, alpha=0.3)
    ax_solar.set_title(
        f"Mean Monthly Solar Irradiance and Mean Monthly Price{title_suffix}",
        fontsize=10,
        fontweight="bold",
    )
    ax_solar.set_ylabel("Solar Irradiance (W/m^2)", fontsize=9)
    ax_solar_price.set_ylabel("Price (EUR/MW)", fontsize=9, color="seagreen")
    ax_solar_price.tick_params(axis="y", labelcolor="seagreen")
    ax_solar.grid(True, linestyle="-", linewidth=0.4, alpha=0.3)
    combined_handles = wind_handles + solar_handles
    combined_labels = wind_labels + solar_labels
    unique_handles = []
    unique_labels = []
    seen_labels = set()
    for handle, label in zip(combined_handles, combined_labels):
        if label in seen_labels:
            continue
        seen_labels.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)

    fig.legend(
        unique_handles,
        unique_labels,
        fontsize=7,
        ncol=min(4, len(unique_labels)),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
    )

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax_wind.set_xlabel("Month", fontsize=9, fontweight="bold")
    ax_solar.set_xlabel("Month", fontsize=9, fontweight="bold")
    ax_wind.set_xticks(months)
    ax_wind.set_xticklabels(month_labels)
    ax_wind.set_xlim(1, 12)
    ax_solar.set_xticks(months)
    ax_solar.set_xticklabels(month_labels)
    ax_solar.set_xlim(1, 12)
    fig.subplots_adjust(bottom=0.30, wspace=0.28)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename_base = os.path.join(
            save_path,
            "mean_monthly_wind_price_and_solar_price_by_site",
        )
        fig.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight")
        print(f"Plot saved to:\n  {filename_base}.png")

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


     


if __name__ == "__main__":
    # Storage path for plots
    PLOT_STORAGE_PATH = r"C:\Users\malth\HPP\hydesign\HPP\DataStoreage"

    # Print summary table
    summary_table = load_and_build_summary_table(
        wind_col="WS_150",
        solar_col="ghi",
        price_col="price",
    )
    # print(summary_table.to_string())
    # print("\n" + "="*80 + "\n")

    # Plot wind speed distribution for default sites (0, 5, 3) and save
    #fig, axes = plot_wind_speed_distribution(
    #    site_ids=[0, 5, 3],
    #    save_path=PLOT_STORAGE_PATH
    #)
    #plt.close(fig)

    # Plot solar irradiance distribution for default sites and save
    #fig, axes = plot_solar_irradiance_distribution(
    #    site_ids=[0, 5, 3],
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
    #    site_ids=[0, 5, 3],
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
    #    site_ids=[0, 5, 3],
    #    save_path=PLOT_STORAGE_PATH,
    #)
    #plt.close(fig)

    # Plot daily mean wind speed and solar irradiance overlay for Thetys and Sud Atlantique and save
    #fig, (ax1, ax2) = plot_daily_mean_wind_and_solar_overlay(
    #    target_year=2012,
    #    save_path=PLOT_STORAGE_PATH,
    #)
    #plt.close(fig)

    # Plot monthly mean resource and price for default sites and save
    fig, axes = plot_mean_monthly_resource_and_price(
        site_ids=[1, 4, 5],
        save_path=PLOT_STORAGE_PATH,
    )
    plt.close(fig)

    # Plot 3x2 wind vs price for all sites
    examples_sites = load_examples_sites()
    input_ts_by_site = load_input_timeseries()
    plot_all_sites_scatterhist_resource_vs_price(
        input_ts_by_site=input_ts_by_site,
        examples_sites=examples_sites,
        resource_col="WS_150",
        price_col="Price",
        color="tab:blue",
        alpha=0.5,
        resource_label="Wind Speed (m/s)",
        price_label="Price (EUR/MW)",
        figsize=(14, 10),
        plot_type="wind",
        save_path=PLOT_STORAGE_PATH,
    )

    # Plot 3x2 solar vs price for all sites
    plot_all_sites_scatterhist_resource_vs_price(
        input_ts_by_site=input_ts_by_site,
        examples_sites=examples_sites,
        resource_col="GHI",
        price_col="Price",
        color="darkorange",
        alpha=0.5,
        resource_label="Solar Irradiance (W/m^2)",
        price_label="Price (EUR/MW)",
        figsize=(14, 10),
        plot_type="solar",
        save_path=PLOT_STORAGE_PATH,
    )

    # Uncomment to plot different site combinations:
    # fig, axes = plot_wind_speed_distribution(site_ids=[0, 1, 2], save_path=PLOT_STORAGE_PATH)  # First 3 sites
    # fig, axes = plot_wind_speed_distribution(site_ids=range(6), save_path=PLOT_STORAGE_PATH)    # All 6 sites
    # fig, axes = plot_wind_speed_distribution(site_ids=[0, 5], save_path=PLOT_STORAGE_PATH)      # Just 2 sites
    # plt.show()




