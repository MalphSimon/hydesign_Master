"""
Sensitivity analysis for predefined HPP site configurations.

This module performs a one-factor-at-a-time (OFAT) sensitivity study around the
baseline simulation parameters used by each configured site.

Key features:
- Restricts runs to site names defined in HPP/SiteConfig/*.csv.
- Reuses the existing yearly evaluation flow in HPP/Evaluation.py.
- Varies key economic assumptions requested for decision support:
  - Wind CAPEX
  - PV CAPEX
  - Battery CAPEX
  - Lifetime
  - WACC
  - Grid connection cost
- Writes both detailed and aggregated outputs per site.

Typical usage:
    python HPP/Sensitivity.py --site Golfe_du_Lion NordsoenMidt --start-year 1996 --end-year 2005
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

# Reuse the existing evaluation pipeline to stay consistent with prior results.
# This import works when running the script as: python HPP/Sensitivity.py
import Evaluation as evaluation


# Cost groups used to apply CAPEX multipliers consistently.
WIND_CAPEX_KEYS = [
    "wind_turbine_cost",
    "wind_civil_works_cost",
]
PV_CAPEX_KEYS = [
    "solar_PV_cost",
    "solar_hardware_installation_cost",
    "solar_inverter_cost",
]
BATTERY_CAPEX_KEYS = [
    "battery_energy_cost",
    "battery_power_cost",
    "battery_BOP_installation_commissioning_cost",
    "battery_control_system_cost",
]
WACC_KEYS = [
    "wind_WACC",
    "solar_WACC",
    "battery_WACC",
]
OPEX_KEYS = [
    "wind_fixed_onm_cost",
    "wind_variable_onm_cost",
    "solar_fixed_onm_cost",
    "battery_energy_onm_cost",
]


@dataclass(frozen=True)
class Scenario:
    """Represents a single sensitivity scenario."""

    scenario_id: str
    parameter_group: str
    level: float


def _load_yaml(path: str) -> Dict:
    """Loads a YAML file into a dictionary."""
    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file did not parse to dict: {path}")
    return data


def _write_yaml(path: str, data: Dict) -> None:
    """Writes a dictionary to YAML with stable key order."""
    with open(path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)


def _scale_parameters(sim_pars: Dict, keys: Iterable[str], scale: float) -> Dict:
    """Returns a copy of `sim_pars` with selected scalar keys multiplied by `scale`."""
    updated = dict(sim_pars)
    for key in keys:
        if key not in updated:
            continue
        updated[key] = float(updated[key]) * float(scale)
    return updated


def _set_lifetime(sim_pars: Dict, life_y: int) -> Dict:
    """
    Returns a copy of `sim_pars` with lifetime fields updated.

    HyDesign supports both `N_life` and `life_y` conventions depending on
    assembly path. Setting both avoids ambiguity.
    """
    updated = dict(sim_pars)
    updated["N_life"] = int(life_y)
    updated["life_y"] = int(life_y)
    return updated


def _build_ofat_scenarios(
    include_lifetime: bool = True,
    include_battery: bool = True,
) -> List[Scenario]:
    """
    Builds OFAT scenarios around baseline values.

    The baseline scenario is always included first.
    """
    scenarios = [Scenario("baseline", "baseline", 1.0)]

    # Multipliers are intentionally modest to remain decision-relevant while
    # avoiding unrealistic corner cases.
    capex_levels = [0.8, 0.9, 1.1, 1.2]
    wacc_levels = [0.8, 0.9, 1.1, 1.2]
    grid_levels = [0.8, 0.9, 1.1, 1.2]

    for lvl in capex_levels:
        scenarios.append(Scenario(f"wind_capex_x{lvl:.2f}", "wind_capex", lvl))
    for lvl in capex_levels:
        scenarios.append(Scenario(f"pv_capex_x{lvl:.2f}", "pv_capex", lvl))
    if include_battery:
        for lvl in capex_levels:
            scenarios.append(
                Scenario(f"battery_capex_x{lvl:.2f}", "battery_capex", lvl)
            )
    for lvl in wacc_levels:
        scenarios.append(Scenario(f"wacc_x{lvl:.2f}", "wacc", lvl))
    for lvl in grid_levels:
        scenarios.append(Scenario(f"grid_connection_x{lvl:.2f}", "grid_connection", lvl))

    # Revenue-side assumptions.
    for pct in range(-10, 45, 5):
        if pct == 0:
            continue
        mult = 1.0 + pct / 100.0
        scenarios.append(
            Scenario(f"price_level_{pct:+d}pct", "price_level", mult)
        )

    for lvl in [0.8, 0.9, 1.1, 1.2, 1.3, 1.4]:
        scenarios.append(
            Scenario(f"price_volatility_x{lvl:.2f}", "price_volatility", lvl)
        )

    # OPEX assumptions.
    for lvl in [0.8, 0.9, 1.1, 1.2]:
        scenarios.append(Scenario(f"opex_x{lvl:.2f}", "opex", lvl))

    # Lifetime is specified directly in years, not as a multiplier.
    if include_lifetime:
        for years in range(20, 35, 2):
            scenarios.append(Scenario(f"lifetime_{years}y", "lifetime", float(years)))

    return scenarios


def _apply_scenario(sim_pars: Dict, baseline_life: int, scenario: Scenario) -> Tuple[Dict, int]:
    """
    Applies one scenario to baseline simulation parameters.

    Returns:
        tuple[dict, int]: (updated_sim_pars, effective_lifetime_years)
    """
    if scenario.parameter_group == "baseline":
        return dict(sim_pars), int(baseline_life)

    if scenario.parameter_group == "wind_capex":
        return _scale_parameters(sim_pars, WIND_CAPEX_KEYS, scenario.level), int(baseline_life)

    if scenario.parameter_group == "pv_capex":
        return _scale_parameters(sim_pars, PV_CAPEX_KEYS, scenario.level), int(baseline_life)

    if scenario.parameter_group == "battery_capex":
        return _scale_parameters(sim_pars, BATTERY_CAPEX_KEYS, scenario.level), int(baseline_life)

    if scenario.parameter_group == "wacc":
        return _scale_parameters(sim_pars, WACC_KEYS, scenario.level), int(baseline_life)

    if scenario.parameter_group == "grid_connection":
        return _scale_parameters(sim_pars, ["hpp_grid_connection_cost"], scenario.level), int(baseline_life)

    if scenario.parameter_group == "opex":
        return _scale_parameters(sim_pars, OPEX_KEYS, scenario.level), int(
            baseline_life
        )

    if scenario.parameter_group in {"price_level", "price_volatility"}:
        return dict(sim_pars), int(baseline_life)

    if scenario.parameter_group == "lifetime":
        life_y = int(round(scenario.level))
        return _set_lifetime(sim_pars, life_y), life_y

    raise ValueError(f"Unsupported scenario group: {scenario.parameter_group}")


def _safe_float(series: pd.Series) -> pd.Series:
    """Converts a series to float, coercing failures to NaN."""
    return pd.to_numeric(series, errors="coerce")


def _prepare_input_ts_for_scenario(
    input_ts_path: str,
    scenario: Scenario,
) -> Optional[str]:
    """
    Creates a temporary input time series file for price-based scenarios.

    Returns:
        Optional[str]: Path to temporary CSV when needed, else None.
    """
    if scenario.parameter_group not in {"price_level", "price_volatility"}:
        return None

    input_ts = evaluation._read_input_ts(input_ts_path).copy()
    price_cols = [
        col for col in input_ts.columns if "price" in str(col).lower()
    ]
    if not price_cols:
        return None

    for col in price_cols:
        s = pd.to_numeric(input_ts[col], errors="coerce")
        if scenario.parameter_group == "price_level":
            s_mod = s * float(scenario.level)
        else:
            s_mean = float(s.mean())
            s_mod = s_mean + float(scenario.level) * (s - s_mean)
        input_ts[col] = s_mod

    # Make datetime serialization explicit so Evaluation._read_input_ts can
    # always reconstruct a DatetimeIndex from the temporary file.
    if not isinstance(input_ts.index, pd.DatetimeIndex):
        parsed_idx = pd.to_datetime(
            input_ts.index,
            errors="coerce",
            dayfirst=True,
        )
        if parsed_idx.isna().any():
            bad_vals = input_ts.index[parsed_idx.isna()]
            sample = ", ".join(str(v) for v in bad_vals[:3])
            raise ValueError(
                "Cannot write scenario input CSV because index contains "
                f"non-datetime values. Sample: {sample}"
            )
        input_ts.index = parsed_idx

    ts_out = input_ts.copy()
    ts_out.index = ts_out.index.strftime("%d-%m-%Y %H:%M:%S")

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        delete=False,
        encoding="utf-8",
    ) as tmp_ts:
        tmp_ts_path = tmp_ts.name

    ts_out.to_csv(tmp_ts_path, sep=";", index_label="timestamp")
    return tmp_ts_path


def _summarize_by_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates yearly outputs into per-scenario summary statistics.

    Summary includes mean and P10/P50/P90 for key financial metrics.
    """
    if df.empty:
        return pd.DataFrame()

    metric_cols = [
        "NPV [MEuro]",
        "NPV_over_CAPEX",
        "IRR",
        "LCOE [Euro/MWh]",
        "Revenues [MEuro]",
        "LLCR [-]",
    ]
    metric_cols = [col for col in metric_cols if col in df.columns]

    group_cols = ["site", "scenario_id", "parameter_group", "level"]
    rows = []

    for keys, grp in df.groupby(group_cols, dropna=False):
        row = {
            "site": keys[0],
            "scenario_id": keys[1],
            "parameter_group": keys[2],
            "level": keys[3],
            "n_weather_years": int(grp["weather_year"].nunique()) if "weather_year" in grp else len(grp),
        }
        for col in metric_cols:
            vals = _safe_float(grp[col]).dropna().to_numpy()
            if vals.size == 0:
                row[f"{col} mean"] = np.nan
                row[f"{col} std"] = np.nan
                row[f"{col} p10"] = np.nan
                row[f"{col} p50"] = np.nan
                row[f"{col} p90"] = np.nan
            else:
                row[f"{col} mean"] = float(np.mean(vals))
                if vals.size > 1:
                    row[f"{col} std"] = float(np.std(vals, ddof=1))
                else:
                    row[f"{col} std"] = np.nan
                row[f"{col} p10"] = float(np.percentile(vals, 10))
                row[f"{col} p50"] = float(np.percentile(vals, 50))
                row[f"{col} p90"] = float(np.percentile(vals, 90))
        rows.append(row)

    return pd.DataFrame(rows)


def _build_tornado_frame(summary_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """
    Builds tornado-chart input with min/max delta versus baseline by parameter group.

    The returned frame has one row per parameter group and three columns:
    group label, min delta, and max delta.
    """
    if summary_df.empty or metric_col not in summary_df.columns:
        return pd.DataFrame()

    baseline_rows = summary_df.loc[summary_df["scenario_id"] == "baseline"]
    if baseline_rows.empty:
        return pd.DataFrame()

    baseline_val = pd.to_numeric(baseline_rows.iloc[0][metric_col], errors="coerce")
    if pd.isna(baseline_val):
        return pd.DataFrame()

    label_map = {
        "wind_capex": "Wind CAPEX",
        "pv_capex": "PV CAPEX",
        "battery_capex": "Battery CAPEX",
        "wacc": "WACC",
        "grid_connection": "Grid connection",
        "lifetime": "Lifetime",
        "price_level": "Price level",
        "price_volatility": "Price volatility",
        "opex": "OPEX",
    }

    def _format_range(group_name: str, grp: pd.DataFrame) -> str:
        levels = pd.to_numeric(grp["level"], errors="coerce").dropna().to_numpy()
        if levels.size == 0:
            return ""

        low = float(np.min(levels))
        high = float(np.max(levels))
        if np.isclose(low, high):
            return ""

        if group_name == "lifetime":
            return f" ({low:.0f}-{high:.0f} y)"

        if group_name == "price_level":
            low_pct = (low - 1.0) * 100.0
            high_pct = (high - 1.0) * 100.0
            return f" ({low_pct:+.0f}% to {high_pct:+.0f}%)"

        return f" ({low:.2f}x-{high:.2f}x)"

    rows = []
    std_col = metric_col.replace(" mean", " std")
    has_std = std_col in summary_df.columns
    work_df = summary_df.loc[summary_df["parameter_group"] != "baseline"].copy()
    for group_name, grp in work_df.groupby("parameter_group", dropna=False):
        grp_vals = pd.to_numeric(grp[metric_col], errors="coerce")
        valid_mask = grp_vals.notna()
        if not valid_mask.any():
            continue

        grp_valid = grp.loc[valid_mask].copy()
        grp_valid["_metric"] = grp_vals.loc[valid_mask].astype(float)
        grp_valid["_delta"] = grp_valid["_metric"] - float(baseline_val)

        min_idx = grp_valid["_delta"].idxmin()
        max_idx = grp_valid["_delta"].idxmax()
        delta_min = float(grp_valid.loc[min_idx, "_delta"])
        delta_max = float(grp_valid.loc[max_idx, "_delta"])

        std_min = np.nan
        std_max = np.nan
        if has_std:
            std_min = pd.to_numeric(grp_valid.loc[min_idx, std_col], errors="coerce")
            std_max = pd.to_numeric(grp_valid.loc[max_idx, std_col], errors="coerce")

        range_txt = _format_range(str(group_name), grp)
        rows.append(
            {
                "parameter_group": group_name,
                "label": label_map.get(str(group_name), str(group_name)) + range_txt,
                "delta_min": delta_min,
                "delta_max": delta_max,
                "std_min": float(std_min) if pd.notna(std_min) else np.nan,
                "std_max": float(std_max) if pd.notna(std_max) else np.nan,
                "impact_abs": float(max(abs(delta_min), abs(delta_max))),
            }
        )

    if not rows:
        return pd.DataFrame()

    tornado_df = pd.DataFrame(rows)
    tornado_df = tornado_df.sort_values("impact_abs", ascending=True)
    return tornado_df


def _save_tornado_plot(
    summary_df: pd.DataFrame,
    metric_col: str,
    x_label: str,
    title: str,
    output_path: str,
) -> bool:
    """
    Saves a horizontal tornado chart for one metric.

    Returns True if a chart was written, False otherwise.
    """
    if plt is None:
        return False

    tornado_df = _build_tornado_frame(summary_df, metric_col)
    if tornado_df.empty:
        return False

    y = np.arange(len(tornado_df))
    left = tornado_df["delta_min"].to_numpy()
    width = tornado_df["delta_max"].to_numpy() - tornado_df["delta_min"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, max(4, 0.55 * len(tornado_df) + 1.5)))
    ax.barh(y, width, left=left, color="#5B8FF9", alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(tornado_df["label"].tolist())
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.grid(axis="x", alpha=0.25)

    # If std is available (multi-year runs), add thin whiskers at both ends
    # of the tornado bar to communicate inter-annual variability.
    if "std_min" in tornado_df.columns and "std_max" in tornado_df.columns:
        for i, row in enumerate(tornado_df.itertuples(index=False)):
            if pd.notna(row.std_min) and row.std_min > 0:
                ax.hlines(i, row.delta_min - row.std_min, row.delta_min,
                          colors="#264653", linewidth=1.4)
            if pd.notna(row.std_max) and row.std_max > 0:
                ax.hlines(i, row.delta_max, row.delta_max + row.std_max,
                          colors="#264653", linewidth=1.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def _generate_tornado_plots(site_name: str, summary_df: pd.DataFrame, output_dir: str) -> List[str]:
    """
    Generates standard tornado plots for a site and returns saved file paths.
    """
    if plt is None:
        print("matplotlib is unavailable; skipping tornado plots.")
        return []

    # Keep tornado focused on non-lifetime drivers.
    tornado_df = summary_df.loc[
        summary_df["parameter_group"] != "lifetime"
    ].copy()
    if tornado_df.empty:
        return []

    plot_specs = [
        (
            "NPV [MEuro] mean",
            "Delta NPV [MEuro] vs baseline",
            "NPV tornado (mean)",
            "npv",
        ),
        (
            "LCOE [Euro/MWh] mean",
            "Delta LCOE [Euro/MWh] vs baseline",
            "LCOE tornado (mean)",
            "lcoe",
        ),
        (
            "NPV_over_CAPEX mean",
            "Delta NPV/CAPEX [-] vs baseline",
            "NPV/CAPEX tornado (mean)",
            "npv_over_capex",
        ),
    ]

    saved_paths: List[str] = []
    for metric_col, x_label, title, suffix in plot_specs:
        out_path = os.path.join(output_dir, f"{site_name}_tornado_{suffix}.png")
        ok = _save_tornado_plot(
            summary_df=tornado_df,
            metric_col=metric_col,
            x_label=x_label,
            title=f"{site_name} - {title}",
            output_path=out_path,
        )
        if ok:
            saved_paths.append(out_path)

    return saved_paths


def _save_lifetime_plot(
    site_name: str,
    summary_df: pd.DataFrame,
    metric_col: str,
    y_label: str,
    output_path: str,
) -> bool:
    """Saves a dedicated lifetime sensitivity plot for one metric."""
    if plt is None:
        return False

    work = summary_df.loc[summary_df["parameter_group"] == "lifetime"].copy()
    if work.empty or metric_col not in work.columns:
        return False

    work["_x"] = pd.to_numeric(work["level"], errors="coerce")
    work["_y"] = pd.to_numeric(work[metric_col], errors="coerce")
    work = work.dropna(subset=["_x", "_y"]).sort_values("_x")
    if work.empty:
        return False

    std_col = metric_col.replace(" mean", " std")
    yerr = None
    if std_col in work.columns:
        std_vals = pd.to_numeric(work[std_col], errors="coerce")
        if std_vals.notna().any():
            yerr = std_vals.to_numpy()

    baseline_rows = summary_df.loc[summary_df["scenario_id"] == "baseline"]
    baseline_y = None
    if not baseline_rows.empty and metric_col in baseline_rows.columns:
        baseline_y = pd.to_numeric(
            baseline_rows.iloc[0][metric_col],
            errors="coerce",
        )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(work["_x"], work["_y"], marker="o", color="#3A86FF")
    if yerr is not None:
        ax.errorbar(
            work["_x"],
            work["_y"],
            yerr=yerr,
            fmt="none",
            ecolor="#1D3557",
            elinewidth=1.2,
            capsize=3,
        )

    if pd.notna(baseline_y):
        ax.axhline(
            float(baseline_y),
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="Baseline",
        )
        ax.legend(loc="best")

    ax.set_title(f"{site_name} - Lifetime sensitivity")
    ax.set_xlabel("Lifetime [years]")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def _generate_lifetime_plots(
    site_name: str,
    summary_df: pd.DataFrame,
    output_dir: str,
) -> List[str]:
    """Generates dedicated lifetime sensitivity plots and returns file paths."""
    if plt is None:
        return []

    lifetime_df = summary_df.loc[summary_df["parameter_group"] == "lifetime"]
    if lifetime_df.empty:
        return []

    plot_specs = [
        (
            "NPV [MEuro] mean",
            "NPV [MEuro]",
            "lifetime_npv",
        ),
        (
            "LCOE [Euro/MWh] mean",
            "LCOE [Euro/MWh]",
            "lifetime_lcoe",
        ),
        (
            "NPV_over_CAPEX mean",
            "NPV/CAPEX [-]",
            "lifetime_npv_over_capex",
        ),
    ]

    saved_paths: List[str] = []
    for metric_col, y_label, suffix in plot_specs:
        out_path = os.path.join(output_dir, f"{site_name}_{suffix}.png")
        ok = _save_lifetime_plot(
            site_name=site_name,
            summary_df=summary_df,
            metric_col=metric_col,
            y_label=y_label,
            output_path=out_path,
        )
        if ok:
            saved_paths.append(out_path)

    return saved_paths


def _evaluate_site_sensitivity(
    site_name: str,
    start_year: int,
    end_year: int,
    price_add: float,
    output_dir: str,
    save_plots: bool,
    include_lifetime: bool,
    fixed_lifetime_years: Optional[int] = None,
) -> Tuple[str, str, List[str]]:
    """
    Runs OFAT sensitivity for a single site and writes result CSV files.

    Returns:
        tuple[str, str, list[str]]: Paths to detailed CSV, summary CSV, and plots.
    """
    site_config_dir = evaluation._get_site_config_dir()

    ex_site = evaluation._load_site_row(site_name)
    design = evaluation._load_site_design(site_name, site_config_dir)

    sim_pars_path = evaluation.examples_filepath + ex_site["sim_pars_fn"]
    input_ts_path = evaluation.examples_filepath + ex_site["input_ts_fn"]
    base_sim_pars = _load_yaml(sim_pars_path)

    baseline_life = int(base_sim_pars.get("life_y", base_sim_pars.get("N_life", 25)))

    if fixed_lifetime_years is not None:
        baseline_life = int(fixed_lifetime_years)
        base_sim_pars = _set_lifetime(base_sim_pars, baseline_life)

    has_battery = bool(design.get("b_P", 0) > 0 and design.get("b_E_h", 0) > 0)
    scenarios = _build_ofat_scenarios(
        include_lifetime=include_lifetime,
        include_battery=has_battery,
    )

    if not has_battery:
        print(
            f"Site {site_name}: battery not used in design "
            f"(b_P={design.get('b_P')}, b_E_h={design.get('b_E_h')}); "
            "skipping battery CAPEX scenarios."
        )
    total_scenarios = len(scenarios)
    weather_years_count = end_year - start_year + 1
    total_model_runs = total_scenarios * weather_years_count

    print(
        f"Sensitivity plan for {site_name}: "
        f"{total_scenarios} scenarios x {weather_years_count} weather year(s) "
        f"= {total_model_runs} model run(s)"
    )

    rows: List[pd.DataFrame] = []
    sensitivity_start = time.time()

    for idx, scenario in enumerate(scenarios, start=1):
        sim_pars_mod, life_y = _apply_scenario(base_sim_pars, baseline_life, scenario)
        scenario_start = time.time()
        scenario_input_ts_path = input_ts_path
        tmp_input_ts_path: Optional[str] = None

        print(
            f"[{idx}/{total_scenarios}] Running scenario: {scenario.scenario_id} "
            f"(group={scenario.parameter_group}, level={scenario.level})"
        )

        # Use a temporary YAML file so each scenario remains isolated and reproducible.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False, encoding="utf-8"
        ) as tmp:
            tmp_path = tmp.name
        try:
            _write_yaml(tmp_path, sim_pars_mod)
            tmp_input_ts_path = _prepare_input_ts_for_scenario(
                input_ts_path=input_ts_path,
                scenario=scenario,
            )
            if tmp_input_ts_path is not None:
                scenario_input_ts_path = tmp_input_ts_path

            df = evaluation.evaluate_yearly_lifetime(
                site_name=site_name,
                latitude=ex_site["latitude"],
                longitude=ex_site["longitude"],
                altitude=ex_site["altitude"],
                sim_pars_fn=tmp_path,
                input_ts_fn=scenario_input_ts_path,
                design=design,
                start_year=start_year,
                end_year=end_year,
                lifetime_years=life_y,
                price_add=price_add,
                save_hourly_csv=False,
            )
            df.insert(0, "scenario_id", scenario.scenario_id)
            df.insert(1, "parameter_group", scenario.parameter_group)
            df.insert(2, "level", scenario.level)
            rows.append(df)

            scenario_secs = time.time() - scenario_start
            elapsed = time.time() - sensitivity_start
            avg = elapsed / idx
            remaining = max(total_scenarios - idx, 0) * avg
            print(
                f"[{idx}/{total_scenarios}] Completed in {scenario_secs:.1f}s | "
                f"ETA ~ {remaining / 60.0:.1f} min"
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if tmp_input_ts_path and os.path.exists(tmp_input_ts_path):
                os.remove(tmp_input_ts_path)

    detailed = pd.concat(rows, ignore_index=True)
    summary = _summarize_by_scenario(detailed)

    os.makedirs(output_dir, exist_ok=True)
    detailed_csv = os.path.join(output_dir, f"{site_name}_sensitivity_detail_{start_year}_{end_year}.csv")
    summary_csv = os.path.join(output_dir, f"{site_name}_sensitivity_summary_{start_year}_{end_year}.csv")

    detailed.to_csv(detailed_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    plot_paths: List[str] = []
    if save_plots:
        plot_paths = _generate_tornado_plots(
            site_name=site_name,
            summary_df=summary,
            output_dir=output_dir,
        )
        plot_paths.extend(
            _generate_lifetime_plots(
                site_name=site_name,
                summary_df=summary,
                output_dir=output_dir,
            )
        )

    return detailed_csv, summary_csv, plot_paths


def main() -> None:
    """CLI entry point for sensitivity analysis."""
    evaluation._init_local_hydesign_imports()

    parser = argparse.ArgumentParser(
        description="Run OFAT sensitivity analysis using predefined site configuration files."
    )
    parser.add_argument(
        "--site",
        nargs="+",
        default=None,
        help="Site names from HPP/SiteConfig. If omitted, all site configs are processed.",
    )
    parser.add_argument(
        "--list-sites",
        action="store_true",
        help="List available site names and exit.",
    )
    parser.add_argument("--start-year", type=int, default=1982)
    parser.add_argument("--end-year", type=int, default=1982)
    parser.add_argument(
        "--price-add",
        type=float,
        default=30.0,
        help="Price offset in EUR/MWh used by the evaluation script.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(evaluation._get_evaluations_dir(), "Sensitivity"),
        help="Directory where sensitivity CSV files are written.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip tornado plot generation.",
    )
    parser.add_argument(
        "--exclude-lifetime",
        action="store_true",
        help="Do not include lifetime sensitivity scenarios (keeps baseline lifetime).",
    )
    args = parser.parse_args()

    available_sites = evaluation._available_site_configs(evaluation._get_site_config_dir())

    if args.list_sites:
        print("Available site configs:")
        for name in available_sites:
            print(f"- {name}")
        return

    if args.site is None:
        target_sites = available_sites
    else:
        target_sites = args.site

    invalid = [site for site in target_sites if site not in available_sites]
    if invalid:
        raise ValueError(
            "Invalid site(s): "
            + ", ".join(invalid)
            + ". Use --list-sites to see supported names from HPP/SiteConfig."
        )

    started = time.time()
    print("Starting sensitivity analysis")
    print(f"Sites: {', '.join(target_sites)}")
    print(f"Weather years: {args.start_year}..{args.end_year}")
    print(f"Output dir: {args.output_dir}")

    for site_name in target_sites:
        print("-" * 80)
        print(f"Processing site: {site_name}")
        site_start = time.time()
        detail_path, summary_path, plot_paths = _evaluate_site_sensitivity(
            site_name=site_name,
            start_year=args.start_year,
            end_year=args.end_year,
            price_add=args.price_add,
            output_dir=args.output_dir,
            save_plots=not args.skip_plots,
            include_lifetime=not args.exclude_lifetime,
        )
        mins = (time.time() - site_start) / 60.0
        print(f"Saved detailed results: {detail_path}")
        print(f"Saved summary results:  {summary_path}")
        if plot_paths:
            print("Saved tornado plots:")
            for path in plot_paths:
                print(f"- {path}")
        elif not args.skip_plots:
            print("No tornado plots were produced (missing metrics or plotting backend).")
        print(f"Site runtime: {mins:.2f} min")

    total_mins = (time.time() - started) / 60.0
    print("-" * 80)
    print(f"Completed sensitivity analysis in {total_mins:.2f} min")


def run_one_site_one_year_25y_example() -> None:
    """
    Code-native example run (no terminal arguments needed).

    This runs one site for one weather year while fixing project lifetime to
    25 years and excluding lifetime sensitivity scenarios.
    """
    evaluation._init_local_hydesign_imports()

    site_name = "Golfe_du_Lion"
    #weather_year = 1996
    output_dir = os.path.join(evaluation._get_evaluations_dir(), "Sensitivity")

    detail_path, summary_path, plot_paths = _evaluate_site_sensitivity(
        site_name=site_name,
        start_year=1982,
        end_year=2015,
        price_add=30.0,
        output_dir=output_dir,
        save_plots=True,
        include_lifetime=True,
        fixed_lifetime_years=25,
    )

    print("Example run completed")
    print(f"Detailed CSV: {detail_path}")
    print(f"Summary CSV:  {summary_path}")
    if plot_paths:
        print("Plots:")
        for path in plot_paths:
            print(f"- {path}")


if __name__ == "__main__":
    # Set to True to run the in-code example configuration directly.
    RUN_CODE_EXAMPLE = True

    if RUN_CODE_EXAMPLE:
        run_one_site_one_year_25y_example()
    else:
        main()

