"""
Sensitivity analysis for HPP configurations using HiFiEMS evaluation.

This module performs a one-factor-at-a-time (OFAT) sensitivity study
specifically designed for the HiFiEMS energy management system.

Key features:
- Works with HiFiEMS-configured sites (e.g., Golfe_du_Lion_HiFiEMS)
- Parallel evaluation across all scenario-year combinations using ProcessPoolExecutor
- Varies key economic and technical assumptions:
  - Wind CAPEX
  - PV CAPEX
  - WACC
  - Grid connection cost
  - OPEX
  - Electricity price
- Exports detailed results and summary statistics per scenario

Typical usage:
    python HPP/HelperFunctions/Sensitivity_HiFiEMS.py --site Golfe_du_Lion_HiFiEMS --start-year 1982 --end-year 2015
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

# Add parent directories to path to ensure local imports
script_file = os.path.abspath(__file__)
helper_dir = os.path.dirname(script_file)  # HelperFunctions/
hpp_dir = os.path.dirname(helper_dir)  # HPP/
repo_root = os.path.dirname(hpp_dir)  # hydesign/ (repo root)

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
if hpp_dir not in sys.path:
    sys.path.insert(0, hpp_dir)

# Import HiFiEMS evaluation module
import Evaluation_HiFiEMS as evaluation


# Cost groups for CAPEX multipliers
WIND_CAPEX_KEYS = [
    "wind_turbine_cost",
    "wind_civil_works_cost",
]
PV_CAPEX_KEYS = [
    "solar_PV_cost",
    "solar_hardware_installation_cost",
    "solar_inverter_cost",
]
WACC_KEYS = [
    "wind_WACC",
    "solar_WACC",
    "battery_WACC",
]
# Separate OPEX by technology
WIND_OPEX_KEYS = [
    "wind_fixed_onm_cost",
    "wind_variable_onm_cost",
]
SOLAR_OPEX_KEYS = [
    "solar_fixed_onm_cost",
]
ELECTRICITY_PRICE_KEYS = [
    "electricity_price",
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


def _build_ofat_scenarios(scenario_filter: Optional[str] = None) -> List[Scenario]:
    """
    Builds OFAT scenarios around baseline values.
    
    Args:
        scenario_filter: Optional filter to include only specific scenario types.
                        Options: 'all' (default), 'wind_solar'/'generation', 'costs', 'finance', 'price'
    
    The baseline scenario is always included first.
    """
    scenarios = [Scenario("baseline", "baseline", 1.0)]
    
    capex_levels = [0.8, 0.9, 1.1, 1.2]
    wacc_levels = [0.8, 0.9, 1.1, 1.2]
    grid_levels = [0.8, 0.9, 1.1, 1.2]
    opex_levels = [0.8, 0.9, 1.1, 1.2]
    # Wind and solar generation: ±20% in 5% steps (0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2)
    generation_levels = [0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2]
    
    # Default filter
    if scenario_filter is None or scenario_filter.lower() == 'all':
        # Include all scenarios
        include_capex = True
        include_wacc = True
        include_grid = True
        include_opex = True
        include_price = True
        include_generation = True
    elif scenario_filter.lower() in ('wind_solar', 'generation'):
        # Only wind and solar generation
        include_capex = False
        include_wacc = False
        include_grid = False
        include_opex = False
        include_price = False
        include_generation = True
    elif scenario_filter.lower() == 'costs':
        # Only cost-related scenarios (CAPEX, OPEX, Grid)
        include_capex = True
        include_wacc = False
        include_grid = True
        include_opex = True
        include_price = False
        include_generation = False
    elif scenario_filter.lower() == 'finance':
        # Only financial scenarios (WACC, Price)
        include_capex = False
        include_wacc = True
        include_grid = False
        include_opex = False
        include_price = True
        include_generation = False
    elif scenario_filter.lower() == 'price':
        # Only electricity price
        include_capex = False
        include_wacc = False
        include_grid = False
        include_opex = False
        include_price = True
        include_generation = False
    else:
        raise ValueError(f"Unknown scenario filter: {scenario_filter}. Valid options: all, wind_solar, generation, costs, finance, price")
    
    if include_capex:
        for lvl in capex_levels:
            scenarios.append(Scenario(f"wind_capex_x{lvl:.2f}", "wind_capex", lvl))
        for lvl in capex_levels:
            scenarios.append(Scenario(f"pv_capex_x{lvl:.2f}", "pv_capex", lvl))
    
    if include_wacc:
        for lvl in wacc_levels:
            scenarios.append(Scenario(f"wacc_x{lvl:.2f}", "wacc", lvl))
    
    if include_grid:
        for lvl in grid_levels:
            scenarios.append(Scenario(f"grid_connection_x{lvl:.2f}", "grid_connection", lvl))
    
    if include_opex:
        # Separate OPEX by technology
        for lvl in opex_levels:
            scenarios.append(Scenario(f"wind_opex_x{lvl:.2f}", "wind_opex", lvl))
        for lvl in opex_levels:
            scenarios.append(Scenario(f"solar_opex_x{lvl:.2f}", "solar_opex", lvl))
    
    if include_price:
        # Electricity price: 0.8 and 1.2 only
        for lvl in [0.8, 1.2]:
            scenarios.append(Scenario(f"electricity_price_x{lvl:.2f}", "electricity_price", lvl))
    
    if include_generation:
        # Wind and solar generation sensitivity: ±20% in 5% steps
        for lvl in generation_levels:
            scenarios.append(Scenario(f"wind_generation_x{lvl:.2f}", "wind_generation", lvl))
        for lvl in generation_levels:
            scenarios.append(Scenario(f"solar_generation_x{lvl:.2f}", "solar_generation", lvl))
    
    return scenarios


def _apply_scenario(
    sim_pars: Dict, scenario: Scenario
) -> Dict:
    """
    Applies one scenario to baseline simulation parameters.
    
    Returns:
        dict: updated_sim_pars
    """
    if scenario.parameter_group == "baseline":
        return dict(sim_pars)
    
    if scenario.parameter_group == "wind_capex":
        return _scale_parameters(sim_pars, WIND_CAPEX_KEYS, scenario.level)
    
    if scenario.parameter_group == "pv_capex":
        return _scale_parameters(sim_pars, PV_CAPEX_KEYS, scenario.level)
    
    if scenario.parameter_group == "wacc":
        return _scale_parameters(sim_pars, WACC_KEYS, scenario.level)
    
    if scenario.parameter_group == "grid_connection":
        return _scale_parameters(sim_pars, ["hpp_grid_connection_cost"], scenario.level)
    
    if scenario.parameter_group == "wind_opex":
        return _scale_parameters(sim_pars, WIND_OPEX_KEYS, scenario.level)
    
    if scenario.parameter_group == "solar_opex":
        return _scale_parameters(sim_pars, SOLAR_OPEX_KEYS, scenario.level)
    
    if scenario.parameter_group == "electricity_price":
        # Don't modify YAML parameters for electricity price scenarios
        # The price_add parameter will handle the adjustment
        return dict(sim_pars)
    
    if scenario.parameter_group == "wind_generation":
        # Don't modify YAML parameters for wind generation scenarios
        # The generation multiplier will be applied to weather data during evaluation
        return dict(sim_pars)
    
    if scenario.parameter_group == "solar_generation":
        # Don't modify YAML parameters for solar generation scenarios
        # The generation multiplier will be applied to weather data during evaluation
        return dict(sim_pars)
    
    raise ValueError(f"Unsupported scenario group: {scenario.parameter_group}")


def _safe_float(series: pd.Series) -> pd.Series:
    """Converts a series to float, coercing failures to NaN."""
    return pd.to_numeric(series, errors="coerce")


def _evaluate_scenario_year(
    scenario: Scenario,
    year: int,
    site_name: str,
    base_site_name: str,
    base_sim_pars: Dict,
    year_df: pd.DataFrame,
    design: Dict,
    lifetime_years: int,
    ex_site: Dict,
    examples_filepath: str,
) -> Tuple[Optional[Dict], Scenario, int]:
    """
    Worker function to evaluate a single scenario-year combination.
    
    Handles wind and solar generation sensitivity by creating scaled data files.
    
    Returns:
        tuple: (result_row, scenario, year) or (None, scenario, year) if failed
    """
    try:
        sim_pars_mod = _apply_scenario(base_sim_pars, scenario)
        
        # Calculate price_add for electricity price scenarios
        # For multiplicative scaling, pass the multiplier as price_add
        price_add = 0.0
        if scenario.parameter_group == "electricity_price":
            # Pass the multiplier directly (1.0, 1.25, 1.5, etc.)
            # Evaluation code will multiply: new_price = old_price * price_multiplier
            price_add = scenario.level
        
        # Handle wind and solar generation multipliers
        wind_generation_mult = 1.0
        solar_generation_mult = 1.0
        if scenario.parameter_group == "wind_generation":
            wind_generation_mult = scenario.level
        elif scenario.parameter_group == "solar_generation":
            solar_generation_mult = scenario.level
        
        # Create temporary directory for this evaluation
        with tempfile.TemporaryDirectory(prefix=f"sens_{scenario.scenario_id}_{year}_") as temp_dir:
            # Write scenario-specific YAML
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yml", delete=False, dir=temp_dir, encoding="utf-8"
            ) as tmp:
                tmp_path = tmp.name
            
            try:
                # Ensure wind/solar file paths are absolute (resolve from examples_filepath if needed)
                if not os.path.isabs(sim_pars_mod.get("wind_fn", "")):
                    # Path is relative - resolve from examples_filepath
                    wind_fn_abs = os.path.join(examples_filepath, sim_pars_mod["wind_fn"])
                else:
                    wind_fn_abs = sim_pars_mod["wind_fn"]
                
                if not os.path.isabs(sim_pars_mod.get("solar_fn", "")):
                    # Path is relative - resolve from examples_filepath
                    solar_fn_abs = os.path.join(examples_filepath, sim_pars_mod["solar_fn"])
                else:
                    solar_fn_abs = sim_pars_mod["solar_fn"]
                
                # Scale wind and solar data files if needed
                if wind_generation_mult != 1.0 and "wind_fn" in sim_pars_mod and os.path.isfile(wind_fn_abs):
                    try:
                        wind_data = pd.read_csv(wind_fn_abs, index_col=0, parse_dates=False, sep=None, engine="python")
                        # Scale all numeric columns (catches "Measurement", "Power", etc.)
                        numeric_cols = wind_data.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            wind_data[col] = wind_data[col] * wind_generation_mult
                        
                        # Save scaled wind data to temp directory
                        wind_fn_scaled = os.path.join(temp_dir, f"Winddata_{year}_scaled.csv")
                        wind_data.to_csv(wind_fn_scaled)
                        sim_pars_mod["wind_fn"] = wind_fn_scaled
                    except Exception as e:
                        print(f"Warning: Could not scale wind data for {scenario.scenario_id}: {e}")
                
                if solar_generation_mult != 1.0 and "solar_fn" in sim_pars_mod and os.path.isfile(solar_fn_abs):
                    try:
                        solar_data = pd.read_csv(solar_fn_abs, index_col=0, parse_dates=False, sep=None, engine="python")
                        # Scale all numeric columns (catches "Measurement", "Power", etc.)
                        numeric_cols = solar_data.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            solar_data[col] = solar_data[col] * solar_generation_mult
                        
                        # Save scaled solar data to temp directory
                        solar_fn_scaled = os.path.join(temp_dir, f"Solardata_{year}_scaled.csv")
                        solar_data.to_csv(solar_fn_scaled)
                        sim_pars_mod["solar_fn"] = solar_fn_scaled
                    except Exception as e:
                        print(f"Warning: Could not scale solar data for {scenario.scenario_id}: {e}")
                
                _write_yaml(tmp_path, sim_pars_mod)
                
                # Call evaluate_single_year
                row, hourly_df = evaluation.evaluate_single_year(
                    year=year,
                    parent_temp_dir=temp_dir,
                    site_name=site_name,
                    base_site_name=base_site_name,
                    sim_pars_fn=tmp_path,
                    year_df=year_df,
                    design=design,
                    lifetime_years=lifetime_years,
                    price_add=price_add,
                    save_hourly=False,
                    ex_site=ex_site,
                    examples_filepath=examples_filepath,
                )
                
                return (row, scenario, year)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    except Exception as e:
        print(f"Error evaluating {scenario.scenario_id} year {year}: {e}")
        return (None, scenario, year)


def _safe_float(series: pd.Series) -> pd.Series:
    """Converts a series to float, coercing failures to NaN."""
    return pd.to_numeric(series, errors="coerce")


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


def main():
    """CLI entry point for HiFiEMS sensitivity analysis."""
    # Import hydesign examples path from local workspace
    from hydesign.examples import examples_filepath
    
    parser = argparse.ArgumentParser(
        description="Run OFAT sensitivity analysis using HiFiEMS evaluation."
    )
    parser.add_argument(
        "--site",
        nargs='+',
        default=["Sud_Atlantique_HiFiEMS"],
        help="List of HiFiEMS site names (e.g., Golfe_du_Lion_HiFiEMS)",
    )
    parser.add_argument("--start-year", type=int, default=1982)
    parser.add_argument("--end-year", type=int, default=2015)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for results (default: HPP/Evaluations/Sensitivity_HiFiEMS)",
    )
    parser.add_argument(
        "--scenarios",
        default="all",
        help="Scenario filter: 'all' (default), 'wind_solar'/'generation' (wind+solar generation), 'costs' (capex/opex/grid), 'finance' (wacc/price), 'price' (electricity price only)",
    )
    args = parser.parse_args()
    
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(
            script_dir, "..", "Evaluations", "Sensitivity_HiFiEMS"
        )
    
    # Process each site
    for site_name in args.site:
        try:
            print(f"\n{'='*70}")
            print(f"Processing HiFiEMS Site: {site_name}")
            print(f"{'='*70}")
            
            start_time = time.time()
            
            # Get site metadata and configuration
            ex_site = evaluation._get_site_row(site_name, examples_filepath)
            site_config_dir = evaluation._get_site_config_dir()
            
            # Extract base site name (strip _HiFiEMS suffix if present)
            base_site_name = site_name.replace("_HiFiEMS", "")
            design = evaluation._load_site_design(base_site_name, site_config_dir)
            
            # Get simulation parameters
            sim_pars_path = examples_filepath + ex_site["sim_pars_fn"]
            input_ts_path = examples_filepath + ex_site["input_ts_fn"]
            base_sim_pars = _load_yaml(sim_pars_path)
            
            # Extract lifetime from configuration (fixed for all scenarios)
            lifetime_years = int(base_sim_pars.get("life_y", base_sim_pars.get("N_life", 25)))
            
            # Build scenarios
            scenarios = _build_ofat_scenarios(scenario_filter=args.scenarios)
            
            total_scenarios = len(scenarios)
            weather_years = args.end_year - args.start_year + 1
            total_runs = total_scenarios * weather_years
            
            print(
                f"HiFiEMS Sensitivity plan for {site_name}: "
                f"{total_scenarios} scenarios x {weather_years} weather year(s) = {total_runs} model run(s)"
            )
            
            # Read input time series once
            input_ts = evaluation._read_input_ts(input_ts_path)
            yearly_groups = {year: group for year, group in input_ts.groupby(input_ts.index.year)}
            
            # Prepare all scenario-year tasks
            tasks = []
            for scenario in scenarios:
                for year in range(args.start_year, args.end_year + 1):
                    if year not in yearly_groups:
                        continue
                    year_df = yearly_groups[year]
                    tasks.append((scenario, year, year_df))
            
            # Run scenarios in parallel
            all_results = []
            sensitivity_start = time.time()
            
            # Use ProcessPoolExecutor for parallel evaluation
            with ProcessPoolExecutor(max_workers=None) as executor:
                # Submit all tasks
                future_to_task = {}
                for scenario, year, year_df in tasks:
                    future = executor.submit(
                        _evaluate_scenario_year,
                        scenario=scenario,
                        year=year,
                        site_name=site_name,
                        base_site_name=base_site_name,
                        base_sim_pars=base_sim_pars,
                        year_df=year_df,
                        design=design,
                        lifetime_years=lifetime_years,
                        ex_site=ex_site,
                        examples_filepath=examples_filepath,
                    )
                    future_to_task[future] = (scenario, year)
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_task):
                    scenario, year = future_to_task[future]
                    completed += 1
                    
                    try:
                        row, res_scenario, res_year = future.result()
                        if row and "error" not in row:
                            all_results.append((row, res_scenario, res_year))
                        
                        elapsed = time.time() - sensitivity_start
                        avg_per_task = elapsed / completed
                        remaining = (len(tasks) - completed) * avg_per_task
                        
                        print(
                            f"[{completed}/{len(tasks)}] Completed {scenario.scenario_id} year {year} | "
                            f"ETA ~ {remaining / 60.0:.1f} min"
                        )
                    except Exception as e:
                        print(f"[{completed}/{len(tasks)}] Error in {scenario.scenario_id} year {year}: {e}")
            
            # Organize results by scenario
            scenario_results_map = {}
            for row, scenario, year in all_results:
                if scenario.scenario_id not in scenario_results_map:
                    scenario_results_map[scenario.scenario_id] = ([], scenario)
                scenario_results_map[scenario.scenario_id][0].append(row)
            
            # Convert to DataFrames and add metadata
            all_results_list = []
            for scenario_id, (rows, scenario) in scenario_results_map.items():
                if rows:
                    df = pd.DataFrame(rows)
                    df.insert(0, "scenario_id", scenario.scenario_id)
                    df.insert(1, "parameter_group", scenario.parameter_group)
                    df.insert(2, "level", scenario.level)
                    all_results_list.append(df)
            
            # Combine and summarize results
            detailed_df = pd.concat(all_results_list, ignore_index=True)
            summary_df = _summarize_by_scenario(detailed_df)
            
            # Save outputs
            os.makedirs(args.output_dir, exist_ok=True)
            detailed_csv = os.path.join(
                args.output_dir, f"{site_name}_hifiems_sensitivity_detail_{args.start_year}_{args.end_year}.csv"
            )
            summary_csv = os.path.join(
                args.output_dir, f"{site_name}_hifiems_sensitivity_summary_{args.start_year}_{args.end_year}.csv"
            )
            
            detailed_df.to_csv(detailed_csv, index=False)
            summary_df.to_csv(summary_csv, index=False)
            
            print(f"\nResults saved:")
            print(f"  Detailed: {detailed_csv}")
            print(f"  Summary:  {summary_csv}")
            
            elapsed_min = (time.time() - start_time) / 60.0
            print(f"Site completed in {elapsed_min:.1f} minutes\n")
            
        except Exception as e:
            print(f"Error processing site '{site_name}': {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
