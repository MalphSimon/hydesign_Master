"""
Evaluation of Hybrid Power Plants (HPPs) using site-specific configuration files.

This script evaluates the performance of a site design across multiple weather years, 
calculating key metrics (LCOE, NPV, etc.) and applying price offsets to simulate 
market conditions. Results are exported to CSV for analysis.
"""

import argparse
import os
import sys
import time
import tempfile

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# --- Global Placeholders ---
# These are populated at runtime by _init_local_hydesign_imports
hpp_model = None
examples_filepath = None
calculate_bankability_metrics = None


def _init_local_hydesign_imports():
    """
    Dynamically imports hydesign modules and sets global references.
    
    Adjusts the system path to include the repository root, allowing for
    local imports of HPP and hydesign packages.
    """
    global hpp_model
    global examples_filepath
    global calculate_bankability_metrics

    # Calculate repo root based on this file's location
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from HPP.HelperFunctions.Bankability import (
        calculate_bankability_metrics as local_calculate_bankability_metrics,
    )
    from hydesign.assembly.hpp_assembly import hpp_model as local_hpp_model
    from hydesign.examples import examples_filepath as local_examples_filepath

    calculate_bankability_metrics = local_calculate_bankability_metrics
    hpp_model = local_hpp_model
    examples_filepath = local_examples_filepath


def _get_site_config_dir():
    """Returns the absolute path to the 'SiteConfig' directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "SiteConfig")


def _get_evaluations_dir():
    """Returns the absolute path to the 'Evaluations' directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "Evaluations")


def _available_site_configs(site_config_dir):
    """
    Lists all available CSV site configurations in the specified directory.
    
    Args:
        site_config_dir (str): Path to the directory containing config files.
        
    Returns:
        list: Sorted list of site names (filenames without extensions).
    """
    if not os.path.isdir(site_config_dir):
        return []
    return sorted(
        os.path.splitext(file_name)[0]
        for file_name in os.listdir(site_config_dir)
        if file_name.lower().endswith(".csv")
    )


def _load_site_row(site_name):
    """
    Retrieves metadata (lat, lon, etc.) for a specific site from the examples master file.
    
    Args:
        site_name (str): The name of the site to look up.
        
    Returns:
        pd.Series: Row containing site metadata.
    """
    examples_sites = pd.read_csv(
        f"{examples_filepath}examples_sites.csv", index_col=0, sep=";"
    )
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    if ex_site.empty:
        raise ValueError(
            f"Site '{site_name}' not found in examples_sites.csv."
        )
    return ex_site.iloc[0]


def _load_site_design(site_name, site_config_dir):
    """
    Loads specific HPP design parameters from a site's CSV configuration.
    
    Args:
        site_name (str): Name of the site.
        site_config_dir (str): Path to directory where CSVs are stored.
        
    Returns:
        dict: A dictionary of design parameters (MW, heights, ratios, etc.).
    """
    site_config_path = os.path.join(site_config_dir, f"{site_name}.csv")
    if not os.path.isfile(site_config_path):
        available = _available_site_configs(site_config_dir)
        raise FileNotFoundError(
            f"Site config file not found: {site_config_path}. "
            f"Available site configs: {available}"
        )

    site_config = pd.read_csv(
        site_config_path,
        index_col=0,
        header=None,
        sep=",",
    )
    # Extract the last column (values) and drop any rows with NaN indices
    values = site_config.iloc[:, -1]
    values = values[~values.index.isna()]

    return {
        "clearance": float(values.loc["clearance [m]"]),
        "sp": float(values.loc["sp [W/m2]"]),
        "p_rated": float(values.loc["p_rated [MW]"]),
        "Nwt": int(float(values.loc["Nwt"])),
        "wind_MW_per_km2": float(values.loc["wind_MW_per_km2 [MW/km2]"]),
        "solar_MW": float(values.loc["solar_MW [MW]"]),
        "surface_tilt": float(values.loc["surface_tilt [deg]"]),
        "surface_azimuth": float(values.loc["surface_azimuth [deg]"]),
        "DC_AC_ratio": float(values.loc["DC_AC_ratio"]),
        "b_P": float(values.loc["b_P [MW]"]),
        "b_E_h": float(values.loc["b_E_h [h]"]),
        "cost_of_batt_degr": float(
            values.loc["cost_of_battery_P_fluct_in_peak_price_ratio"]
        ),
    }


def _read_input_ts(input_ts_fn):
    """
    Reads a time series CSV file, automatically detecting the delimiter.
    
    Args:
        input_ts_fn (str): Path to the time series CSV.
        
    Returns:
        pd.DataFrame: Sorted DataFrame with a DatetimeIndex.
    """
    sep = None
    engine = "python"
    try:
        # Sniff the delimiter by counting occurrences in the header
        with open(input_ts_fn, "r", encoding="utf-8") as f:
            header_line = f.readline()
        sep_candidates = [";", ",", "\t"]
        counts = {c: header_line.count(c) for c in sep_candidates}
        best_sep = max(counts, key=counts.get)
        if counts[best_sep] > 0:
            sep = best_sep
            engine = "c"
    except Exception:
        pass

    input_ts = pd.read_csv(
        input_ts_fn,
        index_col=0,
        parse_dates=False,
        sep=sep,
        engine=engine,
    )
    
    # Attempt to convert index to datetime
    parsed_index = pd.to_datetime(
        input_ts.index,
        errors="coerce",
        dayfirst=True,
    )
    if parsed_index.notna().all():
        input_ts.index = parsed_index
        
    if not isinstance(input_ts.index, pd.DatetimeIndex):
        raise ValueError(
            f"Input time series index is not datetime: {input_ts_fn}"
        )
    return input_ts.sort_index()


def _normalize_year_8760(year_df, year):
    """
    Ensures a year dataframe contains exactly 8760 hours by removing leap days.
    
    Args:
        year_df (pd.DataFrame): Data for a specific year.
        year (int): The year being processed.
        
    Returns:
        pd.DataFrame: Normalized dataframe.
    """
    year_df = year_df.copy()
    if year_df.empty:
        raise ValueError(f"No rows found for year {year}")

    # Drop February 29th if it exists
    leap_mask = (year_df.index.month == 2) & (year_df.index.day == 29)
    if leap_mask.any():
        year_df = year_df.loc[~leap_mask]

    expected_hours = 365 * 24
    if len(year_df) != expected_hours:
        raise ValueError(
            f"Year {year} has {len(year_df)} rows after leap-day handling; "
            f"expected {expected_hours}."
        )

    return year_df


def _repeat_year_to_lifetime(year_df, base_year, lifetime_years):
    """
    Projects a single year's data over a lifetime by repeating it.
    
    Args:
        year_df (pd.DataFrame): Source weather/price year.
        base_year (int): Starting year for the new index.
        lifetime_years (int): Duration to repeat.
        
    Returns:
        pd.DataFrame: Expanded lifetime dataframe.
    """
    year_no_index = year_df.reset_index(drop=True)
    repeated_df = pd.concat(
        [year_no_index] * lifetime_years,
        axis=0,
        ignore_index=True,
    )

    repeated_index = pd.date_range(
        start=f"{base_year}-01-01 00:00:00",
        periods=len(repeated_df),
        freq="h",
    )
    repeated_df.index = repeated_index
    repeated_df.index.name = year_df.index.name
    return repeated_df


def _build_design_vector(design):
    """Converts design dictionary to a list in the order expected by hydesign."""
    return [
        design["clearance"],
        design["sp"],
        design["p_rated"],
        design["Nwt"],
        design["wind_MW_per_km2"],
        design["solar_MW"],
        design["surface_tilt"],
        design["surface_azimuth"],
        design["DC_AC_ratio"],
        design["b_P"],
        design["b_E_h"],
        design["cost_of_batt_degr"],
    ]


def _get_prob_var(prob, var_names):
    """
    Safely retrieves a variable's value from the OpenMDAO problem object.
    
    Args:
        prob: The hydesign problem object.
        var_names (str or list): Names of variables to search for.
    """
    if isinstance(var_names, str):
        var_names = [var_names]

    for var_name in var_names:
        try:
            values = np.asarray(prob.get_val(var_name)).reshape(-1)
        except Exception:
            try:
                values = np.asarray(prob[var_name]).reshape(-1)
            except Exception:
                continue

        if values.size > 0:
            return values
    return None


def _extract_sp_solar_generation(year_df, solar_capacity_mw):
    """
    Calculates solar production based on 'sp' column in the input timeseries.
    
    Args:
        year_df (pd.DataFrame): Hourly data containing 'sp'.
        solar_capacity_mw (float): Installed solar capacity.
    """
    sp_col = None
    for col in year_df.columns:
        if str(col).strip().lower() == "sp":
            sp_col = col
            break

    if sp_col is None:
        return None

    sp_ts = pd.to_numeric(year_df[sp_col], errors="coerce").fillna(0.0)
    sp_ts = sp_ts.clip(lower=0.0)

    if solar_capacity_mw <= 0.0:
        return {"solar_gwh": 0.0, "solar_cf": np.nan, "solar_source": "SP"}

    solar_gwh = float(sp_ts.sum() * solar_capacity_mw / 1000.0)
    solar_cf = float(sp_ts.mean())
    return {
        "solar_gwh": solar_gwh,
        "solar_cf": solar_cf,
        "solar_source": "SP",
    }


def _extract_mean_annual_generation(hpp, lifetime_years, solar_capacity_mw, year_df):
    """
    Aggregates mean annual energy production (AEP) from model outputs.
    
    Returns:
        dict: Mean annual GWh for wind, solar, and battery.
    """
    wind_t = _get_prob_var(hpp.prob, ["wind_t_rel", "wind_t_ext_deg", "wind_t"])
    solar_t = _get_prob_var(hpp.prob, ["solar_t_rel", "solar_t_ext_deg", "solar_t"])
    b_t = _get_prob_var(hpp.prob, ["b_t_rel", "b_t"])

    if lifetime_years <= 0:
        lifetime_years = 1

    # Wind Calc
    if wind_t is None:
        wind_gwh = np.nan
    else:
        wind_gwh = float(np.nansum(np.nan_to_num(wind_t, nan=0.0)) / 1000.0 / lifetime_years)

    # Solar Calc (Preference to SP column, fallback to model)
    sp_solar = _extract_sp_solar_generation(year_df, solar_capacity_mw)
    if sp_solar is not None:
        solar_gwh = sp_solar["solar_gwh"]
        solar_cf = sp_solar["solar_cf"]
        solar_source = sp_solar["solar_source"]
    elif solar_t is None:
        solar_gwh = solar_cf = np.nan
        solar_source = "model"
    else:
        solar_gwh = float(np.nansum(np.nan_to_num(solar_t, nan=0.0)) / 1000.0 / lifetime_years)
        solar_cf = float((solar_gwh * 1000.0) / (solar_capacity_mw * 365.0 * 24.0)) if solar_capacity_mw > 0 else np.nan
        solar_source = "model"

    # Battery Discharge
    battery_gwh = float(np.sum(np.clip(np.nan_to_num(b_t, nan=0.0), 0.0, None)) / 1000.0 / lifetime_years) if b_t is not None else np.nan

    return {
        "Mean Annual Wind Electricity [GWh]": wind_gwh,
        "Mean Annual Solar Electricity [GWh]": solar_gwh,
        "Mean Annual Battery Discharge [GWh]": battery_gwh,
        "Capacity factor solar [-]": solar_cf,
        "Solar production source": solar_source,
    }


def evaluate_single_year(year, site_name, latitude, longitude, altitude, sim_pars_fn, 
                        design_x, design, yearly_groups, lifetime_years, temp_dir, price_add):
    """
    Evaluates a design for one specific weather year.
    
    Args:
        year (int): The weather year to simulate.
        ... (design/site parameters)
        price_add (float): Fixed price offset applied to all hourly prices.
        
    Returns:
        tuple: (result_dict, hourly_production_df)
    """
    if hpp_model is None or examples_filepath is None or calculate_bankability_metrics is None:
        _init_local_hydesign_imports()
    
    year_df = _normalize_year_8760(yearly_groups.get(year, pd.DataFrame()), year)

    # Apply Price Offset
    if price_add != 0:
        price_cols = [col for col in year_df.columns if 'price' in col.lower()]
        for col in price_cols:
            year_df[col] = year_df[col] + price_add

    # Create temporary CSV for hydesign to read
    year_input_ts_fn = os.path.join(temp_dir, f"input_ts_{site_name}_{year}.csv")
    year_df.to_csv(year_input_ts_fn, sep=";")

    # Initialize and run the model
    hpp = hpp_model(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        num_batteries=5,
        work_dir="./",
        sim_pars_fn=sim_pars_fn,
        input_ts_fn=year_input_ts_fn,
    )

    outs = hpp.evaluate(*design_x)
    eval_df = hpp.evaluation_in_df(design_x, outs)
    
    # Consolidate results
    row = eval_df.iloc[0].to_dict()
    row.update({
        "site": site_name,
        "weather_year": year,
        "lifetime_years": lifetime_years,
        "price_added": price_add,
        "input_rows_per_year": len(year_df),
        "input_rows_lifetime": len(year_df) * lifetime_years,
    })
    row.update(_extract_mean_annual_generation(hpp, lifetime_years, design["solar_MW"], year_df))
    row.update(calculate_bankability_metrics(row))

    # Extract hourly time series for export
    wind_t = _get_prob_var(hpp.prob, "wind_t")
    solar_t = _get_prob_var(hpp.prob, "solar_t")

    hourly_df = None
    if wind_t is not None and solar_t is not None:
        hourly_df = pd.DataFrame({
            "time": year_df.index,
            "wind_t": wind_t[:len(year_df)],
            "solar_t": solar_t[:len(year_df)]
        })

    print(f"Evaluated year {year} (Price offset: +{price_add})")
    return row, hourly_df


def _append_bankability_quantiles(results_df):
    """
    Adds LLCR P50 and P90 quantiles based on the distribution of weather years.
    
    LLCR P90 is reported as the 10th percentile (downside risk).
    """
    metric_cols = {
        "LLCR [-]": ("LLCR P50 [-]", "LLCR P90 [-]"),
    }
    available_metrics = {
        metric: quantiles
        for metric, quantiles in metric_cols.items()
        if metric in results_df.columns
    }
    if not available_metrics:
        return results_df

    results_df = results_df.copy()
    for metric_col, (p50_col, p90_col) in available_metrics.items():
        metric_values = (
            pd.to_numeric(results_df[metric_col], errors="coerce")
            .dropna()
            .to_numpy()
        )
        if metric_values.size == 0:
            continue

        results_df[p50_col] = float(np.nanpercentile(metric_values, 50))
        results_df[p90_col] = float(np.nanpercentile(metric_values, 10))
    return results_df


def evaluate_yearly_lifetime(site_name, latitude, longitude, altitude, sim_pars_fn, 
                             input_ts_fn, design, start_year, end_year, lifetime_years,
                             price_add, save_hourly_csv=True):
    """
    Orchestrates the parallel evaluation of a site across multiple weather years.

    Args:
        save_hourly_csv (bool): If True, saves aggregated hourly wind/solar
            production to CSV. Set to False for large batch workflows where
            hourly export is not required.
    """
    design_x = _build_design_vector(design)
    input_ts = _read_input_ts(input_ts_fn)
    yearly_groups = {year: group for year, group in input_ts.groupby(input_ts.index.year)}

    # Use a temporary directory for intermediary CSVs passed to the hydesign model
    with tempfile.TemporaryDirectory(prefix=f"hpp_eval_{site_name}_") as temp_dir:
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(evaluate_single_year)(
                year, site_name, latitude, longitude, altitude, sim_pars_fn, 
                design_x, design, yearly_groups, lifetime_years, temp_dir, price_add
            )
            for year in range(start_year, end_year + 1)
        )

    rows = [r for r, h in results]
    all_hourly = [h for r, h in results if h is not None]

    # Save aggregated hourly production only when explicitly requested.
    if save_hourly_csv and all_hourly:
        hourly_all = pd.concat(all_hourly, axis=0, ignore_index=True)
        eval_dir = _get_evaluations_dir()
        os.makedirs(eval_dir, exist_ok=True)
        hourly_csv = os.path.join(eval_dir, f"{site_name}_hourly_production_{start_year}_{end_year}.csv")
        hourly_all.to_csv(hourly_csv, index=False)
        print(f"Saved hourly wind/solar production: {hourly_csv}")

    results_df = pd.DataFrame(rows)
    results_df = _append_bankability_quantiles(results_df)
    return results_df


def main():
    """Main entry point for the HPP evaluation CLI."""
    _init_local_hydesign_imports()

    # --- CONFIGURATION ---
    DEFAULT_PRICE_ADD = 30
    # ---------------------

    parser = argparse.ArgumentParser(description="Evaluate site designs across multiple weather years.")
    parser.add_argument("--site", nargs='+', default=["Golfe_du_Lion"], 
                        help="List of site names to process.")
    parser.add_argument("--list-sites", action="store_true", help="Show all available site configs and exit.")
    parser.add_argument("--start-year", type=int, default=1982)
    parser.add_argument("--end-year", type=int, default=1982)
    parser.add_argument("--lifetime-years", type=int, default=1)
    parser.add_argument("--output-csv", default=None, help="Custom output path for results.")
    parser.add_argument("--price-add", type=float, default=DEFAULT_PRICE_ADD, 
                        help=f"Price offset in Eur/MWh (Default: {DEFAULT_PRICE_ADD})")
    args = parser.parse_args()

    site_config_dir = _get_site_config_dir()
    
    if args.list_sites:
        sites = _available_site_configs(site_config_dir)
        print("Available site configs:", *[f"- {s}" for s in sites], sep="\n")
        return

    # Loop through requested sites
    for site_name in args.site:
        try:
            print(f"\nProcessing Site: {site_name}")
            ex_site = _load_site_row(site_name)
            design = _load_site_design(site_name, site_config_dir)

            start = time.time()
            yearly_results_df = evaluate_yearly_lifetime(
                site_name=site_name,
                latitude=ex_site["latitude"],
                longitude=ex_site["longitude"],
                altitude=ex_site["altitude"],
                sim_pars_fn=examples_filepath + ex_site["sim_pars_fn"],
                input_ts_fn=examples_filepath + ex_site["input_ts_fn"],
                design=design,
                start_year=args.start_year,
                end_year=args.end_year,
                lifetime_years=args.lifetime_years,
                price_add=args.price_add
            )
            
            # Determine output filename
            output_csv = args.output_csv
            if output_csv is None:
                os.makedirs(_get_evaluations_dir(), exist_ok=True)
                p_suffix = f"_p{args.price_add}" if args.price_add != 0 else ""
                output_csv = os.path.join(
                    _get_evaluations_dir(), 
                    f"{site_name}_yearly_eval_{args.start_year}_{args.end_year}_life{args.lifetime_years}{p_suffix}.csv"
                )
            elif len(args.site) > 1:
                # Append site name to custom output path if processing multiple sites
                base, ext = os.path.splitext(output_csv)
                output_csv = f"{base}_{site_name}{ext}"

            yearly_results_df.to_csv(output_csv, index=False)
            print(f"Results saved to: {output_csv}")
            print(f"Total time: {(time.time() - start) / 60:.2f} min")
            
        except Exception as e:
            print(f"Error processing site '{site_name}': {e}")


if __name__ == "__main__":
    main()