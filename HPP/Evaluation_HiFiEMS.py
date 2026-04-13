"""
Evaluation of Hybrid Power Plants (HPPs) using HiFiEMS-specific configuration files.
"""

import argparse
import os
import sys
import tempfile
import yaml
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def _init_local_hydesign_imports():
    global hpp_model
    global examples_filepath
    global calculate_bankability_metrics

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

def _get_site_row(site_name):
    examples_sites = pd.read_csv(
        f"{examples_filepath}examples_sites.csv", index_col=0, sep=";")
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    if ex_site.empty:
        raise ValueError(f"Site '{site_name}' not found in examples_sites.csv.")
    return ex_site.iloc[0]

def _get_site_config_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "SiteConfig")

def _load_site_design(site_name, site_config_dir):
    site_config_path = os.path.join(site_config_dir, f"{site_name}.csv")
    if not os.path.isfile(site_config_path):
        raise FileNotFoundError(f"Site config file not found: {site_config_path}")
    
    site_config = pd.read_csv(site_config_path, index_col=0, header=None, sep=",")
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
        "cost_of_batt_degr": float(values.loc["cost_of_battery_P_fluct_in_peak_price_ratio"]),
    }

def _build_design_vector(design):
    """Converts the design dictionary into a list/array for hpp.evaluate()"""
    return [
        design["clearance"], design["sp"], design["p_rated"], design["Nwt"],
        design["wind_MW_per_km2"], design["solar_MW"], design["surface_tilt"],
        design["surface_azimuth"], design["DC_AC_ratio"], design["b_P"],
        design["b_E_h"], design["cost_of_batt_degr"]
    ]

def _read_input_ts(input_ts_fn):
    sep = None
    engine = "python"
    try:
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

    input_ts = pd.read_csv(input_ts_fn, index_col=0, parse_dates=False, sep=sep, engine=engine)
    input_ts.index = pd.to_datetime(input_ts.index, errors="coerce", dayfirst=True)
    
    if not isinstance(input_ts.index, pd.DatetimeIndex):
        raise ValueError(f"Input time series index is not datetime: {input_ts_fn}")
    return input_ts.sort_index()

def evaluate_hifiems_site(site_name, start_year, end_year, lifetime_years, price_add, output_csv=None):
    _init_local_hydesign_imports()
    
    # Load Site Metadata
    ex_site = _get_site_row(site_name)
    sim_pars_fn = examples_filepath + ex_site["sim_pars_fn"]
    input_ts_fn = examples_filepath + ex_site["input_ts_fn"]
    
    # Load Site Design Configuration
    base_site_name = site_name.replace("_HiFiEMS", "")
    site_config_dir = _get_site_config_dir()
    design = _load_site_design(base_site_name, site_config_dir)
    design_x = _build_design_vector(design)

    # Read and split input time series
    input_ts = _read_input_ts(input_ts_fn)
    yearly_groups = {year: group for year, group in input_ts.groupby(input_ts.index.year)}

    def evaluate_single_year(year, temp_dir):
        year_df = yearly_groups.get(year, pd.DataFrame()).copy()
        if year_df.empty:
            raise ValueError(f"No data for year {year}")

        # Apply Price Offset
        if price_add != 0:
            price_cols = [col for col in year_df.columns if 'price' in col.lower()]
            for col in price_cols:
                year_df[col] = year_df[col] + price_add

        # Write temp CSV for this specific year
        year_input_ts_fn = os.path.join(temp_dir, f"input_ts_{site_name}_{year}.csv")
        year_df.to_csv(year_input_ts_fn, sep=";")

        # --- Dynamically update YAML for this year ---
        with open(sim_pars_fn, "r") as f:
            sim_pars = yaml.safe_load(f)

        # Set correct file paths for this year, using hardcoded suffix per site
        hifiems_dir = os.path.join(os.path.dirname(os.path.dirname(sim_pars_fn)), "HiFiEMS_inputs")
        site_suffix_map = {
            "NordsoenMidt": "DK",
            "Golfe_du_Lion": "FRs",
            "Sud_Atlantique": "FRw",
            "Thetys": "NL",
            "SicilySouth": "IT",
            "Vestavind": "NO",
        }
        # Remove _HiFiEMS if present for lookup
        base_site = site_name.replace("_HiFiEMS", "")
        suffix = site_suffix_map.get(base_site, "")
        wind_fn = f"Power/Winddata{year}{'_' + suffix if suffix else ''}.csv"
        solar_fn = f"Power/Solardata{year}{'_' + suffix if suffix else ''}.csv"
        market_fn = f"Market/Market{year}{'_' + suffix if suffix else ''}.csv"
        wind_path = os.path.join(hifiems_dir, wind_fn)
        solar_path = os.path.join(hifiems_dir, solar_fn)
        market_path = os.path.join(hifiems_dir, market_fn)
        sim_pars["wind_fn"] = wind_path
        sim_pars["solar_fn"] = solar_path
        sim_pars["market_fn"] = market_path
        # Removed print of wind, solar, market file paths

        # Write temp YAML for this year
        temp_yaml_fn = os.path.join(temp_dir, f"hpp_pars_{site_name}_{year}.yml")
        with open(temp_yaml_fn, "w") as f_yaml:
            yaml.dump(sim_pars, f_yaml)
        # Removed print of YAML file contents

        # Initialize and evaluate model
        hpp = hpp_model(
            latitude=ex_site["latitude"],
            longitude=ex_site["longitude"],
            altitude=ex_site["altitude"],
            num_batteries=5,
            work_dir=temp_dir,
            sim_pars_fn=temp_yaml_fn,
            input_ts_fn=year_input_ts_fn,
        )

        outs = hpp.evaluate(*design_x)
        eval_df = hpp.evaluation_in_df(None, outs)

        row = eval_df.iloc[0].to_dict()
        row.update({
            "site": site_name,
            "weather_year": year,
            "lifetime_years": lifetime_years,
            "price_added": price_add,
            "input_rows_per_year": len(year_df),
        })

        # Add bankability metrics
        bank_metrics = calculate_bankability_metrics(row)
        row.update(bank_metrics)
        return row

    # Parallel Execution
    with tempfile.TemporaryDirectory(prefix=f"hifiems_eval_{site_name}_") as temp_dir:
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(evaluate_single_year)(year, temp_dir)
            for year in range(start_year, end_year + 1)
        )
    
    # Save Results
    results_df = pd.DataFrame(results)
    if output_csv is None:
        output_csv = f"{site_name}_HiFiEMS_eval_{start_year}_{end_year}_life{lifetime_years}.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate HPP site(s) using HiFiEMS config.")
    parser.add_argument("--sites", default="Sud_Atlantique_HiFiEMS", help="Comma-separated list of site names, or 'ALL' for all HiFiEMS sites.")
    parser.add_argument("--start-year", type=int, default=1982)
    parser.add_argument("--end-year", type=int, default=2015)
    parser.add_argument("--lifetime-years", type=int, default=25)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--price-add", type=float, default=30)
    args = parser.parse_args()

    # Load all HiFiEMS site names from examples_sites.csv if needed
    if args.sites.strip().upper() == "ALL":
        # Import examples_filepath
        _init_local_hydesign_imports()
        sites_df = pd.read_csv(f"{examples_filepath}examples_sites.csv", sep=";")
        site_names = [n for n in sites_df['name'] if n.endswith('_HiFiEMS')]
    else:
        site_names = [s.strip() for s in args.sites.split(",") if s.strip()]

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Evaluations", "HiFiEMS")
    os.makedirs(output_dir, exist_ok=True)
    for site in site_names:
        print(f"\n=== Running evaluation for site: {site} ===")
        output_csv = args.output_csv
        if output_csv is None:
            output_csv = os.path.join(output_dir, f"{site}_HiFiEMS_eval_{args.start_year}_{args.end_year}_life{args.lifetime_years}.csv")
        else:
            # If user provides a filename, save it in the output_dir unless it's an absolute path
            if not os.path.isabs(output_csv):
                output_csv = os.path.join(output_dir, output_csv)
        evaluate_hifiems_site(
            site_name=site,
            start_year=args.start_year,
            end_year=args.end_year,
            lifetime_years=args.lifetime_years,
            price_add=args.price_add,
            output_csv=output_csv,
        )

if __name__ == "__main__":
    main()