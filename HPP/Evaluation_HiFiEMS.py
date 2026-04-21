"""
Evaluation of Hybrid Power Plants (HPPs) using HiFiEMS-specific configuration files.
MODIFIED: Now includes hourly production export.
"""

import argparse
import os
import sys
import tempfile
import yaml
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Global placeholders for dynamic imports
hpp_model = None
examples_filepath = None
calculate_bankability_metrics = None

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
    from hydesign.assembly.hpp_assembly_hifi import hpp_model as local_hpp_model
    from hydesign.examples import examples_filepath as local_examples_filepath

    calculate_bankability_metrics = local_calculate_bankability_metrics
    hpp_model = local_hpp_model
    examples_filepath = local_examples_filepath

def _get_site_row(site_name):
    examples_sites = pd.read_csv(f"{examples_filepath}examples_sites.csv", index_col=0, sep=";")
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
    return [
        design["clearance"], design["sp"], design["p_rated"], design["Nwt"],
        design["wind_MW_per_km2"], design["solar_MW"], design["surface_tilt"],
        design["surface_azimuth"], design["DC_AC_ratio"], design["b_P"],
        design["b_E_h"], design["cost_of_batt_degr"]
    ]

def _read_input_ts(input_ts_fn):
    input_ts = pd.read_csv(input_ts_fn, index_col=0, parse_dates=False, sep=None, engine="python")
    input_ts.index = pd.to_datetime(input_ts.index, errors="coerce", dayfirst=True)
    if not isinstance(input_ts.index, pd.DatetimeIndex):
        raise ValueError(f"Input time series index is not datetime: {input_ts_fn}")
    return input_ts.sort_index()

def _get_prob_var(prob, var_names):
    """Utility to extract values from OpenMDAO problem."""
    if isinstance(var_names, str): var_names = [var_names]
    for var_name in var_names:
        try:
            return np.asarray(prob.get_val(var_name)).reshape(-1)
        except:
            try: return np.asarray(prob[var_name]).reshape(-1)
            except: continue
    return None

def evaluate_hifiems_site(site_name, start_year, end_year, lifetime_years, price_add, output_csv=None, save_hourly=True):
    _init_local_hydesign_imports()
    
    ex_site = _get_site_row(site_name)
    sim_pars_fn = examples_filepath + ex_site["sim_pars_fn"]
    input_ts_fn = examples_filepath + ex_site["input_ts_fn"]
    
    base_site_name = site_name.replace("_HiFiEMS", "")
    site_config_dir = _get_site_config_dir()
    design = _load_site_design(base_site_name, site_config_dir)
    design_x = _build_design_vector(design)

    input_ts = _read_input_ts(input_ts_fn)
    yearly_groups = {year: group for year, group in input_ts.groupby(input_ts.index.year)}

    def evaluate_single_year(year, temp_dir):
        year_df = yearly_groups.get(year, pd.DataFrame()).copy()
        if year_df.empty: raise ValueError(f"No data for year {year}")

        if price_add != 0:
            price_cols = [col for col in year_df.columns if 'price' in col.lower()]
            for col in price_cols: year_df[col] = year_df[col] + price_add

        year_input_ts_fn = os.path.join(temp_dir, f"input_ts_{site_name}_{year}.csv")
        year_df.to_csv(year_input_ts_fn, sep=";")

        with open(sim_pars_fn, "r") as f:
            sim_pars = yaml.safe_load(f)

        out_dir = sim_pars.get('out_dir', 'Not specified')
        abs_out_dir = os.path.abspath(out_dir)
        
        # Ensure output directory exists
        os.makedirs(abs_out_dir, exist_ok=True)
        
        # Update sim_pars to use absolute path for out_dir (no trailing separator, os.path.join handles it)
        sim_pars['out_dir'] = abs_out_dir

        hifiems_dir = os.path.join(os.path.dirname(os.path.dirname(sim_pars_fn)), "HiFiEMS_inputs")
        site_suffix_map = {
            "NordsoenMidt": "DK", "Golfe_du_Lion": "FRs", "Sud_Atlantique": "FRw",
            "Thetys": "NL", "SicilySouth": "IT", "Vestavind": "NO",
        }
        suffix = site_suffix_map.get(base_site_name, "")
        
        # Use absolute paths to avoid working directory issues
        hifiems_dir_abs = os.path.abspath(hifiems_dir)
        sim_pars["wind_fn"] = os.path.join(hifiems_dir_abs, f"Power/Winddata{year}_{suffix}.csv")
        sim_pars["solar_fn"] = os.path.join(hifiems_dir_abs, f"Power/Solardata{year}_{suffix}.csv")
        sim_pars["market_fn"] = os.path.join(hifiems_dir_abs, f"Market/Market{year}_{suffix}.csv")

        temp_yaml_fn = os.path.join(temp_dir, f"hpp_pars_{site_name}_{year}.yml")
        with open(temp_yaml_fn, "w") as f_yaml: yaml.dump(sim_pars, f_yaml)

        hpp = hpp_model(
            latitude=ex_site["latitude"], longitude=ex_site["longitude"], altitude=ex_site["altitude"],
            num_batteries=5, work_dir=temp_dir, sim_pars_fn=temp_yaml_fn, input_ts_fn=year_input_ts_fn,
        )

        # Calculate wind_MW from turbine parameters (Nwt * p_rated)
        wind_MW = design["Nwt"] * design["p_rated"]
        
        # HiFiEMS evaluate() takes: wind_MW, solar_MW, b_P, b_E_h, wind_MW_per_km2
        # Pass original values - if b_P is 0, EMS should handle it (no battery optimization)
        try:
            outs = hpp.evaluate(wind_MW, design["solar_MW"], design["b_P"], design["b_E_h"], design["wind_MW_per_km2"])
            eval_df = hpp.evaluation_in_df(None, outs)
            row = eval_df.iloc[0].to_dict()
            row.update({"site": site_name, "weather_year": year, "lifetime_years": lifetime_years, "price_added": price_add})
            row.update(calculate_bankability_metrics(row))
        except Exception as e:
            print(f"ERROR: Year {year} evaluation failed: {type(e).__name__}: {str(e)[:200]}")
            # Return a minimal row with NaN values for failed year
            row = {
                "site": site_name, "weather_year": year, "lifetime_years": lifetime_years, 
                "price_added": price_add, "NPV [MEuro]": np.nan, "IRR": np.nan,
                "LCOE [Euro/MWh]": np.nan, "Mean Annual Electricity Sold [GWh]": np.nan
            }
            hourly_df = None
            return row, hourly_df

        # --- COPY EMS OUTPUT FILES BEFORE TEMP DIR IS DELETED ---
        out_dir = os.path.abspath(sim_pars.get('out_dir', './test/'))
        if os.path.exists(out_dir):
            import shutil
            ems_files = ['act_signal.csv', 'curtailment.csv', 'Degradation.csv', 
                        'energy_imbalance.csv', 'revenue.csv', 'schedule.csv', 'slope.csv', 'SoC.csv']
            for fname in ems_files:
                temp_file = os.path.join(temp_dir, fname)
                out_file = os.path.join(out_dir, fname)
                if os.path.exists(temp_file):
                    try:
                        shutil.copy2(temp_file, out_file)
                    except Exception:
                        pass
        
        # --- Extraction of Hourly Data ---
        hourly_df = None
        if save_hourly:
            wind_t = _get_prob_var(hpp.prob, "wind_t")
            solar_t = _get_prob_var(hpp.prob, "solar_t")
            b_t = _get_prob_var(hpp.prob, "b_t")
            
            hourly_df = pd.DataFrame({
                "time": year_df.index,
                "weather_year": year,
                "wind_MW": wind_t[:len(year_df)] if wind_t is not None else 0,
                "solar_MW": solar_t[:len(year_df)] if solar_t is not None else 0,
                "battery_MW": b_t[:len(year_df)] if b_t is not None else 0
            })
        
        return row, hourly_df

    with tempfile.TemporaryDirectory(prefix=f"hifiems_eval_{site_name}_") as temp_dir:
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(evaluate_single_year)(year, temp_dir)
            for year in range(start_year, end_year + 1)
        )
    
    # Save Annual Metrics
    rows = [r for r, h in results]
    results_df = pd.DataFrame(rows)
    if output_csv is None:
        output_csv = f"{site_name}_eval_{start_year}_{end_year}.csv"
    results_df.to_csv(output_csv, index=False)

    # Save Aggregated Hourly Data
    if save_hourly:
        hourly_list = [h for r, h in results if h is not None]
        if hourly_list:
            hourly_all = pd.concat(hourly_list, axis=0)
            hourly_path = output_csv.replace(".csv", "_hourly.csv")
            hourly_all.to_csv(hourly_path, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", default="Golfe_du_Lion_HiFiEMS", help="Comma-separated list of site names to evaluate (e.g. 'Sud_Atlantique_HiFiEMS,Thetys_HiFiEMS')            ")
    parser.add_argument("--start-year", type=int, default=1982)
    parser.add_argument("--end-year", type=int, default=1992)
    parser.add_argument("--lifetime-years", type=int, default=25)
    parser.add_argument("--price-add", type=float, default=42)
    args = parser.parse_args()

    _init_local_hydesign_imports()
    site_names = [s.strip() for s in args.sites.split(",") if s.strip()]
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Evaluations", "HiFiEMS")
    os.makedirs(output_dir, exist_ok=True)

    for site in site_names:
        csv_path = os.path.join(output_dir, f"{site}_eval_{args.start_year}_{args.end_year}.csv")
        evaluate_hifiems_site(site, args.start_year, args.end_year, args.lifetime_years, args.price_add, csv_path)

if __name__ == "__main__":
    main()