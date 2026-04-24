"""
Evaluation of Hybrid Power Plants (HPPs) using HiFiEMS-specific configuration files.
FIXED: Price adjustment now correctly modifies the market file and updates sim_pars.
"""

import argparse
import os
import sys
import tempfile
import yaml
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# --- Helper Functions ---

def _get_site_row(site_name, examples_filepath):
    examples_sites = pd.read_csv(os.path.join(examples_filepath, "examples_sites.csv"), index_col=0, sep=";")
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

def _get_prob_var(prob, var_names):
    if isinstance(var_names, str): var_names = [var_names]
    for var_name in var_names:
        try:
            return np.asarray(prob.get_val(var_name)).reshape(-1)
        except:
            try: return np.asarray(prob[var_name]).reshape(-1)
            except: continue
    return None

def _read_input_ts(input_ts_fn):
    input_ts = pd.read_csv(input_ts_fn, index_col=0, parse_dates=False, sep=None, engine="python")
    input_ts.index = pd.to_datetime(input_ts.index, errors="coerce", dayfirst=True)
    if not isinstance(input_ts.index, pd.DatetimeIndex):
        raise ValueError(f"Input time series index is not datetime: {input_ts_fn}")
    return input_ts.sort_index()

# --- Worker Function (Isolated) ---

def evaluate_single_year(year, parent_temp_dir, site_name, base_site_name, sim_pars_fn, 
                        year_df, design, lifetime_years, price_add, save_hourly, ex_site):
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from hydesign.assembly.hpp_assembly_hifi import hpp_model
        
        # 1. Isolated Workspace
        year_temp_dir = os.path.join(parent_temp_dir, f"year_{year}")
        os.makedirs(year_temp_dir, exist_ok=True)
        
        # 2. Load and Prepare Config
        with open(sim_pars_fn, "r") as f:
            sim_pars = yaml.safe_load(f)
        
        sim_pars['out_dir'] = os.path.abspath(year_temp_dir)

        # 3. Resolve Data Paths
        hifiems_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(sim_pars_fn)), "HiFiEMS_inputs"))
        suffix_map = {
            "NordsoenMidt": "DK", "Golfe_du_Lion": "FRs", "Sud_Atlantique": "FRw", 
            "Sud_Atlantique_Solar": "FRw", "Sud_Atlantique_Wind": "FRw",
            "Thetys": "NLda", "Thetys_Solar": "NLda", "Thetys_Wind": "NLda", 
            "SicilySouth": "IT", "Vestavind": "NO"
        }
        suffix = suffix_map.get(base_site_name, "")
        
        sim_pars["wind_fn"] = os.path.join(hifiems_dir, f"Power/Winddata{year}_{suffix}.csv")
        sim_pars["solar_fn"] = os.path.join(hifiems_dir, f"Power/Solardata{year}_{suffix}.csv")
        market_fn_orig = os.path.join(hifiems_dir, f"Market/Market{year}_{suffix}.csv")

        # 4. CRITICAL: Handle Price Adjustment with "Surgical" precision
        if price_add != 0:
            market_fn_adj = os.path.join(year_temp_dir, f"Market{year}_adj.csv")
            
            with open(market_fn_orig, 'r') as f:
                lines = f.readlines()
            
            # Detect delimiter (market files use comma, not semicolon)
            header_line = lines[0].strip()
            delimiter = ',' if ',' in header_line else ';'
            
            # Identify which columns to adjust from the header
            # Target: cleared prices (SM_cleared, BM_Up_cleared, BM_Down_cleared, reg_cleared)
            # and forecast columns (SM_forecast, reg_forecast)
            header = header_line.split(delimiter)
            target_indices = [i for i, col in enumerate(header) 
                             if 'cleared' in col.lower() or 'forecast' in col.lower()]
            
            new_lines = [lines[0]] # Keep original header line
            for line in lines[1:]:
                parts = line.strip().split(delimiter)
                for idx in target_indices:
                    try:
                        # Add the price adjustment to the numeric value
                        val = float(parts[idx])
                        parts[idx] = str(val + price_add)
                    except (ValueError, IndexError):
                        continue
                new_lines.append(delimiter.join(parts) + '\n')
            
            with open(market_fn_adj, 'w') as f:
                f.writelines(new_lines)
            
            sim_pars["market_fn"] = market_fn_adj
            print(f"Worker {year}: Surgically adjusted {len(target_indices)} price columns in Market file with delimiter '{delimiter}'.", file=sys.stderr)
        else:
            sim_pars["market_fn"] = market_fn_orig

        # 5. Save Worker-Specific YAML
        temp_yaml_fn = os.path.join(year_temp_dir, f"hpp_pars_{year}.yml")
        with open(temp_yaml_fn, "w") as f_yaml:
            yaml.dump(sim_pars, f_yaml)

        # 6. Prepare Input TS (Saved for model initialization)
        year_input_ts_fn = os.path.join(year_temp_dir, f"input_ts_{site_name}_{year}.csv")
        year_df.to_csv(year_input_ts_fn, sep=";")

        # 7. Model Evaluation
        hpp = hpp_model(
            latitude=ex_site["latitude"], longitude=ex_site["longitude"], altitude=ex_site["altitude"],
            num_batteries=5, work_dir=year_temp_dir, sim_pars_fn=temp_yaml_fn, input_ts_fn=year_input_ts_fn,
        )
        
        wind_MW = design["Nwt"] * design["p_rated"]
        outs = hpp.evaluate(wind_MW, design["solar_MW"], design["b_P"], design["b_E_h"], design["wind_MW_per_km2"])
        eval_df = hpp.evaluation_in_df(None, outs)
        
        row = eval_df.iloc[0].to_dict()
        row.update({"site": site_name, "weather_year": year, "lifetime_years": lifetime_years, "price_added": price_add})
        
        hourly_df = None
        if save_hourly:
            wind_t = _get_prob_var(hpp.prob, "wind_t")
            solar_t = _get_prob_var(hpp.prob, "solar_t")
            b_t = _get_prob_var(hpp.prob, "b_t")
            
            # Extract market prices from EMS model (hourly resolution)
            # SM_price_cleared is the spot market cleared price (hourly)
            sm_price_cleared = _get_prob_var(hpp.prob, "SM_price_cleared")
            bm_up_price = _get_prob_var(hpp.prob, "BM_up_price_cleared")
            bm_dw_price = _get_prob_var(hpp.prob, "BM_dw_price_cleared")
            
            hourly_df = pd.DataFrame({
                "time": year_df.index,
                "weather_year": year,
                "wind_MW": wind_t[:len(year_df)] if wind_t is not None else 0,
                "solar_MW": solar_t[:len(year_df)] if solar_t is not None else 0,
                "battery_MW": b_t[:len(year_df)] if b_t is not None else 0,
                "SM_price_cleared": sm_price_cleared[:len(year_df)] if sm_price_cleared is not None else np.nan,
                "BM_up_price": bm_up_price[:len(year_df)] if bm_up_price is not None else np.nan,
                "BM_dw_price": bm_dw_price[:len(year_df)] if bm_dw_price is not None else np.nan,
            })
        return row, hourly_df
    
    except Exception as e:
        # Using stderr ensures error messages are more likely to appear in the console during parallel runs
        print(f"Error evaluating year {year}: {e}", file=sys.stderr)
        return {"weather_year": year, "error": str(e)}, None

# --- Core Execution Logic ---

def evaluate_hifiems_site(site_name, start_year, end_year, lifetime_years, price_add, output_csv=None, save_hourly=True):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from hydesign.examples import examples_filepath

    ex_site = _get_site_row(site_name, examples_filepath)
    sim_pars_fn = os.path.join(examples_filepath, ex_site["sim_pars_fn"])
    input_ts_fn = os.path.join(examples_filepath, ex_site["input_ts_fn"])
    
    base_site_name = site_name.replace("_HiFiEMS", "")
    site_config_dir = _get_site_config_dir()
    design = _load_site_design(base_site_name, site_config_dir)

    input_ts = _read_input_ts(input_ts_fn)
    yearly_groups = {year: group for year, group in input_ts.groupby(input_ts.index.year)}

    years_to_run = [y for y in range(start_year, end_year + 1) if y in yearly_groups]

    print(f"Starting evaluation for {site_name} (Price Add: {price_add})...")

    with tempfile.TemporaryDirectory(prefix=f"hifiems_eval_{site_name}_") as temp_dir:
        parallel_results = Parallel(n_jobs=-1, verbose=10)(
            delayed(evaluate_single_year)(
                year, temp_dir, site_name, base_site_name, sim_pars_fn, 
                yearly_groups[year], design, lifetime_years, price_add, save_hourly, ex_site
            )
            for year in years_to_run
        )

    summary_rows = [res[0] for res in parallel_results if res[0] is not None]
    summary_df = pd.DataFrame(summary_rows)
    
    if output_csv:
        summary_df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

    if save_hourly:
        hourly_list = [res[1] for res in parallel_results if res[1] is not None]
        if hourly_list:
            master_hourly = pd.concat(hourly_list)
            hourly_path = output_csv.replace(".csv", "_hourly.csv")
            master_hourly.to_csv(hourly_path, index=False)
            print(f"Hourly production saved to {hourly_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", default="Thetys_Solar_HiFiEMS", help="Comma-separated sites")
    parser.add_argument("--start-year", type=int, default=1982)
    parser.add_argument("--end-year", type=int, default=1992)
    parser.add_argument("--lifetime-years", type=int, default=25)
    parser.add_argument("--price-add", type=float, default=42.0)
    args = parser.parse_args()

    site_names = [s.strip() for s in args.sites.split(",") if s.strip()]
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Evaluations", "HiFiEMS", "New")
    os.makedirs(output_dir, exist_ok=True)

    for site in site_names:
        csv_path = os.path.join(output_dir, f"{site}_eval_{args.start_year}_{args.end_year}_p{args.price_add}.csv")
        evaluate_hifiems_site(site, args.start_year, args.end_year, args.lifetime_years, args.price_add, csv_path)

if __name__ == "__main__":
    main()