"""
Hybrid Power Plant (HPP) Optimization Driver.

This script uses Efficient Global Optimization (EGO) to find the optimal design
parameters (wind, solar, battery) for a given site. It supports batch processing
of multiple sites.
"""

import argparse
import os
import sys
from multiprocessing import freeze_support

# Resolve local repository imports before package imports below.
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pandas as pd
from hydesign.assembly.hpp_assembly_sp import hpp_model
from hydesign.Parallel_EGO import EfficientGlobalOptimizationDriver
from hydesign.examples import examples_filepath

# --- Environment Setup ---
# Limit internal multi-threading of linear algebra libraries to prevent 
# contention during parallel HPP optimizations.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- Resource Allocation ---
n_procs = os.cpu_count()
if n_procs > 2:
    n_procs -= 1
    n_doe = int(2 * n_procs)  # Number of initial Design of Experiments points
else:
    n_procs -= 0
    n_doe = int(4 * n_procs)

# --- Directory Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
site_config_dir = os.path.join(script_dir, 'SiteConfig')
os.makedirs(site_config_dir, exist_ok=True)
prepared_ts_dir = os.path.join(site_config_dir, '_prepared_input_ts')
os.makedirs(prepared_ts_dir, exist_ok=True)

# --- Optimization Variable Configuration ---
# 'design': Variable the optimizer can change within 'limits'.
# 'fixed': Variable that remains constant at 'value'.
OPT_VARIABLES = {
    'clearance [m]': {
        'var_type': 'fixed',
        'value': 27
    },
    'sp [W/m2]': {
        'var_type': 'fixed',
        'value': 343,
    },
    'p_rated [MW]': {
        'var_type': 'fixed',
        'value': 15
    },
    'Nwt': {
        'var_type': 'design',
        'limits': [0, 100],
        'types': 'float'
    },
    'wind_MW_per_km2 [MW/km2]': {
        'var_type': 'design',
        'limits': [5, 9],
        'types': 'float'
    },
    'solar_MW [MW]': {
        'var_type': 'fixed',
        'value': 0
    },
    'surface_tilt [deg]': {
        'var_type': 'design',
        'limits': [0, 17],
        'types': 'float'
    },
    'surface_azimuth [deg]': {
        'var_type': 'fixed',
        'value': 180
    },
    'DC_AC_ratio': {
        'var_type': 'design',
        'limits': [1, 1.50],
        'types': 'float'
    },
    'b_P [MW]': {
        'var_type': 'fixed',
        'value':0
    },
    'b_E_h [h]': {
        'var_type': 'fixed',
        'value': 0
    },
    'cost_of_battery_P_fluct_in_peak_price_ratio': {
        'var_type': 'design',
        'limits': [0, 20],
        'types': 'float'
    },
}


def _sanitize_site_filename(site_name):
    """
    Cleans site names to ensure they are valid filesystem names.
    
    Args:
        site_name (str): The raw site name.
        
    Returns:
        str: A filesystem-safe string.
    """
    invalid_chars = '<>:"/\\|?* '
    site_filename = ''.join(
        '_' if char in invalid_chars else char for char in str(site_name)
    ).strip('._')
    return site_filename or 'UnknownSite'


def _build_inputs(ex_site, price_increment=0.0):
    """
    Constructs the input dictionary required by EfficientGlobalOptimizationDriver.
    """
    site_name = ex_site['name']
    site_filename = _sanitize_site_filename(site_name)
    raw_input_ts_fn = examples_filepath + ex_site['input_ts_fn']
    
    # Generate prepared timeseries with potential price offsets
    input_ts_fn = _prepare_input_ts_file(
        raw_input_ts_fn,
        site_filename,
        price_increment=price_increment,
    )

    return {
        'name': site_name,
        'longitude': ex_site['longitude'],
        'latitude': ex_site['latitude'],
        'altitude': ex_site['altitude'],
        'input_ts_fn': input_ts_fn,
        'sim_pars_fn': examples_filepath + ex_site['sim_pars_fn'],
        'opt_var': 'NPV_over_CAPEX',  # The objective function to maximize
        'num_batteries': 10,
        'n_procs': n_procs,
        'n_doe': n_doe,
        'n_clusters': n_procs,
        'n_seed': 0,
        'max_iter': 4,
        'final_design_fn': os.path.join(site_config_dir, f'{site_filename}.csv'),
        'npred': 5e3,
        'tol': 1e-4,
        'min_conv_iter': 2,
        'work_dir': './',
        'hpp_model': hpp_model,
        'variables': OPT_VARIABLES,
    }


def _prepare_input_ts_file(input_ts_fn, site_filename, price_increment=0.0):
    """
    Standardizes column names and applies price increments to the input timeseries.
    """
    weather = pd.read_csv(input_ts_fn, index_col=0, sep=';')

    # Normalize case for radiation columns
    rename_map = {}
    for wanted in ('ghi', 'dni', 'dhi'):
        if wanted not in weather.columns:
            matches = [col for col in weather.columns if str(col).lower() == wanted]
            if matches:
                rename_map[matches[0]] = wanted

    if rename_map:
        weather = weather.rename(columns=rename_map)

    # Convert essential columns to numeric
    cols_to_fix = ('ghi', 'dni', 'dhi', 'WS_150', 'WP_150', 'SP', 'Price')
    for col in cols_to_fix:
        if col in weather.columns:
            weather[col] = pd.to_numeric(weather[col], errors='coerce')

    weather = _increase_price_per_timestamp(weather, price_increment)

    prepared_fn = os.path.join(prepared_ts_dir, f'{site_filename}_input_ts.csv')
    weather.to_csv(prepared_fn)
    return prepared_fn


def _increase_price_per_timestamp(weather_df, increment, price_col='Price'):
    """Adds a fixed increment (offset) to the price column in the timeseries."""
    if price_col not in weather_df.columns:
        return weather_df

    inc = pd.to_numeric(increment, errors='coerce')
    if pd.isna(inc) or float(inc) == 0.0:
        return weather_df

    weather_df = weather_df.copy()
    weather_df[price_col] = (
        pd.to_numeric(weather_df[price_col], errors='coerce') + float(inc)
    )
    return weather_df


def _run_driver(inputs):
    """Initializes and executes the EGO Driver."""
    egod = EfficientGlobalOptimizationDriver(**inputs)
    egod.run()
    return egod, egod.result


def _run_one_site(ex_site, price_increment=0.0):
    """
    Handles the execution logic for a single site.
    """
    inputs = _build_inputs(ex_site, price_increment=price_increment)
    print(f"\nRunning optimization for site: {inputs['name']}")
    
    if price_increment != 0.0:
        print(f"Applying price increment: {price_increment}")
    egod, result = _run_driver(inputs)

    summary = {
        'name': inputs['name'],
        'site_config_file': inputs['final_design_fn'],
    }
    
    # Extract key metrics from the optimization result
    metrics = ('NPV_over_CAPEX', 'NPV [MEuro]', 'LCOE [Euro/MWh]', 'wind [MW]', 'solar [MW]')
    for col in metrics:
        if col in result.columns:
            summary[col] = pd.to_numeric(result.iloc[0][col], errors='coerce')

    summary['status'] = 'ok'
    return summary


def main():
    """Main entry point for optimization CLI."""
    parser = argparse.ArgumentParser(description='Run HPP site optimization.')
    parser.add_argument('--site', default='Sud_Atalantique_Wind_HiFiEMS', 
                        help="Site name, row index, or 'all'.")
    parser.add_argument('--list-sites', action='store_true', help='List available sites and exit')
    parser.add_argument('--price-increment', type=float, default=0.0, 
                        help='Fixed increment added to hourly prices.') 
    args = parser.parse_args()

    # Load master site list
    examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0, sep=';')

    if args.list_sites:
        print('Available sites:')
        for idx, row in examples_sites.iterrows():
            print(f"- {idx}: {row['name']}")
        return

    # Filter sites based on CLI argument
    if str(args.site).lower() == 'all':
        selected_sites = [row for _, row in examples_sites.iterrows()]
    else:
        selected_by_name = examples_sites.loc[examples_sites['name'] == args.site]
        if not selected_by_name.empty:
            selected_sites = [selected_by_name.iloc[0]]
        elif str(args.site).isdigit() and int(args.site) in examples_sites.index:
            selected_sites = [examples_sites.loc[int(args.site)]]
        else:
            raise ValueError(f"Unknown site '{args.site}'. Use --list-sites for options.")

    summaries = []
    for ex_site in selected_sites:
        try:
            summaries.append(_run_one_site(ex_site, price_increment=args.price_increment))
        except Exception as exc:
            site_name = str(ex_site['name'])
            print(f"Site failed: {site_name}. Reason: {exc}")
            summaries.append({
                'name': site_name,
                'status': 'failed',
                'error': str(exc),
            })

    # Save master summary of all runs
    summary_df = pd.DataFrame(summaries)
    summary_fn = os.path.join(site_config_dir, 'all_sites_optimization_summary.csv')
    summary_df.to_csv(summary_fn, index=False)
    print(f"\nSaved optimization summary: {summary_fn}")


if __name__ == '__main__':
    # freeze_support is required for multi-processing on Windows
    freeze_support()
    main()