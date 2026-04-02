import argparse
import os
import sys
from multiprocessing import freeze_support

# Ensure imports resolve to the local repository version when running this file directly.
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from hydesign.assembly.hpp_assembly_sp import hpp_model
from hydesign.Parallel_EGO import EfficientGlobalOptimizationDriver
from hydesign.examples import examples_filepath
import pandas as pd

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

n_procs = os.cpu_count()
if n_procs > 2:
    n_procs -= 1
    n_doe = int(2 * n_procs)
else:
    n_procs -= 0
    n_doe = int(4 * n_procs)

script_dir = os.path.dirname(os.path.abspath(__file__))
site_config_dir = os.path.join(script_dir, 'SiteConfig')
os.makedirs(site_config_dir, exist_ok=True)
prepared_ts_dir = os.path.join(site_config_dir, '_prepared_input_ts')
os.makedirs(prepared_ts_dir, exist_ok=True)

OPT_VARIABLES = {
    'clearance [m]':
        #{'var_type':'design',
        #  'limits':[20],
        #  'types':'int'
        #  },
         {'var_type':'fixed',
           'value': 27
           },
     'sp [W/m2]':
       # {'var_type':'design',
       #  'limits':[260, 343],
       # 'types':'int'
       #   },
         {'var_type':'fixed',
          'value': 343,
          },
    'p_rated [MW]':
        #{'var_type':'design',
        #  'limits':[7, 15],
        #  'types':'int'
        #  },
         {'var_type':'fixed',
           'value': 15
           },
    'Nwt':
        {'var_type':'design',
          'limits':[0, 100],
          'types':'int'
          },
        # {'var_type':'fixed',
        #   'value': 200
        #   },
    'wind_MW_per_km2 [MW/km2]':
        {'var_type':'design',
          'limits':[5, 9],
          'types':'float'
          },
        # {'var_type':'fixed',
        #   'value': 7
        #   },
    'solar_MW [MW]':
       #  {'var_type':'design',
       #    'limits':[0, 1000],
       #    'types':'int'
       #    },
        {'var_type':'fixed',
          'value': 0
          },
    'surface_tilt [deg]':
         #{'var_type':'design',
         #  'limits':[0, 17],
         #  'types':'float'
         #  },
        {'var_type':'fixed',
        'value': 0
          },
    'surface_azimuth [deg]':
         #{'var_type':'design',
         #  'limits':[150, 210],
         #  'types':'float'
         #  },
        {'var_type':'fixed',
        'value': 180
         },
    'DC_AC_ratio':
         {'var_type':'design',
          'limits':[1, 1.50],
          'types':'float'
          },
         #{'var_type':'fixed',
         #'value':1.5,
         #},
    'b_P [MW]':
         {'var_type':'design',
           'limits':[0, 100],
           'types':'int'
           },
        #{'var_type':'fixed',
        #  'value': 50
        #  },
    'b_E_h [h]':
         {'var_type':'design',
           'limits':[1, 8],
           'types':'int'
           },
        #{'var_type':'fixed',
        #  'value': 6
         # },
    'cost_of_battery_P_fluct_in_peak_price_ratio':
         {'var_type':'design',
           'limits':[0, 20],
           'types':'float'
           },
        # {'var_type':'fixed',
        # 'value': 10
        #   },
        }


def _sanitize_site_filename(site_name):
    invalid_chars = '<>:"/\\|?* '
    site_filename = ''.join(
        '_' if char in invalid_chars else char for char in str(site_name)
    ).strip('._')
    return site_filename or 'UnknownSite'


def _extract_solar_cf(result, egod, input_ts_fn=None):
    # Prefer explicit solar profile (SP) from weather inputs when available.
    if input_ts_fn:
        try:
            weather = pd.read_csv(input_ts_fn, index_col=0)
            if 'SP' in weather.columns:
                sp = pd.to_numeric(weather['SP'], errors='coerce').dropna()
                if not sp.empty:
                    return float(sp.mean())
        except Exception:
            pass

    solar_cf_col = None
    for candidate in (
        'Capacity factor solar [-]',
        'Solar_CF',
        'solar_cf',
    ):
        if candidate in result.columns:
            solar_cf_col = candidate
            break

    if solar_cf_col is None:
        for col in result.columns:
            col_l = str(col).lower()
            if 'solar' in col_l and 'capacity factor' in col_l:
                solar_cf_col = col
                break

    if solar_cf_col is not None:
        solar_cf = float(
            pd.to_numeric(result.iloc[0][solar_cf_col], errors='coerce')
        )
        return solar_cf

    try:
        prob = egod.hpp_m.prob
        solar_capacity_mw = float(
            pd.to_numeric(result.iloc[0].get('solar [MW]'), errors='coerce')
        )
        if not pd.notna(solar_capacity_mw):
            solar_capacity_mw = float(
                pd.to_numeric(
                    prob.get_val('solar_MW'),
                    errors='coerce',
                ).reshape(-1)[0]
            )

        if pd.notna(solar_capacity_mw) and solar_capacity_mw > 0:
            solar_t_ext_deg = pd.to_numeric(
                pd.Series(prob.get_val('solar_t_ext_deg').reshape(-1)),
                errors='coerce',
            )
            mean_solar_mw = float(solar_t_ext_deg.mean())
            return mean_solar_mw / solar_capacity_mw
    except Exception:
        return float('nan')

    return float('nan')


def _build_inputs(ex_site, price_increment=0.0):
    site_name = ex_site['name']
    site_filename = _sanitize_site_filename(site_name)
    raw_input_ts_fn = examples_filepath + ex_site['input_ts_fn']
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
        'opt_var': 'NPV_over_CAPEX',
        'num_batteries': 10,
        'n_procs': n_procs,
        'n_doe': n_doe,
        'n_clusters': n_procs,
        'n_seed': 1,
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
    weather = pd.read_csv(input_ts_fn, index_col=0, sep=';')

    rename_map = {}
    for wanted in ('ghi', 'dni', 'dhi'):
        if wanted not in weather.columns:
            matches = [
                col for col in weather.columns if str(col).lower() == wanted
            ]
            if matches:
                rename_map[matches[0]] = wanted

    if rename_map:
        weather = weather.rename(columns=rename_map)

    for col in ('ghi', 'dni', 'dhi', 'WS_150', 'WP_150', 'SP', 'Price'):
        if col in weather.columns:
            weather[col] = pd.to_numeric(weather[col], errors='coerce')

    weather = _increase_price_per_timestamp(
        weather,
        price_increment,
    )

    prepared_fn = os.path.join(prepared_ts_dir, f'{site_filename}_input_ts.csv')
    weather.to_csv(prepared_fn)
    return prepared_fn


def _increase_price_per_timestamp(weather_df, increment, price_col='Price'):
    """Increase the price value at every timestamp by a fixed amount."""
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


def _build_retry_variables():
    # Conservative bounds to improve chance of valid DOE points.
    return {
        'clearance [m]': {'var_type': 'fixed', 'value': 27},
        'sp [W/m2]': {'var_type': 'fixed', 'value': 343},
        'p_rated [MW]': {'var_type': 'fixed', 'value': 15},
        'Nwt': {'var_type': 'design', 'limits': [5, 80], 'types': 'int'},
        'wind_MW_per_km2 [MW/km2]': {
            'var_type': 'design',
            'limits': [5.0, 8.0],
            'types': 'float',
        },
        'solar_MW [MW]': {'var_type': 'design', 'limits': [0, 400], 'types': 'int'},
        'surface_tilt [deg]': {
            'var_type': 'design',
            'limits': [5.0, 17.0],
            'types': 'float',
        },
        'surface_azimuth [deg]': {'var_type': 'fixed', 'value': 180},
        'DC_AC_ratio': {'var_type': 'design', 'limits': [1.0, 1.30], 'types': 'float'},
        'b_P [MW]': {'var_type': 'design', 'limits': [0, 60], 'types': 'int'},
        'b_E_h [h]': {'var_type': 'design', 'limits': [2, 8], 'types': 'int'},
        'cost_of_battery_P_fluct_in_peak_price_ratio': {
            'var_type': 'design',
            'limits': [0.0, 10.0],
            'types': 'float',
        },
    }


def _run_driver(inputs):
    egod = EfficientGlobalOptimizationDriver(**inputs)
    egod.run()
    return egod, egod.result


def _run_one_site(ex_site, price_increment=0.0):
    inputs = _build_inputs(ex_site, price_increment=price_increment)
    print(f"\nRunning optimization for site: {inputs['name']}")
    if price_increment != 0.0:
        print(
            f"Applying temporary price increment per timestamp: "
            f"{price_increment}"
        )
    retry_used = False

    try:
        egod, result = _run_driver(inputs)
    except RuntimeError as exc:
        msg = str(exc)
        if 'All simulations failed during initial DOE' not in msg:
            raise

        print(
            'Initial DOE failed for this site. '
            'Retrying with conservative bounds...'
        )
        retry_used = True
        retry_inputs = dict(inputs)
        retry_inputs['variables'] = _build_retry_variables()
        retry_inputs['n_doe'] = max(8, int(inputs['n_doe'] / 2))
        egod, result = _run_driver(retry_inputs)

    solar_cf = _extract_solar_cf(
        result,
        egod,
        input_ts_fn=inputs.get('input_ts_fn'),
    )
    if pd.notna(solar_cf):
        print(f"Solar capacity factor: {solar_cf:.4f} ({solar_cf * 100:.2f}%)")
    else:
        print('Solar capacity factor: N/A')

    summary = {
        'name': inputs['name'],
        'site_config_file': inputs['final_design_fn'],
    }
    for col in (
        'NPV_over_CAPEX',
        'NPV [MEuro]',
        'LCOE [Euro/MWh]',
        'wind [MW]',
        'solar [MW]',
    ):
        if col in result.columns:
            summary[col] = pd.to_numeric(result.iloc[0][col], errors='coerce')
    summary['solar_cf'] = solar_cf
    summary['status'] = 'ok'
    summary['retry_used'] = retry_used
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Run site optimization for one site or all sites in one run.'
    )
    parser.add_argument(
        '--site',
        default='Sud_Atlantique_Wind',
        help=(
            "Site name from examples_sites.csv, row index, or 'all' "
            'to run every site'
        ),
    )
    parser.add_argument(
        '--list-sites',
        action='store_true',
        help='List available sites and exit',
    )
    parser.add_argument(
        '--price-increment',
        type=float,
        default=0.0,
        help=(
            'Temporary per-run increment added to each Price timestamp '
            'during input preprocessing.'
        ),
    )
    args = parser.parse_args()

    examples_sites = pd.read_csv(
        f'{examples_filepath}examples_sites.csv',
        index_col=0,
        sep=';',
    )

    if args.list_sites:
        print('Available sites:')
        for idx, row in examples_sites.iterrows():
            print(f"- {idx}: {row['name']}")
        return

    if str(args.site).lower() == 'all':
        selected_sites = [row for _, row in examples_sites.iterrows()]
    else:
        selected_by_name = examples_sites.loc[examples_sites['name'] == args.site]
        if not selected_by_name.empty:
            selected_sites = [selected_by_name.iloc[0]]
        elif str(args.site).isdigit() and int(args.site) in examples_sites.index:
            selected_sites = [examples_sites.loc[int(args.site)]]
        else:
            raise ValueError(
                f"Unknown site '{args.site}'. Use --list-sites to inspect options."
            )

    summaries = []
    for ex_site in selected_sites:
        try:
            summaries.append(
                _run_one_site(
                    ex_site,
                    price_increment=args.price_increment,
                )
            )
        except Exception as exc:
            site_name = str(ex_site['name'])
            print(f"Site failed and will be skipped: {site_name}")
            print(f"Reason: {exc}")
            summaries.append(
                {
                    'name': site_name,
                    'status': 'failed',
                    'retry_used': True,
                    'error': str(exc),
                }
            )

    summary_df = pd.DataFrame(summaries)
    summary_fn = os.path.join(site_config_dir, 'all_sites_optimization_summary.csv')
    summary_df.to_csv(summary_fn, index=False)
    print(f"\nSaved optimization summary: {summary_fn}")


if __name__ == '__main__':
    freeze_support()
    main()
