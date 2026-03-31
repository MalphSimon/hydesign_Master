"""Evaluation of HPPs using site-specific configuration files.

Usage examples:
    python HPP/Evaluation.py --site NordsoenMidt
    python HPP/Evaluation.py --list-sites
"""

import argparse
import os
import sys
import time
import tempfile

import numpy as np
import pandas as pd
from Bankability import calculate_bankability_metrics
from joblib import Parallel, delayed

hpp_model = None
examples_filepath = None


def _init_local_hydesign_imports():
    global hpp_model
    global examples_filepath

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from hydesign.assembly.hpp_assembly import hpp_model as local_hpp_model
    from hydesign.examples import examples_filepath as local_examples_filepath

    hpp_model = local_hpp_model
    examples_filepath = local_examples_filepath


def _get_site_config_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "SiteConfig")


def _get_evaluations_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "Evaluations")


def _available_site_configs(site_config_dir):
    if not os.path.isdir(site_config_dir):
        return []
    return sorted(
        os.path.splitext(file_name)[0]
        for file_name in os.listdir(site_config_dir)
        if file_name.lower().endswith(".csv")
    )


def _load_site_row(site_name):
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
        # Keep previous robust behavior if delimiter sniffing fails.
        pass

    input_ts = pd.read_csv(
        input_ts_fn,
        index_col=0,
        parse_dates=False,
        sep=sep,
        engine=engine,
    )
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


def _get_single_year_8760(input_ts, year):
    year_df = input_ts.loc[input_ts.index.year == year]
    return _normalize_year_8760(year_df, year)


def _normalize_year_8760(year_df, year):
    year_df = year_df.copy()
    if year_df.empty:
        raise ValueError(f"No rows found for year {year}")

    # hyDesign expects years of 365 days. Remove leap-day rows if present.
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
        return {
            "solar_gwh": 0.0,
            "solar_cf": np.nan,
            "solar_source": "SP",
        }

    solar_gwh = float(sp_ts.sum() * solar_capacity_mw / 1000.0)
    solar_cf = float(sp_ts.mean())
    return {
        "solar_gwh": solar_gwh,
        "solar_cf": solar_cf,
        "solar_source": "SP",
    }


def _extract_mean_annual_generation(
    hpp,
    lifetime_years,
    solar_capacity_mw,
    year_df,
):
    wind_t = _get_prob_var(
        hpp.prob,
        ["wind_t_rel", "wind_t_ext_deg", "wind_t"],
    )
    solar_t = _get_prob_var(
        hpp.prob,
        ["solar_t_rel", "solar_t_ext_deg", "solar_t"],
    )
    b_t = _get_prob_var(hpp.prob, ["b_t_rel", "b_t"])

    if lifetime_years <= 0:
        lifetime_years = 1

    if wind_t is None:
        wind_gwh = np.nan
    else:
        wind_gwh = float(
            np.nansum(np.nan_to_num(wind_t, nan=0.0))
            / 1000.0
            / lifetime_years
        )

    sp_solar = _extract_sp_solar_generation(year_df, solar_capacity_mw)
    if sp_solar is not None:
        solar_gwh = sp_solar["solar_gwh"]
        solar_cf = sp_solar["solar_cf"]
        solar_source = sp_solar["solar_source"]
    elif solar_t is None:
        solar_gwh = np.nan
        solar_cf = np.nan
        solar_source = "model"
    else:
        solar_gwh = float(
            np.nansum(np.nan_to_num(solar_t, nan=0.0))
            / 1000.0
            / lifetime_years
        )
        if solar_capacity_mw > 0.0:
            solar_cf = float(
                (solar_gwh * 1000.0) / (solar_capacity_mw * 365.0 * 24.0)
            )
        else:
            solar_cf = np.nan
        solar_source = "model"

    if b_t is None:
        battery_gwh = np.nan
    else:
        b_t = np.nan_to_num(b_t, nan=0.0)
        battery_gwh = float(
            np.sum(np.clip(b_t, 0.0, None))
            / 1000.0
            / lifetime_years
        )

    return {
        "Mean Annual Wind Electricity [GWh]": wind_gwh,
        "Mean Annual Solar Electricity [GWh]": solar_gwh,
        "Mean Annual Battery Discharge [GWh]": battery_gwh,
        "Capacity factor solar [-]": solar_cf,
        "Solar production source": solar_source,
    }


def evaluate_yearly_lifetime(
    site_name,
    latitude,
    longitude,
    altitude,
    sim_pars_fn,
    input_ts_fn,
    design,
    start_year,
    end_year,
    lifetime_years,
):
    input_ts = _read_input_ts(input_ts_fn)
    design_x = _build_design_vector(design)
    yearly_groups = {
        int(year): year_frame
        for year, year_frame in input_ts.groupby(input_ts.index.year)
    }

    def evaluate_single_year(year, site_name, latitude, longitude, altitude, sim_pars_fn, design_x, design, yearly_groups, lifetime_years, temp_dir):
        year_df = _normalize_year_8760(
            yearly_groups.get(year, pd.DataFrame()),
            year,
        )
        lifetime_df = _repeat_year_to_lifetime(
            year_df, year, lifetime_years
        )

        year_input_ts_fn = os.path.join(
            temp_dir, f"input_ts_{site_name}_{year}_x{lifetime_years}.csv"
        )
        lifetime_df.to_csv(year_input_ts_fn, sep=";")

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
        row = eval_df.iloc[0].to_dict()
        row["site"] = site_name
        row["weather_year"] = year
        row["lifetime_years"] = lifetime_years
        row["input_rows_per_year"] = len(year_df)
        row["input_rows_lifetime"] = len(lifetime_df)
        row.update(
            _extract_mean_annual_generation(
                hpp,
                lifetime_years,
                design["solar_MW"],
                year_df,
            )
        )
        row.update(calculate_bankability_metrics(row))
        print(
            f"Evaluated weather year {year} "
            f"with lifetime={lifetime_years} years"
        )
        return row

    rows = []
    with tempfile.TemporaryDirectory(prefix=f"hpp_eval_{site_name}_") as temp_dir:
        rows = Parallel(n_jobs=16)(
            delayed(evaluate_single_year)(
                year, site_name, latitude, longitude, altitude, sim_pars_fn, design_x, design, yearly_groups, lifetime_years, temp_dir
            )
            for year in range(start_year, end_year + 1)
        )

    yearly_results_df = pd.DataFrame(rows)
    return yearly_results_df


def main():
    _init_local_hydesign_imports()

    parser = argparse.ArgumentParser(
        description="Evaluate one site design from HPP/SiteConfig/<site>.csv"
    )
    parser.add_argument(
        "--site",
        nargs='+',
        default=["Sud_Atlantique", "NordsoenMidt", "SicilySouth", "Golfe_du_Lion", "Thetys", "Vestavind"],
        help="One or more site names from examples_sites.csv and SiteConfig/<site>.csv (space-separated)",
    )
    parser.add_argument(
        "--list-sites",
        action="store_true",
        help="List available site names from HPP/SiteConfig and exit",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1982,
        help="First weather year to evaluate (inclusive)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2015,
        help="Last weather year to evaluate (inclusive)",
    )
    parser.add_argument(
        "--lifetime-years",
        type=int,
        default=25,
        help="Number of times each weather year is repeated",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV file path for the yearly evaluation dataframe",
    )
    args = parser.parse_args()

    site_config_dir = _get_site_config_dir()
    if args.list_sites:
        sites = _available_site_configs(site_config_dir)
        print("Available site configs:")
        for site in sites:
            print(f"- {site}")
        return


    for site_name in args.site:
        try:
            ex_site = _load_site_row(site_name)
            design = _load_site_design(site_name, site_config_dir)

            longitude = ex_site["longitude"]
            latitude = ex_site["latitude"]
            altitude = ex_site["altitude"]
            input_ts_fn = examples_filepath + ex_site["input_ts_fn"]
            sim_pars_fn = examples_filepath + ex_site["sim_pars_fn"]

            start = time.time()
            yearly_results_df = evaluate_yearly_lifetime(
                site_name=site_name,
                latitude=latitude,
                longitude=longitude,
                altitude=altitude,
                sim_pars_fn=sim_pars_fn,
                input_ts_fn=input_ts_fn,
                design=design,
                start_year=args.start_year,
                end_year=args.end_year,
                lifetime_years=args.lifetime_years,
            )
            end = time.time()

            if args.output_csv is None:
                evaluations_dir = _get_evaluations_dir()
                os.makedirs(evaluations_dir, exist_ok=True)
                output_csv = os.path.join(
                    evaluations_dir,
                    (
                        f"{site_name}_yearly_eval_"
                        f"{args.start_year}_{args.end_year}_"
                        f"life{args.lifetime_years}.csv"
                    ),
                )
            else:
                # If multiple sites and output_csv is set, append site name to filename
                if len(args.site) > 1:
                    base, ext = os.path.splitext(args.output_csv)
                    output_csv = f"{base}_{site_name}{ext}"
                else:
                    output_csv = args.output_csv

            yearly_results_df.to_csv(output_csv, index=False)

            print(f"Site: {site_name}")
            print(f"Site config: {os.path.join(site_config_dir, f'{site_name}.csv')}")
            print(
                f"Years evaluated: {args.start_year}-{args.end_year} "
                f"(n={len(yearly_results_df)})"
            )
            print(f"Lifetime repeats per year: {args.lifetime_years}")
            print(f"Saved yearly results: {output_csv}")
            print("Yearly results dataframe (head):")
            print(yearly_results_df.head())
            print(f"exec. time [min]: {(end - start) / 60:.2f}")
        except Exception as e:
            print(f"Error processing site '{site_name}': {e}")


if __name__ == "__main__":
    main()
