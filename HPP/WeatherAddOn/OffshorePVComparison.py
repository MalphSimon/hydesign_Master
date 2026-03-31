# Compare offshore PV results with and without wave-induced tilt effects
# for a single weather year using the same site design.

import argparse
import os
import numpy as np
import pandas as pd

from hydesign.assembly.hpp_assembly import hpp_model
from hydesign.examples import examples_filepath


def _load_site_row(site_name):
    examples_sites = pd.read_csv(
        f"{examples_filepath}examples_sites.csv",
        index_col=0,
        sep=";",
    )
    ex_site = examples_sites.loc[examples_sites.name == site_name]
    if ex_site.empty:
        raise ValueError(
            f"Site '{site_name}' not found in examples_sites.csv"
        )
    return ex_site


def _resolve_input_ts(ex_site, script_dir, input_ts_dir=None):
    input_ts_rel = str(ex_site["input_ts_fn"].values[0])
    input_ts_default = os.path.abspath(
        os.path.join(examples_filepath, input_ts_rel)
    )
    input_ts_base = os.path.splitext(os.path.basename(input_ts_rel))[0]
    input_ts_with_wave = f"{input_ts_base}_with_wave.csv"

    search_dirs = []
    if input_ts_dir:
        search_dirs.append(os.path.abspath(input_ts_dir))
    search_dirs.append(script_dir)
    search_dirs.append(os.path.dirname(input_ts_default))

    unique_dirs = []
    seen_dirs = set()
    for folder in search_dirs:
        key = os.path.normcase(os.path.normpath(folder))
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        unique_dirs.append(folder)

    with_wave_candidates = [
        os.path.join(folder, input_ts_with_wave)
        for folder in unique_dirs
    ]
    with_wave_existing = [
        path for path in with_wave_candidates if os.path.isfile(path)
    ]

    if with_wave_existing:
        return with_wave_existing[0], "with_wave"

    if not os.path.isfile(input_ts_default):
        raise FileNotFoundError(
            f"Input weather file not found: {input_ts_default}"
        )
    return input_ts_default, "default"


def _load_site_design(site_name, script_dir):
    site_config_path = os.path.join(script_dir, "SiteConfig", f"{site_name}.csv")
    if not os.path.isfile(site_config_path):
        raise FileNotFoundError(
            f"Site config file not found: {site_config_path}"
        )

    site_config = pd.read_csv(
        site_config_path,
        index_col=0,
        header=None,
        sep=",",
    )
    site_config.columns = ["value"]

    design = {
        "clearance": float(site_config.loc["clearance [m]", "value"]),
        "sp": float(site_config.loc["sp [W/m2]", "value"]),
        "wt_rated_power_MW": float(site_config.loc["p_rated [MW]", "value"]),
        "Nwt": int(float(site_config.loc["Nwt", "value"])),
        "wind_MW_per_km2": float(
            site_config.loc["wind_MW_per_km2 [MW/km2]", "value"]
        ),
        "solar_MW": float(site_config.loc["solar_MW [MW]", "value"]),
        "surface_tilt_deg": float(
            site_config.loc["surface_tilt [deg]", "value"]
        ),
        "surface_azimuth_deg": float(
            site_config.loc["surface_azimuth [deg]", "value"]
        ),
        "DC_AC_ratio": float(site_config.loc["DC_AC_ratio", "value"]),
        "b_P": float(site_config.loc["b_P [MW]", "value"]),
        "b_E_h": float(site_config.loc["b_E_h [h]", "value"]),
        "cost_of_batt_degr": float(
            site_config.loc[
                "cost_of_battery_P_fluct_in_peak_price_ratio",
                "value",
            ]
        ),
        "rotor_diameter_m": float(site_config.loc["Rotor diam [m]", "value"]),
        "hub_height_m": float(site_config.loc["Hub height [m]", "value"]),
    }
    return design


def _scalar(value):
    return float(np.asarray(value).reshape(-1)[0])


def _build_metric_table(list_out_vars, outs_off, outs_on):
    rows = []
    for out_name, off_value, on_value in zip(list_out_vars, outs_off, outs_on):
        off_scalar = _scalar(off_value)
        on_scalar = _scalar(on_value)
        delta_abs = on_scalar - off_scalar
        if abs(off_scalar) > 1.0e-12:
            delta_pct = 100.0 * delta_abs / off_scalar
        else:
            delta_pct = np.nan
        rows.append(
            {
                "metric": out_name,
                "wave_off": off_scalar,
                "wave_on": on_scalar,
                "delta_abs_on_minus_off": delta_abs,
                "delta_pct_on_minus_off": delta_pct,
            }
        )
    return pd.DataFrame(rows)


def _get_prob_var(prob, var_name):
    try:
        return np.asarray(prob.get_val(var_name)).reshape(-1)
    except Exception:
        try:
            return np.asarray(prob[var_name]).reshape(-1)
        except Exception:
            return None


def _extract_solar_metrics(hpp, solar_capacity_mw):
    solar_t = _get_prob_var(hpp.prob, "solar_t")
    if solar_t is None or solar_t.size == 0:
        return {
            "Solar electricity produced [GWh]": np.nan,
            "Capacity factor solar [-]": np.nan,
        }

    solar_t = np.nan_to_num(solar_t, nan=0.0)
    solar_energy_gwh = float(np.sum(solar_t) / 1000.0)

    if solar_capacity_mw > 0.0:
        solar_cf = float(np.mean(solar_t) / solar_capacity_mw)
    else:
        solar_cf = np.nan

    return {
        "Solar electricity produced [GWh]": solar_energy_gwh,
        "Capacity factor solar [-]": solar_cf,
    }


def _append_extra_metrics(metrics_df, off_metrics, on_metrics):
    extra_rows = []
    for metric_name in [
        "Solar electricity produced [GWh]",
        "Capacity factor solar [-]",
    ]:
        off_value = float(off_metrics.get(metric_name, np.nan))
        on_value = float(on_metrics.get(metric_name, np.nan))
        delta_abs = on_value - off_value
        if np.isfinite(off_value) and abs(off_value) > 1.0e-12:
            delta_pct = 100.0 * delta_abs / off_value
        else:
            delta_pct = np.nan

        extra_rows.append(
            {
                "metric": metric_name,
                "wave_off": off_value,
                "wave_on": on_value,
                "delta_abs_on_minus_off": delta_abs,
                "delta_pct_on_minus_off": delta_pct,
            }
        )

    return pd.concat([metrics_df, pd.DataFrame(extra_rows)], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare offshore PV one-year performance with wave mode OFF and ON."
        )
    )
    parser.add_argument(
        "--site",
        default="NordsoenMidt",
        help="Site name from examples_sites.csv and SiteConfig/<site>.csv",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Weather year to evaluate (must have 8760 hourly rows).",
    )
    parser.add_argument(
        "--price-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to all price columns before comparison.",
    )
    parser.add_argument(
        "--price-offset",
        type=float,
        default=40.0,
        help="Additive EUR/MWh shift applied to all price columns.",
    )
    parser.add_argument(
        "--input-ts-dir",
        default=None,
        help="Optional folder containing input_ts CSV files.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    site_name = args.site

    ex_site = _load_site_row(site_name)
    input_ts_fn, input_source = _resolve_input_ts(
        ex_site=ex_site,
        script_dir=script_dir,
        input_ts_dir=args.input_ts_dir,
    )

    print(f"Site: {site_name}")
    print(f"Input source: {input_source}")
    print(f"Input file: {input_ts_fn}")

    weather_ts = pd.read_csv(input_ts_fn, index_col=0, sep=";")
    weather_ts.index = pd.to_datetime(
        weather_ts.index,
        errors="coerce",
        dayfirst=True,
    )
    weather_ts = weather_ts[weather_ts.index.notna()].sort_index()

    if args.price_scale != 1.0 or args.price_offset != 0.0:
        price_cols = [
            col for col in weather_ts.columns if "price" in col.lower()
        ]
        if price_cols:
            adjusted_prices = weather_ts.loc[:, price_cols] * args.price_scale
            adjusted_prices = adjusted_prices + args.price_offset
            weather_ts.loc[:, price_cols] = adjusted_prices

    hours_per_year = 365 * 24
    year_counts = weather_ts.index.year.value_counts().sort_index()
    available_years = [
        int(year)
        for year, count in year_counts.items()
        if int(count) == hours_per_year
    ]
    if not available_years:
        raise ValueError(
            "No complete weather years (8760 h) were found in input data."
        )

    if args.year is None:
        selected_year = available_years[0]
    else:
        selected_year = int(args.year)
        if selected_year not in available_years:
            raise ValueError(
                f"Selected year {selected_year} is not available as a "
                f"complete 8760-hour year. Available years: {available_years}"
            )

    year_df = weather_ts.loc[weather_ts.index.year == selected_year].copy()

    wave_col_aliases = {
        "wave_slope_deg",
        "wave_tilt_deg",
        "offshore_wave_tilt_deg",
    }
    wave_cols_in_data = [
        col for col in year_df.columns
        if str(col).strip().lower() in wave_col_aliases
    ]
    year_df_off = year_df.copy()
    if wave_cols_in_data:
        year_df_off = year_df_off.drop(columns=wave_cols_in_data)

    site_folder = "".join(
        "_" if char in '<>:"/\\|?*' else char for char in site_name
    ).strip()
    if not site_folder:
        site_folder = "UnknownSite"

    comparison_dir = os.path.join(script_dir, "HPPData", site_folder, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    input_on_fn = os.path.join(
        comparison_dir,
        f"input_ts_{selected_year}_wave_on.csv",
    )
    input_off_fn = os.path.join(
        comparison_dir,
        f"input_ts_{selected_year}_wave_off.csv",
    )
    year_df.to_csv(input_on_fn, sep=";")
    year_df_off.to_csv(input_off_fn, sep=";")

    longitude = float(ex_site["longitude"].values[0])
    latitude = float(ex_site["latitude"].values[0])
    altitude = float(ex_site["altitude"].values[0])

    sim_pars_fn = examples_filepath + ex_site["sim_pars_fn"].values[0]

    design = _load_site_design(site_name=site_name, script_dir=script_dir)
    x = [
        design["clearance"],
        design["sp"],
        design["wt_rated_power_MW"],
        design["Nwt"],
        design["wind_MW_per_km2"],
        design["solar_MW"],
        design["surface_tilt_deg"],
        design["surface_azimuth_deg"],
        design["DC_AC_ratio"],
        design["b_P"],
        design["b_E_h"],
        design["cost_of_batt_degr"],
    ]

    common_kwargs = {
        "latitude": latitude,
        "longitude": longitude,
        "altitude": altitude,
        "rotor_diameter_m": design["rotor_diameter_m"],
        "hub_height_m": design["hub_height_m"],
        "wt_rated_power_MW": design["wt_rated_power_MW"],
        "surface_tilt_deg": design["surface_tilt_deg"],
        "surface_azimuth_deg": design["surface_azimuth_deg"],
        "DC_AC_ratio": design["DC_AC_ratio"],
        "num_batteries": 5,
        "work_dir": "./",
        "sim_pars_fn": sim_pars_fn,
    }

    hpp_off = hpp_model(
        input_ts_fn=input_off_fn,
        use_wave_motion=False,
        **common_kwargs,
    )
    outs_off = hpp_off.evaluate(*x)

    hpp_on = hpp_model(
        input_ts_fn=input_on_fn,
        use_wave_motion=True,
        **common_kwargs,
    )
    outs_on = hpp_on.evaluate(*x)

    metrics = _build_metric_table(hpp_on.list_out_vars, outs_off, outs_on)

    solar_off = _extract_solar_metrics(hpp_off, design["solar_MW"])
    solar_on = _extract_solar_metrics(hpp_on, design["solar_MW"])
    metrics = _append_extra_metrics(metrics, solar_off, solar_on)

    metrics_fn = os.path.join(
        comparison_dir,
        f"offshore_wave_comparison_{selected_year}.csv",
    )
    metrics.to_csv(metrics_fn, index=False)

    print(
        f"Compared one year: {selected_year} "
        f"| rows: {len(year_df)} "
        f"| wave columns found: {wave_cols_in_data}"
    )
    print(f"Comparison CSV written to: {metrics_fn}")

    key_metrics = [
        "AEP [GWh]",
        "AEP with degradation [GWh]",
        "NPV [MEuro]",
        "LCOE [Euro/MWh]",
        "Mean Annual Electricity Sold [GWh]",
        "Solar electricity produced [GWh]",
        "Capacity factor solar [-]",
    ]
    key_table = metrics.loc[metrics["metric"].isin(key_metrics)].copy()
    if not key_table.empty:
        print("")
        print("Key metrics (wave ON minus OFF):")
        for _, row in key_table.iterrows():
            print(
                f"{row['metric']}: "
                f"off={row['wave_off']:.6f}, "
                f"on={row['wave_on']:.6f}, "
                f"delta={row['delta_abs_on_minus_off']:.6f} "
                f"({row['delta_pct_on_minus_off']:.4f}%)"
            )


if __name__ == "__main__":
    main()
