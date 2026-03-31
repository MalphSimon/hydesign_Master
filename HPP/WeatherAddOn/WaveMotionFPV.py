import argparse
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr


VAR_ALIASES = {
    "VMDR_WW": "VMDR",
    "VTM10": "VTPK",
    "VTM01_WW": "VTPK",
}

LAT_NAME_CANDIDATES = ("latitude", "lat", "nav_lat", "y")
LON_NAME_CANDIDATES = ("longitude", "lon", "nav_lon", "x")
TIME_NAME_CANDIDATES = ("time", "time_counter", "valid_time")


def _first_existing_dir(candidates):
    for folder in candidates:
        if os.path.isdir(folder):
            return folder
    return candidates[0]


def _default_examples_sites_csv(script_dir):
    return os.path.abspath(
        os.path.join(
            script_dir,
            "..",
            "..",
            "hydesign",
            "examples",
            "examples_sites.csv",
        )
    )


def resolve_site_coordinates(site_name, sites_csv_path):
    if not os.path.isfile(sites_csv_path):
        raise FileNotFoundError(
            f"Site metadata file not found: {sites_csv_path}"
        )

    site_table = pd.read_csv(sites_csv_path, sep=";")
    required_cols = {"name", "latitude", "longitude"}
    missing_cols = required_cols - set(site_table.columns)
    if missing_cols:
        raise KeyError(
            "Missing required columns in site metadata: "
            f"{sorted(missing_cols)}"
        )

    site_match = site_table.loc[
        site_table["name"].astype(str).str.casefold()
        == str(site_name).casefold()
    ]
    if site_match.empty:
        known_sites = sorted(site_table["name"].dropna().astype(str).unique())
        preview = ", ".join(known_sites[:15])
        if len(known_sites) > 15:
            preview += ", ..."
        raise ValueError(
            f"Site '{site_name}' not found in: {sites_csv_path}. "
            f"Available examples: {preview}"
        )

    row = site_match.iloc[0]
    latitude = float(row["latitude"])
    longitude = float(row["longitude"])

    # Defensive fix for metadata rows with swapped lon/lat values.
    site_case = str(row.get("case", "")).strip().casefold()
    if (
        site_case == "europe"
        and not (35.0 <= latitude <= 72.0 and -20.0 <= longitude <= 35.0)
        and (35.0 <= longitude <= 72.0 and -20.0 <= latitude <= 35.0)
    ):
        print(
            "Warning: Detected swapped coordinates in "
            "site metadata; auto-correcting to Lat/Lon order."
        )
        latitude, longitude = longitude, latitude

    input_ts_rel = str(row.get("input_ts_fn", "")).strip()
    return latitude, longitude, input_ts_rel


def _open_dataset_with_fallback(path):
    """Open NetCDF with robust backend fallbacks for Windows environments."""
    engine_attempts = [None, "h5netcdf", "scipy"]
    last_error = None
    for engine in engine_attempts:
        try:
            if engine is None:
                return xr.open_dataset(path)
            return xr.open_dataset(path, engine=engine)
        except Exception as err:
            last_error = err
    raise last_error


def load_wave_dataset(wave_data_dir):
    file_pattern = os.path.join(wave_data_dir, "*.nc")
    matching_files = sorted(glob.glob(file_pattern))
    if not matching_files:
        raise FileNotFoundError(f"No NetCDF files found in: {wave_data_dir}")

    datasets_by_var = {}
    for path in matching_files:
        ds_single = _open_dataset_with_fallback(path)
        for var_name in ds_single.data_vars:
            target_var = VAR_ALIASES.get(var_name, var_name)
            if target_var in datasets_by_var:
                continue
            if target_var != var_name:
                ds_var = ds_single[[var_name]].rename({var_name: target_var})
            else:
                ds_var = ds_single[[var_name]]
            datasets_by_var[target_var] = ds_var

    if not datasets_by_var:
        raise ValueError("No wave variables found in NetCDF files")

    return xr.merge(
        list(datasets_by_var.values()),
        compat="override",
        join="outer",
        combine_attrs="override",
    )


def _pick_coordinate_name(ds, candidates):
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    return None


def _select_nearest_point(ds, latitude, longitude):
    lat_name = _pick_coordinate_name(ds, LAT_NAME_CANDIDATES)
    lon_name = _pick_coordinate_name(ds, LON_NAME_CANDIDATES)

    if lat_name and lon_name:
        lat_coord = ds[lat_name]
        lon_coord = ds[lon_name]

        if lat_coord.ndim == 1 and lon_coord.ndim == 1:
            return ds.sel(
                {lat_name: latitude, lon_name: longitude},
                method="nearest",
            )

        if lat_coord.ndim == 2 and lon_coord.ndim == 2:
            lat_vals = np.asarray(lat_coord.values, dtype=float)
            lon_vals = np.asarray(lon_coord.values, dtype=float)
            d2 = np.square(lat_vals - latitude) + np.square(lon_vals - longitude)
            d2[~np.isfinite(d2)] = np.inf
            flat_idx = int(np.argmin(d2))
            iy, ix = np.unravel_index(flat_idx, d2.shape)
            if not np.isfinite(d2[iy, ix]):
                raise ValueError("No finite sea-temperature grid point found")

            y_dim, x_dim = lat_coord.dims
            return ds.isel({y_dim: iy, x_dim: ix})

    if lat_name and lon_name:
        return ds.sel(
            {lat_name: latitude, lon_name: longitude},
            method="nearest",
        )
    raise KeyError(
        "Could not find latitude/longitude coordinates in dataset"
    )


def _select_nearest_wave_point_with_data(ds, latitude, longitude):
    """Pick nearest point, fallback to nearest point with finite VHM0."""
    if "VHM0" not in ds.data_vars:
        raise KeyError("Wave dataset must contain VHM0")

    nearest = _select_nearest_point(ds, latitude=latitude, longitude=longitude)
    nearest_vhm0 = np.asarray(nearest["VHM0"].values, dtype=float)
    if np.isfinite(nearest_vhm0).any():
        return nearest

    lat_name = _pick_coordinate_name(ds, LAT_NAME_CANDIDATES)
    lon_name = _pick_coordinate_name(ds, LON_NAME_CANDIDATES)
    if lat_name is None or lon_name is None:
        return nearest

    lat_coord = ds[lat_name]
    lon_coord = ds[lon_name]
    if lat_coord.ndim == 1 and lon_coord.ndim == 1:
        lat_grid, lon_grid = xr.broadcast(lat_coord, lon_coord)
    elif lat_coord.ndim == 2 and lon_coord.ndim == 2:
        lat_grid = lat_coord
        lon_grid = lon_coord
    else:
        return nearest

    time_name = _pick_coordinate_name(ds, TIME_NAME_CANDIDATES)
    valid_mask = xr.apply_ufunc(np.isfinite, ds["VHM0"])
    if time_name and time_name in valid_mask.dims:
        valid_mask = valid_mask.any(dim=time_name)

    spatial_dims = set(lat_grid.dims)
    extra_dims = [dim for dim in valid_mask.dims if dim not in spatial_dims]
    for dim in extra_dims:
        valid_mask = valid_mask.any(dim=dim)

    dist2 = np.square(lat_grid - latitude) + np.square(lon_grid - longitude)
    candidate_dist = dist2.where(valid_mask)
    dist_vals = np.asarray(candidate_dist.values, dtype=float)
    dist_vals[~np.isfinite(dist_vals)] = np.inf
    if np.isinf(dist_vals).all():
        return nearest

    flat_idx = int(np.argmin(dist_vals))
    point_idx = np.unravel_index(flat_idx, dist_vals.shape)
    isel_map = {
        dim: int(idx)
        for dim, idx in zip(candidate_dist.dims, point_idx)
    }
    print(
        "Warning: Nearest wave grid point had no valid VHM0; "
        "using nearest point with finite wave data instead."
    )
    return ds.isel(isel_map)


def load_sea_temp_dataset(sea_temp_dir):
    file_pattern = os.path.join(sea_temp_dir, "*.nc")
    matching_files = sorted(glob.glob(file_pattern))
    if not matching_files:
        raise FileNotFoundError(
            f"No NetCDF files found in sea temp directory: {sea_temp_dir}"
        )

    datasets = []
    time_names = []
    for path in matching_files:
        ds_single = _open_dataset_with_fallback(path)
        data_vars = list(ds_single.data_vars)
        if not data_vars:
            continue
        ds_selected = ds_single[data_vars]
        datasets.append(ds_selected)
        time_names.append(_pick_coordinate_name(ds_selected, TIME_NAME_CANDIDATES))

    if not datasets:
        raise ValueError("No sea-temperature variables found in NetCDF files")

    non_null_time_names = [name for name in time_names if name is not None]
    if non_null_time_names:
        time_name = max(set(non_null_time_names), key=non_null_time_names.count)
        concat_ready = [
            ds
            for ds in datasets
            if time_name in ds.coords or time_name in ds.dims
        ]
        if concat_ready:
            concat_ready = sorted(
                concat_ready,
                key=lambda d: pd.to_datetime(
                    d[time_name].values[0],
                    errors="coerce",
                ),
            )
            return xr.concat(
                concat_ready,
                dim=time_name,
                data_vars="minimal",
                coords="minimal",
                compat="override",
                join="outer",
                combine_attrs="override",
            )

    return xr.merge(
        datasets,
        compat="override",
        join="outer",
        combine_attrs="override",
    )


def build_hourly_sea_temp_series(ds, latitude, longitude):
    point_df = (
        _select_nearest_point(ds, latitude=latitude, longitude=longitude)
        .to_dataframe()
        .reset_index()
    )
    time_name = _pick_coordinate_name(ds, TIME_NAME_CANDIDATES)
    if time_name is None:
        time_name = "time"

    if time_name not in point_df.columns:
        raise KeyError("Sea temperature dataset must include a time coordinate")

    point_df = point_df.rename(columns={time_name: "time"})
    point_df["time"] = pd.to_datetime(point_df["time"], errors="coerce")
    point_df = point_df[point_df["time"].notna()].copy()
    if point_df.empty:
        raise ValueError(
            "No valid sea temperature timestamps found for selected location"
        )

    point_df = point_df.set_index("time").sort_index()
    numeric_cols = [
        col
        for col in point_df.columns
        if pd.api.types.is_numeric_dtype(point_df[col])
    ]
    excluded_cols = {
        "time",
        "y",
        "x",
        "latitude",
        "longitude",
        "lat",
        "lon",
        "nav_lat",
        "nav_lon",
    }
    numeric_cols = [col for col in numeric_cols if col not in excluded_cols]
    if not numeric_cols:
        raise ValueError("No numeric sea temperature variables found")

    sea_1h = (
        point_df[numeric_cols].resample("1h").interpolate(method="linear")
    )
    if sea_1h.empty:
        raise ValueError("Sea temperature time series is empty after resampling")

    return sea_1h


def map_profile_to_target(source_df, target_index):
    src = source_df.copy()
    src = src[src.index.notna()].sort_index()
    src = src[~src.index.duplicated(keep="first")]
    if src.empty:
        raise ValueError("Source profile is empty after timestamp cleaning")

    src = src.assign(
        month=src.index.month,
        day=src.index.day,
        hour=src.index.hour,
    )

    value_cols = [
        col
        for col in src.columns
        if col not in {"month", "day", "hour"}
    ]
    clim_by_mdh = src.groupby(["month", "day", "hour"])[value_cols].mean()
    clim_by_mh = src.groupby(["month", "hour"])[value_cols].mean()
    clim_by_m = src.groupby(["month"])[value_cols].mean()

    target_keys = pd.DataFrame(
        {
            "month": target_index.month,
            "day": target_index.day,
            "hour": target_index.hour,
        },
        index=target_index,
    )

    out = target_keys.join(clim_by_mdh, on=["month", "day", "hour"])
    out_mh = target_keys.join(
        clim_by_mh.add_suffix("_mh"),
        on=["month", "hour"],
    )
    out_m = target_keys.join(
        clim_by_m.add_suffix("_m"),
        on=["month"],
    )

    for col in value_cols:
        month_hour_col = f"{col}_mh"
        month_col = f"{col}_m"
        out[col] = out[col].fillna(out_mh[month_hour_col])
        out[col] = out[col].fillna(out_m[month_col])
        out[col] = out[col].fillna(float(src[col].mean()))

    return out[value_cols]


def verify_location(latitude, longitude, site_name=None):
    """Verify location coordinates and warn if they don't match expected sites."""
    KNOWN_SITES = {
        "Sud_Atlantique": (45.0, -1.78, "South Atlantic"),  # approximate
        "Vestavind": (60.3, 4.4, "Vest Vind, North Sea"),
        "NordsoenMidt": (56.5, 4.0, "Nordsoen Midt, North Sea"),
        "Golfe_du_Lion": (43.0, 4.2, "Gulf of Lion, Mediterranean"),
        "SicilySouth": (37.0, 13.2, "Sicily South, Mediterranean"),
        "Thetys": (52.5, 4.2, "Thetys, Atlantic"),
    }
    
    if site_name and site_name in KNOWN_SITES:
        expected_lat, expected_lon, description = KNOWN_SITES[site_name]
        lat_diff = abs(latitude - expected_lat)
        lon_diff = abs(longitude - expected_lon)

        print(f"\nLOCATION VERIFICATION for '{site_name}'")
        print(f"Expected region: {description}")
        print(f"Expected approx.: Lat={expected_lat:.1f}, Lon={expected_lon:.1f}")
        print(f"Actual location : Lat={latitude:.2f}, Lon={longitude:.2f}")

        if lat_diff > 2 or lon_diff > 2:
            print("\nWARNING: Coordinates differ significantly from expected region!")
            print("Location mismatch could cause wrong sea temperature selection.")
        else:
            print("Coordinates match expected region")


def build_hourly_wave_series(ds, latitude, longitude, fixed_tilt_deg=10.0):
    """Extract nearest-point wave data and compute hourly features."""
    point_df = (
        _select_nearest_wave_point_with_data(
            ds,
            latitude=latitude,
            longitude=longitude,
        )
        .to_dataframe()
        .reset_index()
    )
    time_name = _pick_coordinate_name(ds, TIME_NAME_CANDIDATES)
    if time_name is None:
        time_name = "time"
    if time_name not in point_df.columns:
        raise KeyError("Wave dataset must include a time coordinate")

    point_df = point_df.rename(columns={time_name: "time"})
    point_df["time"] = pd.to_datetime(point_df["time"], errors="coerce")
    point_df = point_df[point_df["time"].notna()].copy()
    if point_df.empty:
        raise ValueError(
            "No valid wave timestamps found for selected location"
        )

    point_df = point_df.set_index("time").sort_index()
    numeric_cols = [
        col
        for col in point_df.columns
        if pd.api.types.is_numeric_dtype(point_df[col])
    ]

    wave_1h = (
        point_df[numeric_cols].resample("1h").interpolate(method="linear")
    )

    if "VMDR" in point_df.columns:
        wave_1h["VMDR"] = point_df["VMDR"].resample("1h").nearest()

    if "VHM0" not in wave_1h.columns:
        raise KeyError("Wave dataset must contain VHM0 to compute wave slope")

    if "VTPK" not in wave_1h.columns:
        vhm0_clip = np.clip(
            pd.to_numeric(wave_1h["VHM0"], errors="coerce"),
            0.0,
            None,
        )
        wave_1h["VTPK"] = 5.0 * np.sqrt(vhm0_clip)

    vhm0 = pd.to_numeric(wave_1h["VHM0"], errors="coerce")
    vtpk = pd.to_numeric(wave_1h["VTPK"], errors="coerce")
    if vhm0.notna().sum() == 0 or vtpk.notna().sum() == 0:
        raise ValueError(
            "Wave variables are all NaN at the selected location. "
            "Try another --site/--lat/--lon, or use wave data with "
            "coverage at this location."
        )

    wavelength = 1.56 * np.square(vtpk)
    slope = np.degrees(np.arctan(np.pi * vhm0 / wavelength))
    slope = pd.Series(slope, index=wave_1h.index)
    slope = slope.replace([np.inf, -np.inf], np.nan).clip(lower=0.0)

    wave_1h["wavelength"] = wavelength
    wave_1h["wave_slope_deg"] = slope
    wave_1h["min_instant_tilt"] = np.maximum(0.0, fixed_tilt_deg - slope)
    wave_1h["max_instant_tilt"] = fixed_tilt_deg + slope

    return wave_1h


def map_2023_profile_to_target(wave_1h, target_index):
    """Map 2023 wave profile onto arbitrary years via climatology keys."""
    keep_cols = [
        col
        for col in (
            "VHM0",
            "VTPK",
            "VMDR",
            "wavelength",
            "wave_slope_deg",
            "min_instant_tilt",
            "max_instant_tilt",
        )
        if col in wave_1h.columns
    ]
    return map_profile_to_target(
        source_df=wave_1h[keep_cols],
        target_index=target_index,
    )


def read_input_timeseries(input_ts_fn):
    weather = pd.read_csv(input_ts_fn, sep=";", index_col=0)

    idx_dayfirst = pd.to_datetime(
        weather.index,
        errors="coerce",
        dayfirst=True,
    )
    idx_default = pd.to_datetime(
        weather.index,
        errors="coerce",
        dayfirst=False,
    )
    if idx_dayfirst.notna().sum() >= idx_default.notna().sum():
        parsed_index = idx_dayfirst
    else:
        parsed_index = idx_default

    valid_mask = parsed_index.notna()
    weather = weather.loc[valid_mask].copy()
    raw_time_strings = pd.Index(weather.index.astype(str))
    weather.index = parsed_index[valid_mask]

    if weather.empty:
        raise ValueError(f"No valid timestamps found in: {input_ts_fn}")
    return weather, raw_time_strings


def write_outputs(
    mapped_wave,
    input_weather,
    input_time_strings,
    input_ts_fn,
    output_merged_fn,
    mapped_sea_temp=None,
    latitude=None,
    longitude=None,
):
    merged = input_weather.copy()
    merged["wave_slope_deg"] = mapped_wave["wave_slope_deg"].values
    if mapped_sea_temp is not None and not mapped_sea_temp.empty:
        for col in mapped_sea_temp.columns:
            merged[col] = mapped_sea_temp[col].values
    merged.insert(0, "time", input_time_strings.values)
    merged.to_csv(output_merged_fn, sep=";", index=False)

    print(f"Input file       : {input_ts_fn}")
    print(f"Merged output    : {output_merged_fn}")
    print(f"Location         : Latitude={latitude:.2f}, Longitude={longitude:.2f}")
    print(f"Time range       : {merged.index[0]} to {merged.index[-1]}")
    print()
    
    # Wave statistics
    print("=" * 70)
    print("WAVE STATISTICS")
    print("=" * 70)
    print(
        "wave_slope_deg   | "
        f"min={mapped_wave['wave_slope_deg'].min():.4f}°, "
        f"max={mapped_wave['wave_slope_deg'].max():.4f}°, "
        f"mean={mapped_wave['wave_slope_deg'].mean():.4f}°"
    )
    
    # Air temperature statistics
    print()
    print("=" * 70)
    print("AIR TEMPERATURE STATISTICS")
    print("=" * 70)
    if "temp_air_1" in input_weather.columns:
        air_temp = pd.to_numeric(input_weather["temp_air_1"], errors="coerce")
        air_temp_clean = air_temp.dropna()
        if not air_temp_clean.empty:
            print(f"Column           : temp_air_1")
            print(f"Min temperature  : {air_temp_clean.min():.2f}°C")
            print(f"Max temperature  : {air_temp_clean.max():.2f}°C")
            print(f"Mean temperature : {air_temp_clean.mean():.2f}°C")
            print(f"Std deviation    : {air_temp_clean.std():.2f}°C")
            print(f"Valid samples    : {len(air_temp_clean)} / {len(air_temp)}")
            
            # Monthly statistics
            monthly_temps = input_weather.groupby(input_weather.index.month)["temp_air_1"].apply(
                lambda x: pd.to_numeric(x, errors="coerce").mean()
            )
            print("\nMonthly air temperature averages:")
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            for month in range(1, 13):
                if month in monthly_temps.index:
                    print(f"  {month_names[month-1]:3s}: {monthly_temps[month]:6.2f}°C")
    
    # Sea temperature statistics
    if mapped_sea_temp is not None and not mapped_sea_temp.empty:
        print()
        print("=" * 70)
        print("SEA TEMPERATURE STATISTICS")
        print("=" * 70)
        for col in mapped_sea_temp.columns:
            sea_temp = pd.to_numeric(mapped_sea_temp[col], errors="coerce")
            sea_temp_clean = sea_temp.dropna()
            if not sea_temp_clean.empty:
                print(f"\nColumn           : {col}")
                print(f"Min temperature  : {sea_temp_clean.min():.2f}°C")
                print(f"Max temperature  : {sea_temp_clean.max():.2f}°C")
                print(f"Mean temperature : {sea_temp_clean.mean():.2f}°C")
                print(f"Std deviation    : {sea_temp_clean.std():.2f}°C")
                print(f"Valid samples    : {len(sea_temp_clean)} / {len(sea_temp)}")
                
                # Monthly statistics
                merged_temp = merged[[col]].copy()
                merged_temp.index = input_weather.index
                monthly_sea_temps = merged_temp.groupby(merged_temp.index.month)[col].apply(
                    lambda x: pd.to_numeric(x, errors="coerce").mean()
                )
                print("\nMonthly sea temperature averages:")
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                for month in range(1, 13):
                    if month in monthly_sea_temps.index:
                        print(f"  {month_names[month-1]:3s}: {monthly_sea_temps[month]:6.2f}°C")
                
                # Sanity checks
                print("\n⚠️  SANITY CHECKS:")
                if sea_temp_clean.min() < -2 or sea_temp_clean.max() > 35:
                    print(f"  ⚠️  WARNING: Unrealistic sea temperatures detected!")
                if sea_temp_clean.std() < 1:
                    print(f"  ⚠️  WARNING: Very low temperature variation ({sea_temp_clean.std():.2f}°C std)")
                    print(f"     This suggests the data might be constant or from a wrong location.")
    else:
        print()
        print("=" * 70)
        print("SEA TEMPERATURE")
        print("=" * 70)
        print("No sea temperature data available")
    print()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate wave-motion time series aligned to input_ts timestamps "
            "using available 2023 wave NetCDF data."
        )
    )
    parser.add_argument(
        "--input-ts",
        default=None,
        help=(
            "Path to input_ts CSV (same format as hydesign examples). "
            "If omitted and --site is used, "
            "it is inferred from examples_sites.csv."
        ),
    )
    parser.add_argument(
        "--site",
        default="Sud_Atlantique",
        help=(
            "Site name from examples_sites.csv. "
            "If provided, latitude/longitude are loaded automatically."
        ),
    )
    parser.add_argument(
        "--sites-csv",
        default=None,
        help=(
            "Optional path to examples_sites.csv. "
            "Defaults to hydesign/examples/examples_sites.csv in this repo."
        ),
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=None,
        help="Site latitude (used when --site is not provided).",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=None,
        help="Site longitude (used when --site is not provided).",
    )
    parser.add_argument(
        "--fixed-tilt",
        type=float,
        default=10.0,
        help="Fixed PV tilt used to compute min/max instantaneous tilt.",
    )
    parser.add_argument(
        "--wave-data-dir",
        default=None,
        help=(
            "Directory with wave NetCDF files. "
            "Defaults to MasterThesisHPP/WaveData."
        ),
    )
    parser.add_argument(
        "--sea-temp-dir",
        default=None,
        help=(
            "Directory with sea-temperature NetCDF files. "
            "Defaults to MasterThesisHPP/SeaTemp."
        ),
    )
    parser.add_argument(
        "--output-merged",
        default=None,
        help=(
            "Output CSV path for input_ts enriched with wave_slope_deg "
            "and optional sea-temperature columns."
        ),
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_wave_dir = _first_existing_dir(
        [
            os.path.join(script_dir, "WaveData"),
            os.path.abspath(os.path.join(script_dir, "..", "WaveData")),
        ]
    )
    default_sea_temp_dir = _first_existing_dir(
        [
            os.path.join(script_dir, "SeaTemp"),
            os.path.abspath(os.path.join(script_dir, "..", "SeaTemp")),
        ]
    )
    wave_data_dir = args.wave_data_dir or default_wave_dir
    sea_temp_dir = args.sea_temp_dir or default_sea_temp_dir

    latitude = args.lat
    longitude = args.lon
    input_ts_fn = args.input_ts

    if args.site:
        sites_csv = args.sites_csv or _default_examples_sites_csv(script_dir)
        site_lat, site_lon, site_input_ts_rel = resolve_site_coordinates(
            site_name=args.site,
            sites_csv_path=sites_csv,
        )
        if latitude is None:
            latitude = site_lat
        if longitude is None:
            longitude = site_lon
        if input_ts_fn is None and site_input_ts_rel:
            input_ts_fn = os.path.abspath(
                os.path.join(os.path.dirname(sites_csv), site_input_ts_rel)
            )

    if latitude is None or longitude is None:
        raise ValueError(
            "Provide either --site or both --lat and --lon."
        )

    verify_location(latitude, longitude, site_name=args.site)

    if input_ts_fn is None:
        raise ValueError(
            "Provide --input-ts, or provide --site with input_ts_fn "
            "in site metadata."
        )

    base_name = os.path.splitext(os.path.basename(input_ts_fn))[0]
    output_merged_fn = args.output_merged or os.path.join(
        script_dir, f"{base_name}_with_wave.csv"
    )

    ds = load_wave_dataset(wave_data_dir)
    wave_1h = build_hourly_wave_series(
        ds=ds,
        latitude=latitude,
        longitude=longitude,
        fixed_tilt_deg=args.fixed_tilt,
    )
    input_weather, input_time_strings = read_input_timeseries(input_ts_fn)
    mapped_wave = map_2023_profile_to_target(
        wave_1h=wave_1h,
        target_index=input_weather.index,
    )

    mapped_sea_temp = None
    if os.path.isdir(sea_temp_dir):
        try:
            ds_sea_temp = load_sea_temp_dataset(sea_temp_dir)
            sea_1h = build_hourly_sea_temp_series(
                ds=ds_sea_temp,
                latitude=latitude,
                longitude=longitude,
            )
            mapped_sea_temp = map_profile_to_target(
                source_df=sea_1h,
                target_index=input_weather.index,
            )
            mapped_sea_temp = mapped_sea_temp.add_prefix("sea_temp_")
        except Exception as err:
            print(
                "Warning: Sea temperature data could not be processed. "
                f"Continuing without it. Error: {err}"
            )
    else:
        print(
            "Warning: Sea temperature directory not found. "
            f"Skipping SeaTemp load: {sea_temp_dir}"
        )

    write_outputs(
        mapped_wave=mapped_wave,
        input_weather=input_weather,
        input_time_strings=input_time_strings,
        input_ts_fn=input_ts_fn,
        output_merged_fn=output_merged_fn,
        mapped_sea_temp=mapped_sea_temp,
        latitude=latitude,
        longitude=longitude,
    )


if __name__ == "__main__":
    main()
