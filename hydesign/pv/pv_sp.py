import numpy as np
import pandas as pd

from hydesign.openmdao_wrapper import ComponentWrapper


def _ensure_datetime_index(df, source_label):
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    idx_dayfirst = pd.to_datetime(df.index, errors="coerce", dayfirst=True)
    idx_monthfirst = pd.to_datetime(df.index, errors="coerce", dayfirst=False)

    if idx_dayfirst.notna().sum() >= idx_monthfirst.notna().sum():
        idx = idx_dayfirst
    else:
        idx = idx_monthfirst

    if idx.isna().any():
        n_invalid = int(idx.isna().sum())
        raise ValueError(
            f"Failed to parse {n_invalid} datetime value(s) in weather index "
            f"from {source_label}."
        )

    out = df.copy()
    out.index = pd.DatetimeIndex(idx)
    out = out.sort_index()
    return out


class pvp_sp:
    """PV model variant using weather column SP as hourly PV CF."""

    def __init__(
        self,
        weather_fn,
        N_time,
        latitude,
        longitude,
        altitude,
        tracking="single_axis",
    ):
        del latitude, longitude, altitude, tracking
        self.weather_fn = weather_fn
        self.N_time = N_time

        weather = pd.read_csv(
            weather_fn,
            index_col=0,
            parse_dates=True,
            sep=None,
            engine="python",
        )
        weather = _ensure_datetime_index(weather, weather_fn)

        sp_col = None
        for col in weather.columns:
            if str(col).strip().lower() == "sp":
                sp_col = col
                break

        if sp_col is None:
            raise ValueError(
                "Expected SP column in weather file for pvp_sp model."
            )

        self.sp_ts = pd.to_numeric(
            weather[sp_col],
            errors="coerce",
        ).fillna(0.0)
        self.sp_ts = self.sp_ts.clip(lower=0.0)

        if len(self.sp_ts) != N_time:
            raise ValueError(
                "SP series length "
                f"{len(self.sp_ts)} does not match N_time={N_time}."
            )

    def compute(
        self,
        surface_tilt,
        surface_azimuth,
        solar_MW,
        land_use_per_solar_MW,
        DC_AC_ratio,
    ):
        del surface_tilt, surface_azimuth, DC_AC_ratio
        Apvp = solar_MW * land_use_per_solar_MW
        solar_t = np.asarray(self.sp_ts) * solar_MW
        return solar_t, Apvp


class pvp_sp_comp(ComponentWrapper):
    def __init__(
        self,
        weather_fn,
        N_time,
        latitude,
        longitude,
        altitude,
        tracking="single_axis",
    ):
        model = pvp_sp(
            weather_fn,
            N_time,
            latitude,
            longitude,
            altitude,
            tracking,
        )
        super().__init__(
            inputs=[
                (
                    "surface_tilt",
                    {"val": 20, "desc": "Solar PV tilt angle in degs"},
                ),
                (
                    "surface_azimuth",
                    {"val": 180, "desc": "Solar PV azimuth angle in degs"},
                ),
                ("DC_AC_ratio", {"desc": "DC/AC PV ratio"}),
                (
                    "solar_MW",
                    {
                        "val": 1,
                        "desc": "Solar PV plant installed capacity",
                        "units": "MW",
                    },
                ),
                (
                    "land_use_per_solar_MW",
                    {
                        "val": 1,
                        "desc": "Solar land use per solar MW",
                        "units": "km**2/MW",
                    },
                ),
            ],
            outputs=[
                (
                    "solar_t",
                    {
                        "desc": "PV power time series",
                        "units": "MW",
                        "shape": [N_time],
                    },
                ),
                ("Apvp", {"desc": "Land use area of WPP", "units": "km**2"}),
            ],
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )
