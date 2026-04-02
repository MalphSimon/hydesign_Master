# Load data from input_ts_fn files listed in examples_sites.csv.

import os
from typing import Dict, Optional

import pandas as pd


def _default_examples_sites_fn() -> str:
    """Return default path to examples_sites.csv in this repository."""
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return os.path.join(
        repo_root,
        "hydesign",
        "examples",
        "examples_sites.csv",
    )


def load_examples_sites(
    examples_sites_fn: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the examples sites table.

    Parameters
    ----------
    examples_sites_fn : Optional[str]
        Path to examples_sites.csv. If None, use the repository default.

    Returns
    -------
    pd.DataFrame
        DataFrame containing site metadata and file references.
    """
    sites_fn = examples_sites_fn or _default_examples_sites_fn()
    return pd.read_csv(sites_fn, index_col=0, sep=";")


def load_input_timeseries(
    examples_sites_fn: Optional[str] = None,
    examples_base_dir: Optional[str] = None,
    parse_dates: bool = True,
) -> Dict[int, pd.DataFrame]:
    """
    Load all input time-series files listed in the input_ts_fn column.

    Parameters
    ----------
    examples_sites_fn : Optional[str]
        Path to examples_sites.csv. If None, use the repository default.
    examples_base_dir : Optional[str]
        Base directory for relative paths from input_ts_fn.
        If None, defaults to the folder containing examples_sites.csv.
    parse_dates : bool
        Whether to parse the first column as datetime index.

    Returns
    -------
    Dict[int, pd.DataFrame]
        Mapping from examples_sites row index to loaded input time series.
    """
    sites_fn = examples_sites_fn or _default_examples_sites_fn()
    examples_sites = load_examples_sites(sites_fn)
    base_dir = examples_base_dir or os.path.dirname(sites_fn)

    if "input_ts_fn" not in examples_sites.columns:
        raise KeyError(
            "Column 'input_ts_fn' not found in examples_sites.csv"
        )

    loaded_ts: Dict[int, pd.DataFrame] = {}

    for idx, row in examples_sites.iterrows():
        rel_path = row.get("input_ts_fn")
        if pd.isna(rel_path) or str(rel_path).strip() == "":
            continue

        ts_fn = os.path.join(base_dir, str(rel_path))
        if not os.path.exists(ts_fn):
            raise FileNotFoundError(
                f"Input time-series file not found: {ts_fn}"
            )

        loaded_ts[idx] = pd.read_csv(
            ts_fn,
            index_col=0,
            sep=";",
            parse_dates=[0] if parse_dates else False,
            dayfirst=True,
        )

    return loaded_ts


if __name__ == "__main__":
    input_ts_by_site = load_input_timeseries()
    print(f"Loaded {len(input_ts_by_site)} input_ts_fn files.")
