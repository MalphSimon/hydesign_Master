"""
Hybrid Power Plant (HPP) Portfolio Aggregator.

This script builds user-defined HPP portfolios by aggregating existing site outputs.
It performs three main types of data processing:
1. Summation: Combines Energy (GWh), Revenue (MEuro), and Capacity (MW).
2. Averaging: Calculates mean values for ratios (IRR, LCOE, CF) and physical specs.
3. Filtering: Removes geographic and site-specific metadata irrelevant to a portfolio.
"""

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration & Global Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Paths relative to the script location
DEFAULT_EVAL_DIR = os.path.join(SCRIPT_DIR, "..", "Evaluations", "Original")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Outputs")
DEFAULT_SITE_CONFIG_DIR = os.path.join(SCRIPT_DIR, "..", "SiteConfig")

# Define which sites belong to which named portfolio
PORTFOLIOS: Dict[str, Sequence[str]] = {
    "NordsoenMidt_Golfe_du_Lion": ["NordsoenMidt", "Golfe_du_Lion"],
    "NordsoenMidt_SicilySouth": ["NordsoenMidt", "SicilySouth"],
    "All_Sites": [
        "NordsoenMidt", "Golfe_du_Lion", "SicilySouth", 
        "Thetys", "Sud_Atlantique", "Vestavind"
    ],
}

# Metadata columns to be removed from the final portfolio files
COLS_TO_DROP = [
    "longitude", "latitude", "altitude", "surface_tilt [deg]", 
    "surface_azimuth [deg]", "cost_of_battery_P_fluct_in_peak_price_ratio", 
    "clearance [m]", "sp [W/m2]", "p_rated [MW]", "Nwt", 
    "wind_MW_per_km2 [MW/km2]", "solar_MW [MW]", "DC_AC_ratio", 
    "b_P [MW]", "b_E_h [h]", "penalty lifetime [MEuro]",
    "Rotor diam [m]", "Hub height [m]"
]

# Metrics that represent ratios or efficiencies; these are averaged across 
# sites rather than summed to maintain physical meaning.
COLS_TO_AVERAGE = [
    "DC_AC_ratio", "b_E_h [h]", "NPV_over_CAPEX", "IRR", "LCOE [Euro/MWh]", 
    "GUF", "Break-even PPA price [Euro/MWh]", 
    "Capacity factor wind [-]", "Capacity factor solar [-]", "DSCR Breach Years",
]

@dataclass
class PortfolioResult:
    """Container for the different levels of aggregated portfolio data."""
    hourly: pd.DataFrame
    yearly: pd.DataFrame
    summary: pd.DataFrame
    capacities: pd.DataFrame

# ---------------------------------------------------------------------------
# Helper Functions (File I/O)
# ---------------------------------------------------------------------------

def _find_single_file(eval_dir: str, pattern: str) -> str:
    """
    Locates a single file matching a glob pattern.
    
    Args:
        eval_dir: Directory to search.
        pattern: Glob pattern (e.g., '*_hourly_*.csv').
        
    Returns:
        str: Path to the first matching file found.
    """
    matches = sorted(glob.glob(os.path.join(eval_dir, pattern)))
    if not matches:
        raise FileNotFoundError(f"No file found for pattern: {pattern}")
    return matches[0]

def available_sites(eval_dir: str) -> List[str]:
    """
    Scans the evaluation directory to identify which sites have results.
    
    Returns:
        List[str]: Alphabetical list of unique site names found.
    """
    files = glob.glob(os.path.join(eval_dir, "*_hourly_production_*.csv"))
    token = "_hourly_production_"
    return sorted(set([os.path.basename(f).split(token)[0] for f in files]))

def _load_hourly(site: str, eval_dir: str) -> pd.DataFrame:
    """Loads and formats hourly production data for a specific site."""
    path = _find_single_file(eval_dir, f"{site}_hourly_production_*.csv")
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["total_t"] = df["wind_t"] + df["solar_t"]
    return df

def _load_yearly(site: str, eval_dir: str) -> pd.DataFrame:
    """Loads the yearly evaluation results (NPV, Revenue, etc.) for a site."""
    path = _find_single_file(eval_dir, f"{site}_yearly_eval_*_life*.csv")
    return pd.read_csv(path)

# ---------------------------------------------------------------------------
# Aggregation Logic
# ---------------------------------------------------------------------------

def _aggregate_hourly(sites: Sequence[str], eval_dir: str) -> pd.DataFrame:
    """
    Aggregates hourly time-series production across multiple sites.
    
    Sums wind and solar production for every timestamp to provide 
    a portfolio-level production profile.
    """
    portfolio_hourly = None
    for site in sites:
        site_df = _load_hourly(site, eval_dir)
        
        if portfolio_hourly is None:
            # Initialize the dataframe with the first site
            portfolio_hourly = site_df[["time"]].copy()
            portfolio_hourly["portfolio_wind_t"] = site_df["wind_t"].values
            portfolio_hourly["portfolio_solar_t"] = site_df["solar_t"].values
        else:
            # Merge and add production to existing totals
            portfolio_hourly = portfolio_hourly.merge(site_df, on="time", how="inner")
            portfolio_hourly["portfolio_wind_t"] += portfolio_hourly["wind_t"]
            portfolio_hourly["portfolio_solar_t"] += portfolio_hourly["solar_t"]
            # Clean up intermediate columns used for the merge
            portfolio_hourly.drop(columns=["wind_t", "solar_t", "total_t"], inplace=True)
            
    portfolio_hourly["portfolio_total_t"] = (
        portfolio_hourly["portfolio_wind_t"] + portfolio_hourly["portfolio_solar_t"]
    )
    return portfolio_hourly

def _aggregate_yearly(sites: Sequence[str], eval_dir: str) -> pd.DataFrame:
    """
    Aggregates yearly financial and operational metrics.
    
    Args:
        sites: List of site names to include.
        eval_dir: Path to the individual site CSVs.
        
    Returns:
        pd.DataFrame: A year-by-year summary of the portfolio performance.
    """
    portfolio_df = None
    num_sites = len(sites)
    
    for site in sites:
        site_df = _load_yearly(site, eval_dir)
        numeric_cols = site_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Don't aggregate the year column
        if "weather_year" in numeric_cols: 
            numeric_cols.remove("weather_year")

        if portfolio_df is None:
            # Initialize portfolio with first site
            portfolio_df = site_df[["weather_year"]].copy()
            for col in numeric_cols:
                portfolio_df[f"portfolio_{col}"] = site_df[col].astype(float).values
        else:
            # Join subsequent sites on weather_year and accumulate sums
            portfolio_df = portfolio_df.merge(
                site_df[["weather_year"] + numeric_cols], 
                on="weather_year", 
                how="inner"
            )
            for col in numeric_cols:
                portfolio_df[f"portfolio_{col}"] += portfolio_df[col]
                portfolio_df.drop(columns=[col], inplace=True)

    # 1. Calculate Averages for Efficiency/Ratio Metrics
    for col_name in COLS_TO_AVERAGE:
        port_col = f"portfolio_{col_name}"
        if port_col in portfolio_df.columns:
            portfolio_df[port_col] = portfolio_df[port_col] / num_sites

    # 2. Cleanup: Remove columns that shouldn't exist at portfolio level
    cols_to_remove = [f"portfolio_{c}" for c in COLS_TO_DROP]
    portfolio_df.drop(
        columns=[c for c in cols_to_remove if c in portfolio_df.columns], 
        inplace=True
    )

    # 3. Handle specific mapping for AEP (Annual Energy Production)
    if "portfolio_AEP with degradation [GWh]" in portfolio_df.columns:
        portfolio_df["portfolio_AEP [GWh]"] = (
            portfolio_df["portfolio_AEP with degradation [GWh]"]
        )

    return portfolio_df.sort_values("weather_year").reset_index(drop=True)

def _risk_summary(portfolio_df: pd.DataFrame, sites: Sequence[str], 
                  eval_dir: str, baseline_site: Optional[str]) -> pd.DataFrame:
    """
    Calculates portfolio diversification benefit.
    
    Compares the standard deviation of portfolio revenue against a single 
    baseline site to quantify risk reduction (delta_sigma).
    """
    if baseline_site is None: 
        baseline_site = list(sites)[0]
        
    baseline_yearly = _load_yearly(baseline_site, eval_dir)
    
    port_rev = portfolio_df["portfolio_Revenues [MEuro]"].astype(float)
    base_rev = baseline_yearly["Revenues [MEuro]"].astype(float)

    summary = {
        "baseline_site": baseline_site,
        "mean_portfolio_revenue_MEuro": float(port_rev.mean()),
        "std_portfolio_revenue_MEuro": float(port_rev.std(ddof=1)),
        # Percentage reduction in standard deviation (volatility)
        "delta_sigma_pct": (
            100.0 * (base_rev.std() - port_rev.std()) / base_rev.std() 
            if base_rev.std() != 0 else 0
        )
    }
    return pd.DataFrame([summary])

# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

def run_all_portfolios(eval_dir: str, output_dir: str, 
                       portfolios: Dict[str, Sequence[str]], 
                       baseline_site: Optional[str]):
    """
    Iterates through the PORTFOLIOS dictionary and generates all output files.
    """
    sites = available_sites(eval_dir)
    for name, p_sites in portfolios.items():
        print(f"Building Portfolio: {name}...")
        
        # Validate that all requested sites exist
        missing = [s for s in p_sites if s not in sites]
        if missing:
            print(f"  [Warning] Missing site data for: {missing}")
            continue

        # Execute aggregation layers
        yearly = _aggregate_yearly(p_sites, eval_dir)
        hourly = _aggregate_hourly(p_sites, eval_dir)
        summary = _risk_summary(yearly, p_sites, eval_dir, baseline_site)
        
        # Persistence
        os.makedirs(output_dir, exist_ok=True)
        hourly.to_csv(os.path.join(output_dir, f"{name}_hourly.csv"), index=False)
        yearly.to_csv(os.path.join(output_dir, f"{name}_yearly.csv"), index=False)
        summary.to_csv(os.path.join(output_dir, f"{name}_summary.csv"), index=False)
        
        print(f"  [OK] Saved {name} outputs to {output_dir}")

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build HPP portfolios from site results.")
    parser.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR, 
                        help="Path to original evaluation results.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, 
                        help="Path to save portfolio results.")
    args = parser.parse_args()

    run_all_portfolios(
        os.path.abspath(args.eval_dir),
        os.path.abspath(args.output_dir),
        PORTFOLIOS,
        None
    )

if __name__ == "__main__":
    main()