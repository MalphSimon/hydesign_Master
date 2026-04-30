"""
Hybrid Power Plant (HPP) Portfolio Aggregator.

This script builds user-defined HPP portfolios by aggregating existing site outputs.
It performs three main types of data processing:
1. Summation: Combines Energy (GWh), Revenue (MEuro), and Capacity (MW).
2. Weighted/Recalculated Financials: Correctly handles NPV, IRR, and NPV/CAPEX.
3. Filtering: Removes geographic and site-specific metadata.
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
DEFAULT_EVAL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Evaluations", "HiFiEMS", "P25"))
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Outputs")

PORTFOLIOS: Dict[str, Sequence[str]] = {
    "Sud_Atlantique": ["Sud_Atlantique_HiFiEMS"],
    "MidEurope": ["Thetys_HiFiEMS", "Sud_Atlantique_HiFiEMS", "Golfe_du_Lion_HiFiEMS"],
    "NorthSouth": ["NordsoenMidt_HiFiEMS", "Sud_Atlantique_HiFiEMS", "SicilySouth_HiFiEMS"],
    "All": [
        "NordsoenMidt_HiFiEMS", "Golfe_du_Lion_HiFiEMS", "SicilySouth_HiFiEMS", 
        "Thetys_HiFiEMS", "Sud_Atlantique_HiFiEMS", "Vestavind_HiFiEMS"
    ],
}

COLS_TO_DROP = [
    "longitude", "latitude", "altitude", "surface_tilt [deg]", 
    "surface_azimuth [deg]", "cost_of_battery_P_fluct_in_peak_price_ratio", 
    "clearance [m]", "sp [W/m2]", "p_rated [MW]", "Nwt", 
    "wind_MW_per_km2 [MW/km2]", "solar_MW [MW]", "DC_AC_ratio", 
    "b_P [MW]", "b_E_h [h]", "penalty lifetime [MEuro]",
    "Rotor diam [m]", "Hub height [m]"
]

# Metrics that represent ratios or efficiencies; these are averaged across 
# sites rather than summed. 
# NOTE: NPV_over_CAPEX and IRR are handled separately for financial accuracy.
COLS_TO_AVERAGE = [
    "DC_AC_ratio", "b_E_h [h]", "LCOE [Euro/MWh]", 
    "GUF", "Break-even PPA price [Euro/MWh]", 
    "Capacity factor wind [-]", "Capacity factor solar [-]", "DSCR Breach Years", "DSCR [-]"
]

# ---------------------------------------------------------------------------
# Helper Functions (File I/O)
# ---------------------------------------------------------------------------

def _find_single_file(eval_dir: str, pattern: str) -> str:
    matches = sorted(glob.glob(os.path.join(eval_dir, pattern)))
    if not matches:
        raise FileNotFoundError(f"No file found for pattern: {pattern}")
    return matches[0]

def available_sites(eval_dir: str) -> List[str]:
    files = glob.glob(os.path.join(eval_dir, "*_eval_*.csv"))
    sites = sorted(set([os.path.basename(f).split("_eval_")[0] for f in files]))
    return sites

def _load_yearly(site: str, eval_dir: str) -> pd.DataFrame:
    all_matches = glob.glob(os.path.join(eval_dir, f"{site}_eval_*.csv"))
    yearly_paths = [f for f in all_matches if "_hourly.csv" not in f]
    if not yearly_paths:
        raise FileNotFoundError(f"No yearly file found for site: {site}")
    return pd.read_csv(yearly_paths[0])

# ---------------------------------------------------------------------------
# Aggregation Logic
# ---------------------------------------------------------------------------

def _aggregate_yearly(sites: Sequence[str], eval_dir: str) -> pd.DataFrame:
    portfolio_df = None
    num_sites = len(sites)
    
    for site in sites:
        site_df = _load_yearly(site, eval_dir)
        numeric_cols = site_df.select_dtypes(include=[np.number]).columns.tolist()
        if "weather_year" in numeric_cols: 
            numeric_cols.remove("weather_year")

        if portfolio_df is None:
            portfolio_df = site_df[["weather_year"]].copy()
            for col in numeric_cols:
                # Initialize IRR as (Value * CAPEX) for weighting
                if col == "IRR" and "CAPEX [MEuro]" in site_df.columns:
                    portfolio_df[col] = (site_df[col] * site_df["CAPEX [MEuro]"]).values
                else:
                    portfolio_df[col] = site_df[col].astype(float).values
        else:
            portfolio_df = portfolio_df.merge(
                site_df[["weather_year"] + numeric_cols], 
                on="weather_year", how="inner", suffixes=('', '_site')
            )
            for col in numeric_cols:
                if col == "IRR" and "CAPEX [MEuro]" in site_df.columns:
                    # Accumulate weighted values
                    portfolio_df[col] += (site_df[col] * site_df["CAPEX [MEuro]"]).values
                else:
                    # Standard summation (Correct for NPV [MEuro], Revenues, etc.)
                    portfolio_df[col] += portfolio_df[f"{col}_site"]
                portfolio_df.drop(columns=[f"{col}_site"], inplace=True)

    # 1. Recalculate Financial Ratios
    if "CAPEX [MEuro]" in portfolio_df.columns:
        # Finalize weighted IRR
        if "IRR" in portfolio_df.columns:
            portfolio_df["IRR"] = portfolio_df["IRR"] / portfolio_df["CAPEX [MEuro]"]
        
        # Correctly recalculate Profitability Index (NPV/CAPEX)
        if "NPV [MEuro]" in portfolio_df.columns:
            portfolio_df["NPV_over_CAPEX"] = portfolio_df["NPV [MEuro]"] / portfolio_df["CAPEX [MEuro]"]

    # 2. Simple Averages for non-financial ratios
    for col_name in COLS_TO_AVERAGE:
        if col_name in portfolio_df.columns:
            portfolio_df[col_name] = portfolio_df[col_name] / num_sites

    # 3. Cleanup
    portfolio_df.drop(columns=[c for c in COLS_TO_DROP if c in portfolio_df.columns], inplace=True)
    if "AEP with degradation [GWh]" in portfolio_df.columns:
        portfolio_df["AEP [GWh]"] = portfolio_df["AEP with degradation [GWh]"]

    return portfolio_df.sort_values("weather_year").reset_index(drop=True)

def _risk_summary(portfolio_df: pd.DataFrame, sites: Sequence[str], 
                 eval_dir: str, baseline_site: Optional[str]) -> pd.DataFrame:
    if baseline_site is None: 
        baseline_site = list(sites)[0]
    baseline_yearly = _load_yearly(baseline_site, eval_dir)
    port_rev = portfolio_df["Revenues [MEuro]"].astype(float)
    base_rev = baseline_yearly["Revenues [MEuro]"].astype(float)
    
    # NPV/CAPEX metrics
    port_npv_capex = portfolio_df["NPV_over_CAPEX"].astype(float) if "NPV_over_CAPEX" in portfolio_df.columns else None
    base_npv_capex = baseline_yearly["NPV_over_CAPEX"].astype(float) if "NPV_over_CAPEX" in baseline_yearly.columns else None

    summary = {
        "baseline_site": baseline_site,
        "mean_portfolio_revenue_MEuro": float(port_rev.mean()),
        "std_portfolio_revenue_MEuro": float(port_rev.std(ddof=1)),
        "delta_sigma_pct": (
            100.0 * (base_rev.std() - port_rev.std()) / base_rev.std() 
            if base_rev.std() != 0 else 0
        )
    }
    
    # Add NPV/CAPEX risk metrics if available
    if port_npv_capex is not None and base_npv_capex is not None:
        summary["mean_portfolio_npv_over_capex"] = float(port_npv_capex.mean())
        summary["std_portfolio_npv_over_capex"] = float(port_npv_capex.std(ddof=1))
        summary["delta_sigma_npv_capex_pct"] = (
            100.0 * (base_npv_capex.std() - port_npv_capex.std()) / base_npv_capex.std() 
            if base_npv_capex.std() != 0 else 0
        )
    
    return pd.DataFrame([summary])

def run_all_portfolios(eval_dir: str, output_dir: str, portfolios: Dict[str, Sequence[str]], baseline_site: Optional[str]):
    sites = available_sites(eval_dir)
    for name, p_sites in portfolios.items():
        print(f"Building Portfolio: {name}...")
        missing = [s for s in p_sites if s not in sites]
        if missing:
            print(f"  [Warning] Missing site data for: {missing}")
            continue

        yearly = _aggregate_yearly(p_sites, eval_dir)
        summary = _risk_summary(yearly, p_sites, eval_dir, baseline_site)
        
        os.makedirs(output_dir, exist_ok=True)
        yearly.to_csv(os.path.join(output_dir, f"{name}_yearly.csv"), index=False)
        summary.to_csv(os.path.join(output_dir, f"{name}_summary.csv"), index=False)
        print(f"  [OK] Saved {name} outputs to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Build HPP portfolios from site results.")
    parser.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    run_all_portfolios(os.path.abspath(args.eval_dir), os.path.abspath(args.output_dir), PORTFOLIOS, None)

if __name__ == "__main__":
    main()