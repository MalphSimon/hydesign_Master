"""
Generate a financial performance indicators table for portfolios.
Similar to Table 6 format.
"""

import os
import pandas as pd
import glob

def generate_portfolio_summary_table(eval_dir: str, output_dir: str) -> pd.DataFrame:
    """
    Generate a summary table with financial performance indicators for each portfolio.
    
    Reads the yearly portfolio data and generates mean, CV, and risk metrics.
    CV (Coefficient of Variation) = std / |mean|
    """
    
    # Find all portfolio yearly files
    yearly_files = glob.glob(os.path.join(output_dir, "*_yearly.csv"))
    
    results = []
    
    for yearly_file in sorted(yearly_files):
        portfolio_name = os.path.basename(yearly_file).replace("_yearly.csv", "")
        df = pd.read_csv(yearly_file)
        
        # Extract key metrics
        mean_npv = df["NPV [MEuro]"].mean() if "NPV [MEuro]" in df.columns else None
        npv_std = df["NPV [MEuro]"].std(ddof=1) if "NPV [MEuro]" in df.columns else None
        # CV = std / |mean| to handle negative values
        npv_cv = (npv_std / abs(mean_npv)) if mean_npv and mean_npv != 0 else None
        
        mean_npv_capex = df["NPV_over_CAPEX"].mean() if "NPV_over_CAPEX" in df.columns else None
        npv_capex_std = df["NPV_over_CAPEX"].std(ddof=1) if "NPV_over_CAPEX" in df.columns else None
        # CV for NPV/CAPEX
        npv_capex_cv = (npv_capex_std / abs(mean_npv_capex)) if mean_npv_capex and mean_npv_capex != 0 else None
        
        mean_irr = df["IRR"].mean() if "IRR" in df.columns else None
        mean_revenue = df["Revenues [MEuro]"].mean() if "Revenues [MEuro]" in df.columns else None
        revenue_std = df["Revenues [MEuro]"].std(ddof=1) if "Revenues [MEuro]" in df.columns else None
        revenue_cv = (revenue_std / mean_revenue) if mean_revenue and mean_revenue != 0 else None
        
        results.append({
            "Portfolio": portfolio_name,
            "Mean NPV [M€]": mean_npv,
            "NPV CV": npv_cv,
            "Mean NPV/CAPEX [%]": mean_npv_capex * 100 if mean_npv_capex else None,
            "NPV/CAPEX CV": npv_capex_cv,
            "Mean IRR [%]": mean_irr * 100 if mean_irr else None,
            "Mean Revenue [M€]": mean_revenue,
            "Revenue CV": revenue_cv,
        })
    
    summary_df = pd.DataFrame(results)
    return summary_df.set_index("Portfolio")

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_EVAL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Evaluations", "HiFiEMS", "P20"))
    DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Outputs")
    
    # Generate summary table
    summary_table = generate_portfolio_summary_table(DEFAULT_EVAL_DIR, DEFAULT_OUTPUT_DIR)
    
    # Save to CSV (unrounded)
    output_file = os.path.join(DEFAULT_OUTPUT_DIR, "portfolio_summary_table.csv")
    summary_table.to_csv(output_file)
    print(f"Summary table saved to: {output_file}\n")
    
    # Display formatted table with rounded values
    print("=" * 140)
    print("Portfolio Financial Performance Indicators")
    print("=" * 140)
    
    # Format for display
    display_df = summary_table.copy()
    display_df["Mean NPV [M€]"] = display_df["Mean NPV [M€]"].round(2)
    display_df["NPV CV"] = display_df["NPV CV"].round(2)
    display_df["Mean NPV/CAPEX [%]"] = display_df["Mean NPV/CAPEX [%]"].round(2)
    display_df["NPV/CAPEX CV"] = display_df["NPV/CAPEX CV"].round(2)
    display_df["Mean IRR [%]"] = display_df["Mean IRR [%]"].round(2)
    display_df["Mean Revenue [M€]"] = display_df["Mean Revenue [M€]"].round(2)
    display_df["Revenue CV"] = display_df["Revenue CV"].round(2)
    
    print(display_df.to_string())
    print("=" * 140)
    print("\nNote: CV = Coefficient of Variation (std / |mean|)")
    print("NPV/CAPEX CV indicates the volatility of returns relative to invested capital.")

if __name__ == "__main__":
    main()
