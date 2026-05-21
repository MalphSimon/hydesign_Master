import argparse
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from pandas import col
from pyparsing import col
import seaborn as sns

# ---------------------------------------------------------------------------
# 1. NAME MAPPING & CONFIGURATIONS
# ---------------------------------------------------------------------------

SITE_DISPLAY_MAP = {
    "Sud_Atlantique": "Sud Atlantique",
    "MidEurope": "Mid-Europe",
    "NorthSouth": "North-South Europe",
    "All": "All Sites",
    "Best": "Best Sites",
}

UNIFORM_COLOR = "#9C7667"

METRICS = [
    {"key": "npv", "title": "NPV", "ylabel": "M EUR", "candidates": ["NPV [MEuro]", "NPV"]},
    {"key": "npv_capex", "title": "NPV/CAPEX", "ylabel": "%", "candidates": ["NPV_over_CAPEX", "NPV/CAPEX"]},
    {"key": "irr", "title": "IRR", "ylabel": "%", "candidates": ["IRR", "IRR [%]"]},
]

SITES_TO_PLOT = [
    "Sud_Atlantique",
    "MidEurope",
    "NorthSouth",
    "All",
    "Best",
        
]

INPUT_DIR_DEFAULT = os.path.join("HPP", "Portfolio", "Outputs") 
OUTPUT_DIR_DEFAULT = os.path.join("HPP", "Portfolio", "Outputs", "Plots")

# ---------------------------------------------------------------------------
# 2. SHARED HELPERS
# ---------------------------------------------------------------------------

def _find_column(df, candidates):
    for col in candidates:
        if col in df.columns: return col
    return None

def _prepare_year_axis(df):
    if "weather_year" in df.columns:
        years = pd.to_numeric(df["weather_year"], errors="coerce")
    else:
        years = pd.Series(np.arange(len(df)), index=df.index, dtype=float)
    return np.arange(len(df)), years

def _get_npv_column(df):
    for cfg in METRICS:
        if cfg['key'] == 'npv':
            return _find_column(df, cfg["candidates"])
    return None

# ---------------------------------------------------------------------------
# 3. PLOTTING FUNCTIONS
# ---------------------------------------------------------------------------



def save_single_site_triple_stack(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    fname = os.path.basename(csv_path)
    # Extract site_id from filename by removing yearly/hourly/eval patterns and extension
    site_id = fname.split("_yearly")[0].split("_hourly")[0].split("_eval_")[0]
    
    display_name = SITE_DISPLAY_MAP.get(site_id, site_id)
    x, labels = _prepare_year_axis(df)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True, sharex=True)
    site_stats = {"Site": display_name, "site_id": site_id}

    for i, cfg in enumerate(METRICS):
        col = _find_column(df, cfg["candidates"])
        if col:
            data = pd.to_numeric(df[col], errors="coerce").dropna()
            y_plot = data * 100 if cfg["ylabel"] == "%" else data
            
            mean_val = y_plot.mean()
            std_val = y_plot.std()
            
            if cfg['key'] == 'npv':
                label_str = f"Mean: {mean_val:.2f} {cfg['ylabel']}\nStd: {std_val:.2f}"
                site_stats["NPV Std Dev"] = std_val
                site_stats["Mean NPV"] = mean_val
            else:
                label_str = f"Mean: {mean_val:.2f} {cfg['ylabel']}"
                site_stats[f"Mean {cfg['title']}"] = mean_val

            axes[i].plot(x, y_plot, marker="o", linewidth=2.2, color=UNIFORM_COLOR, label=label_str)
            axes[i].set_ylabel(f"{cfg['title']} [{cfg['ylabel']}]", fontweight='bold', fontsize=12)
            axes[i].grid(True, alpha=0.2, linestyle='--')
            axes[i].set_title(f"{cfg['title']}", loc='left', fontsize=11, fontweight='bold', alpha=0.7)
            axes[i].legend(loc="upper right", frameon=True, fontsize=10)

    axes[2].set_xticks(x)
    year_num = pd.to_numeric(labels, errors="coerce")
    axes[2].set_xticklabels([f"{int(v) % 100:02d}" if pd.notna(v) else "" for v in year_num])
    axes[2].set_xlabel("Scenario Year", fontweight='bold', fontsize=12)

    plt.suptitle(f"Financial Performance & Volatility: {display_name}", fontsize=17, fontweight='bold')
    fig.savefig(os.path.join(output_dir, f"{site_id}_TripleStack_Summary.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    df['site_display'] = display_name 
    return site_stats, df

def save_npv_distribution_boxplot(all_site_data_frames, output_dir):
    plt.figure(figsize=(12, 7))
    combined_df = pd.concat(all_site_data_frames, ignore_index=True)
    npv_col = _get_npv_column(combined_df)
    if npv_col:
        sns.boxplot(data=combined_df, x='site_display', y=npv_col, palette='viridis', width=0.6)
        sns.stripplot(data=combined_df, x='site_display', y=npv_col, color='black', size=3, jitter=0.2, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Portfolio", fontweight='bold')
        plt.ylabel("NPV (M EUR)", fontweight='bold')
        plt.title("NPV Distribution Across Weather Years", fontsize=16, fontweight='bold')
        plt.axhline(0, color='red', lw=1.5, ls='--')
        plt.savefig(os.path.join(output_dir, "NPV_Volatility_Distribution_Boxplot_Portfolio.png"), dpi=200, bbox_inches='tight')
    plt.close()

def save_npv_capex_distribution_boxplot(all_site_data_frames, output_dir):
    combined_df = pd.concat(all_site_data_frames, ignore_index=True)
    target_col = _find_column(combined_df, ["NPV_over_CAPEX", "NPV/CAPEX"])
    if not target_col: return

    plt.figure(figsize=(12, 7))
    site_colors_6 = ["#43D1D9", "#4B86C2", "#9C7667", "#68A357", "#D4A373", "#B07BA1"]
    ax = sns.boxplot(data=combined_df, x='site_display', y=target_col, palette=site_colors_6, width=0.5, fliersize=0)
    sns.stripplot(data=combined_df, x='site_display', y=target_col, color='black', size=4, jitter=0.15, alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0 if combined_df[target_col].max() < 2 else 100.0))
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.xlabel("Portfolio", fontweight='bold')
    plt.ylabel("NPV / CAPEX [%]", fontweight='bold')
    plt.title("Distribution of NPV/CAPEX Across Scenario Years", fontsize=14, fontweight='bold')
    plt.axhline(0, color='red', lw=1.2, ls='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, "NPV_CAPEX_Boxplot_Percentage_Portfolio.png"), dpi=300, bbox_inches='tight')
    plt.close()

def save_dscr_distribution_boxplot(all_site_data_frames, output_dir):
    combined_df = pd.concat(all_site_data_frames, ignore_index=True)
    target_col = _find_column(combined_df, ["DSCR P90 [-]"])
    if not target_col: return

    plt.figure(figsize=(12, 7))
    site_colors_6 = ["#43D1D9", "#4B86C2", "#9C7667", "#68A357", "#D4A373", "#B07BA1"]
    # site_display is already set with correct display names from SITE_DISPLAY_MAP
    ax = sns.boxplot(data=combined_df, x='site_display', y=target_col, palette=site_colors_6, width=0.5, fliersize=0)
    sns.stripplot(data=combined_df, x='site_display', y=target_col, color='black', size=4, jitter=0.15, alpha=0.3)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.xlabel("Portfolio", fontweight='bold')
    plt.ylabel("DSCR P90", fontweight='bold')
    plt.title("Distribution of DSCR P90 Across Scenario Years", fontsize=14, fontweight='bold')
    plt.axhline(1.20, color='red', lw=1.2, ls='--', alpha=0.6, label='Target DSCR (1.20)')
    plt.legend(loc='upper right', frameon=True)
    plt.savefig(os.path.join(output_dir, "DSCR_Distribution_Boxplot_Portfolio.png"), dpi=300, bbox_inches='tight')
    plt.close()

def save_multi_site_comparison(site_list, input_dir, output_dir):
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    site_colors = ["#43D1D9", "#4B86C2", "#9C7667", "#68A357", "#D4A373", "#B07BA1"]
    site_legend_handles = []

    for idx, site_key in enumerate(site_list):
        search_pattern = os.path.join(input_dir, f"{site_key}_yearly*.csv")
        all_files = glob.glob(search_pattern)
        files = [f for f in all_files if "_hourly" not in f]
        
        if not files: continue
        df = pd.read_csv(files[0])
        x, labels = _prepare_year_axis(df)
        color = site_colors[idx % len(site_colors)]
        display_name = SITE_DISPLAY_MAP.get(site_key, site_key)

        site_legend_handles.append(mlines.Line2D([], [], color=color, marker='o', label=display_name))

        for i, cfg in enumerate(METRICS):
            col = _find_column(df, cfg["candidates"])
            if col:
                data = pd.to_numeric(df[col], errors="coerce").dropna()
                y_plot = data * 100 if cfg["ylabel"] == "%" else data
                
                mean_val = y_plot.mean()
                std_val = y_plot.std()
                
                if cfg['key'] == 'npv':
                    label_str = f"Mean: {mean_val:.1f} {cfg['ylabel']}\nStd: {std_val:.1f}"
                else:
                    label_str = f"Mean: {mean_val:.2f} {cfg['ylabel']}"
                
                axes[i].plot(x, y_plot, marker="o", markersize=4, color=color, alpha=0.7, label=label_str)
                axes[i].set_title(cfg['title'], loc='left', fontweight='bold', fontsize=12, pad=-12)
                axes[i].set_ylabel(f"[{cfg['ylabel']}]", fontweight='bold')
    
    # Add legend to each subplot
    for i in range(len(METRICS)):
        axes[i].legend(loc="upper right", frameon=True, fontsize=9)
        axes[i].grid(True, alpha=0.2, linestyle='--')

    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"{int(v) % 100:02d}" if pd.notna(v) else "" for v in pd.to_numeric(labels, errors="coerce")])
    axes[2].set_xlabel("Scenario Year", fontweight='bold', fontsize=12)
    fig.legend(handles=site_legend_handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.01))
    plt.suptitle("Financial Performance Comparison", fontsize=16, fontweight='bold', y=0.92)
    plt.savefig(os.path.join(output_dir, "Financial_Comparison_Portfolio.png"), dpi=300, bbox_inches='tight')
    plt.close()

def save_multi_site_dscr_yearly(site_list, input_dir, output_dir):
    fig, ax = plt.subplots(figsize=(15, 8))
    site_colors = ["#43D1D9", "#4B86C2", "#9C7667", "#68A357", "#D4A373", "#B07BA1"]
    site_legend_handles = []
    mean_legend_handles = []

    for idx, site_key in enumerate(site_list):
        search_pattern = os.path.join(input_dir, f"{site_key}_yearly*.csv")
        all_files = glob.glob(search_pattern)
        files = [f for f in all_files if "_hourly" not in f]
        
        if not files: continue
        df = pd.read_csv(files[0])
        x, labels = _prepare_year_axis(df)
        color = site_colors[idx % len(site_colors)]
        display_name = SITE_DISPLAY_MAP.get(site_key, site_key)

        dscr_col = _find_column(df, ["DSCR [-]"])
        if dscr_col:
            data = pd.to_numeric(df[dscr_col], errors="coerce")
            mean_dscr = data.mean()
            
            ax.plot(x, data, marker="o", markersize=6, color=color, alpha=0.8, linewidth=2.2, label=display_name)
            site_legend_handles.append(mlines.Line2D([], [], color=color, marker='o', markersize=8, label=display_name, linewidth=2.2))
            mean_legend_handles.append(mlines.Line2D([], [], color=color, marker='o', markersize=8, label=f"Mean: {mean_dscr:.2f} %", linewidth=2.2))

    ax.axhline(1.20, color='red', lw=1.5, ls='--', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v) % 100:02d}" if pd.notna(v) else "" for v in pd.to_numeric(labels, errors="coerce")])
    ax.set_xlabel("Scenario Year", fontweight='bold', fontsize=12)
    ax.set_ylabel("DSCR [-]", fontweight='bold', fontsize=12)
    ax.set_title("DSCR Yearly Comparison Across Sites", fontsize=16, fontweight='bold', loc='center')
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Create mean legend in top right corner
    leg1 = ax.legend(handles=mean_legend_handles, loc='upper right', frameon=True, fontsize=10, labelspacing=1.2)
    ax.add_artist(leg1)
    
    # Create site legend at bottom center
    ax.legend(handles=site_legend_handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.15), frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "DSCR_Yearly_Comparison_Portfolio.png"), dpi=300, bbox_inches='tight')
    plt.close()

def save_multi_site_dscr_p90_yearly(site_list, input_dir, output_dir):
    fig, ax = plt.subplots(figsize=(15, 8))
    site_colors = ["#43D1D9", "#4B86C2", "#9C7667", "#68A357", "#D4A373", "#B07BA1"]
    site_legend_handles = []
    mean_legend_handles = []

    for idx, site_key in enumerate(site_list):
        search_pattern = os.path.join(input_dir, f"{site_key}_yearly*.csv")
        all_files = glob.glob(search_pattern)
        files = [f for f in all_files if "_hourly" not in f]
        
        if not files: continue
        df = pd.read_csv(files[0])
        x, labels = _prepare_year_axis(df)
        color = site_colors[idx % len(site_colors)]
        display_name = SITE_DISPLAY_MAP.get(site_key, site_key)

        dscr_col = _find_column(df, ["DSCR P90 [-]"])
        if dscr_col:
            data = pd.to_numeric(df[dscr_col], errors="coerce")
            mean_dscr = data.mean()
            
            ax.plot(x, data, marker="o", markersize=6, color=color, alpha=0.8, linewidth=2.2, label=display_name)
            site_legend_handles.append(mlines.Line2D([], [], color=color, marker='o', markersize=8, label=display_name, linewidth=2.2))
            mean_legend_handles.append(mlines.Line2D([], [], color=color, marker='o', markersize=8, label=f"Mean: {mean_dscr:.2f} %", linewidth=2.2))

    ax.axhline(1.20, color='red', lw=1.5, ls='--', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v) % 100:02d}" if pd.notna(v) else "" for v in pd.to_numeric(labels, errors="coerce")])
    ax.set_xlabel("Scenario Year", fontweight='bold', fontsize=12)
    ax.set_ylabel("DSCR P90", fontweight='bold', fontsize=12)
    ax.set_title("DSCR P90 Yearly Comparison Across Sites", fontsize=16, fontweight='bold', loc='center')
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Create mean legend in top right corner
    leg1 = ax.legend(handles=mean_legend_handles, loc='upper right', frameon=True, fontsize=10, labelspacing=1.2)
    ax.add_artist(leg1)
    
    # Create site legend at bottom center
    ax.legend(handles=site_legend_handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.15), frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "DSCR_P90_Yearly_Comparison_Portfolio.png"), dpi=300, bbox_inches='tight')
    plt.close()

def save_bankability_metrics_table(site_list, input_dir, output_dir):
    """Print bankability metrics summary table to terminal with Mean values."""
    bankability_cols = ["DSCR Breach Years", "Debt Headroom [MEuro]", "Debt Headroom [% of CAPEX]"]
    
    results = []
    for site_key in site_list:
        search_pattern = os.path.join(input_dir, f"{site_key}_yearly*.csv")
        all_files = glob.glob(search_pattern)
        files = [f for f in all_files if "_hourly" not in f]
        
        if not files: continue
        
        df = pd.read_csv(files[0])
        display_name = SITE_DISPLAY_MAP.get(site_key, site_key)
        
        # Calculate statistics for each metric
        row = {"Site": display_name}
        
        # Add DSCR P90 mean first (mean of the scenario years)
        dscr_p90_col = _find_column(df, ["DSCR P90 [-]"])
        if dscr_p90_col:
            dscr_p90_data = pd.to_numeric(df[dscr_p90_col], errors="coerce")
            row["DSCR P90 (Mean)"] = dscr_p90_data.mean()
        
        # Add other bankability metrics
        for col in bankability_cols:
            if col in df.columns:
                data = pd.to_numeric(df[col], errors="coerce")
                # Multiply Debt Headroom % by 100 to convert to percent
                if col == "Debt Headroom [% of CAPEX]":
                    data = data * 100
                row[f"{col} (Mean)"] = data.mean()
        
        results.append(row)
    
    # Create summary dataframe
    summary_df = pd.DataFrame(results)
    
    # Transpose table: metrics as rows, sites as columns
    summary_df = summary_df.set_index("Site").T
    
    # Format all values to 1 decimal place
    summary_df = summary_df.round(2)
    
    # Print to terminal with nice formatting
    print("\n" + "="*120)
    print("BANKABILITY METRICS SUMMARY (Mean)")
    print("="*120)
    print(summary_df.to_string())
    print("="*120 + "\n")
    
    # Also save as CSV for reference
    summary_df.to_csv(os.path.join(output_dir, "Bankability_Metrics_Summary_Mean.csv"))

def save_bankability_metrics_table_p90(site_list, input_dir, output_dir):
    """Print bankability metrics summary table to terminal with P90 values."""
    bankability_cols = ["DSCR Breach Years", "Debt Headroom [MEuro]", "Debt Headroom [% of CAPEX]"]
    
    results = []
    for site_key in site_list:
        search_pattern = os.path.join(input_dir, f"{site_key}_yearly*.csv")
        all_files = glob.glob(search_pattern)
        files = [f for f in all_files if "_hourly" not in f]
        
        if not files: continue
        
        df = pd.read_csv(files[0])
        display_name = SITE_DISPLAY_MAP.get(site_key, site_key)
        
        # Calculate statistics for each metric
        row = {"Site": display_name}
        
        # Add DSCR P90 mean first (mean of the scenario years)
        dscr_p90_col = _find_column(df, ["DSCR P90 [-]"])
        if dscr_p90_col:
            dscr_p90_data = pd.to_numeric(df[dscr_p90_col], errors="coerce")
            row["DSCR P90 (P90)"] = np.percentile(dscr_p90_data.dropna(), 10)
        
        # Add other bankability metrics
        for col in bankability_cols:
            if col in df.columns:
                data = pd.to_numeric(df[col], errors="coerce")
                # Multiply Debt Headroom % by 100 to convert to percent
                if col == "Debt Headroom [% of CAPEX]":
                    data = data * 100
                row[f"{col} (P90)"] = np.percentile(data.dropna(), 10)
        
        results.append(row)
    
    # Create summary dataframe
    summary_df = pd.DataFrame(results)
    
    # Transpose table: metrics as rows, sites as columns
    summary_df = summary_df.set_index("Site").T
    
    # Format all values to 1 decimal place
    summary_df = summary_df.round(2)
    
    # Print to terminal with nice formatting
    print("\n" + "="*120)
    print("BANKABILITY METRICS SUMMARY (P90 - Conservative Estimates)")
    print("="*120)
    print(summary_df.to_string())
    print("="*120 + "\n")
    
    # Also save as CSV for reference
    summary_df.to_csv(os.path.join(output_dir, "Bankability_Metrics_Summary_P90.csv"))

def save_financial_metrics_table(site_list, input_dir, output_dir):
    """Print financial metrics summary table to terminal with Mean values."""
    results = []
    for site_key in site_list:
        search_pattern = os.path.join(input_dir, f"{site_key}_yearly*.csv")
        all_files = glob.glob(search_pattern)
        files = [f for f in all_files if "_hourly" not in f]
        
        if not files: continue
        
        df = pd.read_csv(files[0])
        display_name = SITE_DISPLAY_MAP.get(site_key, site_key)
        
        # Calculate statistics for each metric
        row = {"Site": display_name}
        
        # NPV metrics (Mean)
        if "NPV [MEuro]" in df.columns:
            npv_data = pd.to_numeric(df["NPV [MEuro]"], errors="coerce").dropna()
            row["Mean NPV [M€]"] = npv_data.mean()
        
        # NPV/CAPEX (Mean)
        if "NPV_over_CAPEX" in df.columns:
            npv_capex_data = pd.to_numeric(df["NPV_over_CAPEX"], errors="coerce").dropna()
            npv_capex_data = npv_capex_data * 100  # Convert to percent
            row["Mean NPV/CAPEX [%]"] = npv_capex_data.mean()
        
        # IRR (Mean)
        if "IRR" in df.columns:
            irr_data = pd.to_numeric(df["IRR"], errors="coerce").dropna()
            row["Mean IRR [%]"] = irr_data.mean() * 100  # Convert to percent
        
        results.append(row)
    
    # Create summary dataframe
    summary_df = pd.DataFrame(results)
    
    # Transpose table: metrics as rows, sites as columns
    summary_df = summary_df.set_index("Site").T
    
    # Format all values to 2 decimal places
    summary_df = summary_df.round(2)
    
    # Print to terminal with nice formatting
    print("\n" + "="*120)
    print("FINANCIAL METRICS SUMMARY (Mean)")
    print("="*120)
    print(summary_df.to_string())
    print("="*120 + "\n")
    
    # Also save as CSV for reference
    summary_df.to_csv(os.path.join(output_dir, "Financial_Metrics_Summary_Mean.csv"))

def save_financial_metrics_table_p90(site_list, input_dir, output_dir):
    """Print financial metrics summary table to terminal with P90 values."""
    results = []
    for site_key in site_list:
        search_pattern = os.path.join(input_dir, f"{site_key}_yearly*.csv")
        all_files = glob.glob(search_pattern)
        files = [f for f in all_files if "_hourly" not in f]
        
        if not files: continue
        
        df = pd.read_csv(files[0])
        display_name = SITE_DISPLAY_MAP.get(site_key, site_key)
        
        # Calculate statistics for each metric
        row = {"Site": display_name}
        
        # NPV metrics (P90)
        if "NPV [MEuro]" in df.columns:
            npv_data = pd.to_numeric(df["NPV [MEuro]"], errors="coerce").dropna()
            row["P90 NPV [M€]"] = np.percentile(npv_data, 10)
        
        # NPV/CAPEX (P90)
        if "NPV_over_CAPEX" in df.columns:
            npv_capex_data = pd.to_numeric(df["NPV_over_CAPEX"], errors="coerce").dropna()
            npv_capex_data = npv_capex_data * 100  # Convert to percent
            row["P90 NPV/CAPEX [%]"] = np.percentile(npv_capex_data, 10)
        
        # IRR (P90)
        if "IRR" in df.columns:
            irr_data = pd.to_numeric(df["IRR"], errors="coerce").dropna()
            row["P90 IRR [%]"] = np.percentile(irr_data * 100, 10)  # Convert to percent
        
        results.append(row)
    
    # Create summary dataframe
    summary_df = pd.DataFrame(results)
    
    # Transpose table: metrics as rows, sites as columns
    summary_df = summary_df.set_index("Site").T
    
    # Format all values to 2 decimal places
    summary_df = summary_df.round(2)
    
    # Print to terminal with nice formatting
    print("\n" + "="*120)
    print("FINANCIAL METRICS SUMMARY (P90 - Conservative Estimates)")
    print("="*120)
    print(summary_df.to_string())
    print("="*120 + "\n")
    
    # Also save as CSV for reference
    summary_df.to_csv(os.path.join(output_dir, "Financial_Metrics_Summary_P90.csv"))

def _calculate_iqr_statistics(npv_data, npv_capex_data, irr_data):
    """Calculate IQR statistics (Q1, Q2/Median, Q3, IQR) for each metric."""
    results = []
    
    # NPV IQR Analysis
    npv_clean = pd.to_numeric(npv_data, errors="coerce").dropna()
    if not npv_clean.empty:
        q1_npv = npv_clean.quantile(0.25)
        q2_npv = npv_clean.quantile(0.50)  # median
        q3_npv = npv_clean.quantile(0.75)
        iqr_npv = q3_npv - q1_npv
        
        results.append({
            "Metric": "NPV [MEuro]",
            "Q1 (25th)": q1_npv,
            "Q2 Median": q2_npv,
            "Q3 (75th)": q3_npv,
            "IQR (Q3-Q1)": iqr_npv,
            "IQR % of Median": (iqr_npv / abs(q2_npv) * 100) if q2_npv != 0 else np.nan,
        })
    
    # NPV/CAPEX IQR Analysis
    npv_capex_clean = pd.to_numeric(npv_capex_data, errors="coerce").dropna()
    if not npv_capex_clean.empty:
        q1_capex = npv_capex_clean.quantile(0.25)
        q2_capex = npv_capex_clean.quantile(0.50)  # median
        q3_capex = npv_capex_clean.quantile(0.75)
        iqr_capex = q3_capex - q1_capex
        
        results.append({
            "Metric": "NPV/CAPEX [%]",
            "Q1 (25th)": q1_capex * 100,
            "Q2 Median": q2_capex * 100,
            "Q3 (75th)": q3_capex * 100,
            "IQR (Q3-Q1)": iqr_capex * 100,
            "IQR % of Median": (iqr_capex / abs(q2_capex) * 100) if q2_capex != 0 else np.nan,
        })
    
    # IRR IQR Analysis
    irr_clean = pd.to_numeric(irr_data, errors="coerce").dropna()
    if not irr_clean.empty:
        q1_irr = irr_clean.quantile(0.25)
        q2_irr = irr_clean.quantile(0.50)  # median
        q3_irr = irr_clean.quantile(0.75)
        iqr_irr = q3_irr - q1_irr
        
        results.append({
            "Metric": "IRR [%]",
            "Q1 (25th)": q1_irr * 100,
            "Q2 Median": q2_irr * 100,
            "Q3 (75th)": q3_irr * 100,
            "IQR (Q3-Q1)": iqr_irr * 100,
            "IQR % of Median": (iqr_irr / abs(q2_irr) * 100) if q2_irr != 0 else np.nan,
        })
    
    return results

def save_single_site_iqr_boxplot(csv_path, output_dir):
    """Create detailed IQR boxplots for a single site showing the spread of each metric."""
    df = pd.read_csv(csv_path)
    fname = os.path.basename(csv_path)
    site_id = fname.split("_yearly")[0].split("_hourly")[0].split("_eval_")[0]
    display_name = SITE_DISPLAY_MAP.get(site_id, site_id)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    
    # Collect data for all metrics
    npv_col = _find_column(df, METRICS[0]["candidates"])
    npv_capex_col = _find_column(df, METRICS[1]["candidates"])
    irr_col = _find_column(df, METRICS[2]["candidates"])
    
    metrics_info = [
        (npv_col, METRICS[0], "NPV", "M EUR"),
        (npv_capex_col, METRICS[1], "NPV/CAPEX", "%"),
        (irr_col, METRICS[2], "IRR", "%"),
    ]
    
    colors = ["#9C7667", "#43D1D9", "#4B86C2"]
    
    for idx, (col, cfg, title, ylabel) in enumerate(metrics_info):
        ax = axes[idx]
        
        if col:
            data = pd.to_numeric(df[col], errors="coerce").dropna()
            if ylabel == "%":
                data = data * 100
            
            # Create boxplot
            bp = ax.boxplot([data], vert=True, patch_artist=True, widths=0.4,
                           boxprops=dict(facecolor=colors[idx], alpha=0.7),
                           medianprops=dict(color='red', linewidth=2),
                           whiskerprops=dict(color='black', linewidth=1.5),
                           capprops=dict(color='black', linewidth=1.5),
                           flierprops=dict(marker='o', markerfacecolor='gray', markersize=6, alpha=0.5))
            
            # Calculate quartiles and IQR
            q1 = data.quantile(0.25)
            q2 = data.quantile(0.50)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            
            # Add scatter points for individual values
            y_vals = data.values
            x_vals = np.random.normal(1, 0.04, size=len(y_vals))
            ax.scatter(x_vals, y_vals, alpha=0.5, s=50, color='black', zorder=3)
            
            # Add text annotations
            textstr = f"Q1: {q1:.2f}\nMedian: {q2:.2f}\nQ3: {q3:.2f}\nIQR: {iqr:.2f}\nIQR%: {(iqr/abs(q2)*100):.1f}%" if q2 != 0 else f"Q1: {q1:.2f}\nMedian: {q2:.2f}\nQ3: {q3:.2f}\nIQR: {iqr:.2f}"
            ax.text(1.3, q2, textstr, fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_ylabel(f"{title} [{ylabel}]", fontweight='bold', fontsize=11)
            ax.set_title(f"{title} IQR Spread", loc='left', fontsize=12, fontweight='bold', alpha=0.8)
            ax.set_xticks([])
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.suptitle(f"IQR Spread Analysis: {display_name}", fontsize=16, fontweight='bold')
    fig.savefig(os.path.join(output_dir, f"{site_id}_IQR_Boxplot_Spread.png"), dpi=200, bbox_inches='tight')
    plt.close()

def save_multi_site_iqr_comparison(site_list, input_dir, output_dir):
    """Create IQR comparison across all sites for each metric."""
    site_colors = ["#43D1D9", "#9C7667", "#4B86C2", "#68A357", "#D4A373", "#B07BA1"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    
    metrics_info = [
        (METRICS[0]["candidates"], METRICS[0]["title"], "M EUR", 0),
        (METRICS[1]["candidates"], METRICS[1]["title"], "%", 1),
        (METRICS[2]["candidates"], METRICS[2]["title"], "%", 2),
    ]
    
    iqr_data_list = []
    
    for idx, (candidates, metric_title, ylabel, ax_idx) in enumerate(metrics_info):
        ax = axes[ax_idx]
        site_iqr_data = []
        
        for site_idx, site_key in enumerate(site_list):
            search_pattern = os.path.join(input_dir, f"{site_key}_yearly*.csv")
            all_files = glob.glob(search_pattern)
            files = [f for f in all_files if "_hourly" not in f]
            
            if not files: continue
            
            df = pd.read_csv(files[0])
            col = _find_column(df, candidates)
            
            if col:
                data = pd.to_numeric(df[col], errors="coerce").dropna()
                if ylabel == "%":
                    data = data * 100
                
                q1 = data.quantile(0.25)
                q2 = data.quantile(0.50)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                
                # Plot IQR as range
                display_name = SITE_DISPLAY_MAP.get(site_key, site_key)
                y_pos = site_idx
                color = site_colors[site_idx % len(site_colors)]
                
                # Draw whiskers (full range)
                ax.plot([data.min(), data.max()], [y_pos, y_pos], 'k-', linewidth=0.8, alpha=0.3)
                
                # Draw IQR box
                ax.barh(y_pos, iqr, left=q1, height=0.6, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
                
                # Draw median line
                ax.plot([q2, q2], [y_pos - 0.3, y_pos + 0.3], 'r-', linewidth=2.5)
                
                # Add quartile labels
                ax.text(q1 - 0.02 * (data.max() - data.min()), y_pos, f'{q1:.1f}', 
                       ha='right', va='center', fontsize=8, fontweight='bold')
                ax.text(q3 + 0.02 * (data.max() - data.min()), y_pos, f'{q3:.1f}', 
                       ha='left', va='center', fontsize=8, fontweight='bold')
                
                site_iqr_data.append({
                    'Site': display_name,
                    'Metric': metric_title,
                    'Q1': q1,
                    'Median': q2,
                    'Q3': q3,
                    'IQR': iqr
                })
        
        ax.set_yticks(range(len(site_list)))
        ax.set_yticklabels([SITE_DISPLAY_MAP.get(s, s) for s in site_list], fontsize=9)
        ax.set_xlabel(f"{metric_title} [{ylabel}]", fontweight='bold', fontsize=11)
        ax.set_title(f"{metric_title} IQR Comparison", loc='left', fontsize=12, fontweight='bold', alpha=0.8)
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        iqr_data_list.append(site_iqr_data)
    
    plt.suptitle("IQR Spread Comparison Across All Sites", fontsize=16, fontweight='bold')
    fig.savefig(os.path.join(output_dir, "Multi_Site_IQR_Comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Save IQR comparison data to CSV
    if iqr_data_list and iqr_data_list[0]:
        iqr_combined = pd.concat([pd.DataFrame(d) for d in iqr_data_list], ignore_index=True)
        iqr_combined.to_csv(os.path.join(output_dir, "Multi_Site_IQR_Comparison.csv"), index=False)

def save_iqr_summary_table(site_list, input_dir, output_dir):
    """Print IQR summary table for all sites across all metrics."""
    results = []
    
    for site_key in site_list:
        search_pattern = os.path.join(input_dir, f"{site_key}_yearly*.csv")
        all_files = glob.glob(search_pattern)
        files = [f for f in all_files if "_hourly" not in f]
        
        if not files: continue
        
        df = pd.read_csv(files[0])
        display_name = SITE_DISPLAY_MAP.get(site_key, site_key)
        
        row = {"Site": display_name}
        
        # NPV IQR
        npv_col = _find_column(df, METRICS[0]["candidates"])
        if npv_col:
            npv_data = pd.to_numeric(df[npv_col], errors="coerce").dropna()
            iqr_npv = npv_data.quantile(0.75) - npv_data.quantile(0.25)
            row["NPV IQR [M€]"] = iqr_npv
        
        # NPV/CAPEX IQR
        npv_capex_col = _find_column(df, METRICS[1]["candidates"])
        if npv_capex_col:
            npv_capex_data = pd.to_numeric(df[npv_capex_col], errors="coerce").dropna() * 100
            iqr_capex = npv_capex_data.quantile(0.75) - npv_capex_data.quantile(0.25)
            row["NPV/CAPEX IQR [%]"] = iqr_capex
        
        # IRR IQR
        irr_col = _find_column(df, METRICS[2]["candidates"])
        if irr_col:
            irr_data = pd.to_numeric(df[irr_col], errors="coerce").dropna() * 100
            iqr_irr = irr_data.quantile(0.75) - irr_data.quantile(0.25)
            row["IRR IQR [%]"] = iqr_irr
        
        results.append(row)
    
    # Create summary dataframe
    summary_df = pd.DataFrame(results)
    
    # Transpose table: metrics as rows, sites as columns
    summary_df = summary_df.set_index("Site").T
    
    # Format all values to 2 decimal places
    summary_df = summary_df.round(2)
    
    # Print to terminal with nice formatting
    print("\n" + "="*140)
    print("IQR (INTERQUARTILE RANGE) SPREAD SUMMARY")
    print("Measures volatility and spread of performance metrics across weather scenarios")
    print("="*140)
    print(summary_df.to_string())
    print("="*140 + "\n")
    
    # Also save as CSV for reference
    summary_df.to_csv(os.path.join(output_dir, "IQR_Spread_Summary.csv"))

# ---------------------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=INPUT_DIR_DEFAULT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    all_site_stats = []
    processed_dataframes = [] 

    for site in SITES_TO_PLOT:
        search_pattern = os.path.join(args.input_dir, f"{site}_yearly*.csv")
        all_files = glob.glob(search_pattern)
        
        # FILTER: Exclude hourly files
        files = [f for f in all_files if "_hourly" not in f]
        
        for f in files:
            print(f"[Processing Summary] {os.path.basename(f)}")
            stats, cleaned_df = save_single_site_triple_stack(f, args.output_dir)
            all_site_stats.append(stats)
            processed_dataframes.append(cleaned_df)

    if all_site_stats:
        summary_df = pd.DataFrame(all_site_stats)
        summary_path = os.path.join(args.output_dir, "Financial_Summary_Statistics.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"[Visualizing] Risk-Return and Distributions...")
        #save_npv_distribution_boxplot(processed_dataframes, args.output_dir)
        #save_npv_capex_distribution_boxplot(processed_dataframes, args.output_dir)
        #save_dscr_distribution_boxplot(processed_dataframes, args.output_dir)
        #save_multi_site_dscr_yearly(SITES_TO_PLOT, args.input_dir, args.output_dir)
        #save_multi_site_dscr_p90_yearly(SITES_TO_PLOT, args.input_dir, args.output_dir)
        #save_bankability_metrics_table(SITES_TO_PLOT, args.input_dir, args.output_dir)
        #save_bankability_metrics_table_p90(SITES_TO_PLOT, args.input_dir, args.output_dir)
        #save_financial_metrics_table(SITES_TO_PLOT, args.input_dir, args.output_dir)
        #save_financial_metrics_table_p90(SITES_TO_PLOT, args.input_dir, args.output_dir)
        #save_multi_site_comparison(SITES_TO_PLOT, args.input_dir, args.output_dir)
        
        # Generate IQR analysis
        print(f"[Generating] IQR analysis and boxplots...")
        for site in SITES_TO_PLOT:
            search_pattern = os.path.join(args.input_dir, f"{site}_yearly*.csv")
            all_files = glob.glob(search_pattern)
            files = [f for f in all_files if "_hourly" not in f]
            for f in files:
                save_single_site_iqr_boxplot(f, args.output_dir)
        
        # Generate multi-site IQR comparison
        print(f"[Generating] Multi-site IQR comparison...")
        save_multi_site_iqr_comparison(SITES_TO_PLOT, args.input_dir, args.output_dir)
        
        # Generate IQR summary table
        print(f"[Generating] IQR summary table...")
        save_iqr_summary_table(SITES_TO_PLOT, args.input_dir, args.output_dir)

    print(f"Execution Complete. Plots saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()