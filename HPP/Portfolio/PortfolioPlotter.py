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
    "NorthSouth": "North-South",
    "All": "All Sites",
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
    site_id = fname.split("_eval_")[0]
    
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
                cv_val = (std_val / abs(mean_val)) if abs(mean_val) > 1e-6 else 0
                label_str = f"Mean: {mean_val:.2f} {cfg['ylabel']}\nCV: {cv_val:.2f}"
                site_stats["NPV CV"] = cv_val
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
        plt.xlabel("Site Location", fontweight='bold')
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
    plt.xlabel("Site Location", fontweight='bold')
    plt.ylabel("NPV / CAPEX [%]", fontweight='bold')
    plt.title("Distribution of NPV/CAPEX Across Scenario Years", fontsize=14, fontweight='bold')
    plt.axhline(0, color='red', lw=1.2, ls='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, "NPV_CAPEX_Boxplot_Percentage_Portfolio.png"), dpi=300, bbox_inches='tight')
    plt.close()

def save_dscr_distribution_boxplot(all_site_data_frames, output_dir):
    combined_df = pd.concat(all_site_data_frames, ignore_index=True)
    target_col = _find_column(combined_df, ["DSCR [-]"])
    if not target_col: return

    plt.figure(figsize=(12, 7))
    site_colors_6 = ["#43D1D9", "#4B86C2", "#9C7667", "#68A357", "#D4A373", "#B07BA1"]
    ax = sns.boxplot(data=combined_df, x='site_display', y=target_col, palette=site_colors_6, width=0.5, fliersize=0)
    sns.stripplot(data=combined_df, x='site_display', y=target_col, color='black', size=4, jitter=0.15, alpha=0.3)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.xlabel("Site Location", fontweight='bold')
    plt.ylabel("DSCR [-]", fontweight='bold')
    plt.title("Distribution of DSCR Across Scenario Years", fontsize=14, fontweight='bold')
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
                    cv_val = (std_val / abs(mean_val)) if abs(mean_val) > 1e-6 else 0
                    label_str = f"Mean: {mean_val:.1f} {cfg['ylabel']}\nCV: {cv_val:.1f}"
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
    fig.legend(handles=site_legend_handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01))
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
    ax.legend(handles=site_legend_handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.15), frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "DSCR_Yearly_Comparison_Portfolio.png"), dpi=300, bbox_inches='tight')
    plt.close()

def save_bankability_metrics_table(site_list, input_dir, output_dir):
    """Print bankability metrics summary table to terminal with Mean values."""
    bankability_cols = ["DSCR [-]", "DSCR Breach Years", "Debt Headroom [MEuro]", "Debt Headroom [% of CAPEX]"]
    
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
        for col in bankability_cols:
            if col in df.columns:
                data = pd.to_numeric(df[col], errors="coerce")
                # Multiply Debt Headroom % by 100 to convert to percent
                if col == "Debt Headroom [% of CAPEX]":
                    data = data * 100
                # Use simplified column name for DSCR
                col_label = "DSCR" if col == "DSCR [-]" else col
                row[f"{col_label} (Mean)"] = data.mean()
                
                # Add minimum DSCR right after mean
                if col == "DSCR [-]":
                    row["DSCR (Min)"] = data.min()
        
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
    bankability_cols = ["DSCR [-]", "DSCR Breach Years", "Debt Headroom [MEuro]", "Debt Headroom [% of CAPEX]"]
    
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
        for col in bankability_cols:
            if col in df.columns:
                data = pd.to_numeric(df[col], errors="coerce")
                # Multiply Debt Headroom % by 100 to convert to percent
                if col == "Debt Headroom [% of CAPEX]":
                    data = data * 100
                # Use simplified column name for DSCR
                col_label = "DSCR" if col == "DSCR [-]" else col
                row[f"{col_label} (P90)"] = np.percentile(data.dropna(), 10)
                
                # Add minimum DSCR right after mean
                if col == "DSCR [-]":
                    row["DSCR (Min)"] = data.min()
        
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
        save_npv_capex_distribution_boxplot(processed_dataframes, args.output_dir)
        save_dscr_distribution_boxplot(processed_dataframes, args.output_dir)
        save_multi_site_dscr_yearly(SITES_TO_PLOT, args.input_dir, args.output_dir)
        save_bankability_metrics_table(SITES_TO_PLOT, args.input_dir, args.output_dir)
        #save_bankability_metrics_table_p90(SITES_TO_PLOT, args.input_dir, args.output_dir)
        save_financial_metrics_table(SITES_TO_PLOT, args.input_dir, args.output_dir)
        #save_financial_metrics_table_p90(SITES_TO_PLOT, args.input_dir, args.output_dir)
        save_multi_site_comparison(SITES_TO_PLOT, args.input_dir, args.output_dir)

    print(f"Execution Complete. Plots saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()