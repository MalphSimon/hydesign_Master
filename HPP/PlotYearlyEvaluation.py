import argparse
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------
# 1. NAME MAPPING & CONFIGURATIONS
# ---------------------------------------------------------------------------

SITE_DISPLAY_MAP = {
    "SicilySouth": "Sicily South (IT)",
    "Golfe_du_Lion": "Golfe du Lion (FRs)",
    "Sud_Atlantique": "Sud Atlantique (FRw)",
    "Thetys": "Thetys (NL)",
    "NordsoenMidt": "Nordsøen Midt (DK)",
    "Vestavind": "Vestavind (NO)",
    "SicilySouth_HiFiEMS": "Sicily South (IT)",
    "Golfe_du_Lion_HiFiEMS": "Golfe du Lion (FRs)",
    "Sud_Atlantique_HiFiEMS": "Sud Atlantique HPP (FRw)",
    "Sud_Atlantique_Solar_HiFiEMS": "Sud Atlantique Solar (FRw)",
    "Sud_Atlantique_Wind_HiFiEMS": "Sud Atlantique Wind (FRw)",
    "Thetys_HiFiEMS": "Thetys HPP (NL)",
    "Thetys_Solar_HiFiEMS": "Thetys Solar (NL)",
    "Thetys_Wind_HiFiEMS": "Thetys Wind (NL)",
    "NordsoenMidt_HiFiEMS": "Nordsøen Midt (DK)",
    "Vestavind_HiFiEMS": "Vestavind (NO)",

}

UNIFORM_COLOR = "#9C7667"

METRICS = [
    {"key": "npv", "title": "NPV", "ylabel": "M EUR", "candidates": ["NPV [MEuro]", "NPV"]},
    {"key": "npv_capex", "title": "NPV/CAPEX", "ylabel": "%", "candidates": ["NPV_over_CAPEX", "NPV/CAPEX"]},
    {"key": "irr", "title": "IRR", "ylabel": "%", "candidates": ["IRR", "IRR [%]"]},
]

SITES_TO_PLOT = ["Thetys_HiFiEMS", "Thetys_Wind_HiFiEMS"] # For single-site triple stack
INPUT_DIR_DEFAULT = os.path.join("HPP", "Evaluations", "HiFiEMS", "New") 
OUTPUT_DIR_DEFAULT = os.path.join("HPP", "Evaluations", "HiFiEMS", "New", "plots")

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
    """Specific helper to find the NPV column for volatility plots."""
    for cfg in METRICS:
        if cfg['key'] == 'npv':
            return _find_column(df, cfg["candidates"])
    return None

# ---------------------------------------------------------------------------
# 3. SINGLE-SITE TRIPLE STACK (With CV in Legend)
# ---------------------------------------------------------------------------

def save_single_site_triple_stack(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    site_id = os.path.basename(csv_path).split("_HiFiEMS_HiFiEMS_eval_")[0]
    display_name = SITE_DISPLAY_MAP.get(site_id, site_id)
    x, labels = _prepare_year_axis(df)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True, sharex=True)
    
    # Pre-populate site name and an identifier for combining later
    site_stats = {"Site": display_name, "site_id": site_id}

    for i, cfg in enumerate(METRICS):
        col = _find_column(df, cfg["candidates"])
        if col:
            data = pd.to_numeric(df[col], errors="coerce").dropna()
            # Convert decimal to percentage for relevant metrics
            y_plot = data * 100 if cfg["ylabel"] == "%" else data
            
            mean_val = y_plot.mean()
            std_val = y_plot.std()
            
            # Use 'npv' key specifically for CV calculation/reporting context
            if cfg['key'] == 'npv':
                cv_val = (std_val / abs(mean_val)) if abs(mean_val) > 1e-6 else 0
                label_str = f"Mean: {mean_val:.2f} {cfg['ylabel']}\nCV: {cv_val:.2f}"
                site_stats[f"{cfg['title']} CV"] = cv_val
            else:
                label_str = f"Mean: {mean_val:.2f} {cfg['ylabel']}"

            # Store mean stats for the table
            site_stats[f"Mean {cfg['title']}"] = mean_val

            axes[i].plot(x, y_plot, marker="o", linewidth=2.2, color=UNIFORM_COLOR, 
                         label=label_str)
            
            axes[i].set_ylabel(f"{cfg['title']} [{cfg['ylabel']}]", fontweight='bold', fontsize=12)
            axes[i].grid(True, alpha=0.2, linestyle='--')
            axes[i].set_title(f"{cfg['title']}", loc='left', fontsize=11, fontweight='bold', alpha=0.7)
            axes[i].tick_params(axis='both', which='major', labelsize=11)
            axes[i].legend(loc="upper right", frameon=True, fontsize=10)

    axes[2].set_xticks(x)
    year_num = pd.to_numeric(labels, errors="coerce")
    axes[2].set_xticklabels([f"{int(v) % 100:02d}" if pd.notna(v) else "" for v in year_num])
    axes[2].set_xlabel("Scenario Year", fontweight='bold', fontsize=12)

    plt.suptitle(f"Financial Performance & Volatility: {display_name}", fontsize=17, fontweight='bold')
    fig.savefig(os.path.join(output_dir, f"{site_id}_TripleStack_Summary.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    # Store the actual cleaned DataFrame for combining later
    df['site_display'] = display_name # Add column for seaborn
    return site_stats, df

# ---------------------------------------------------------------------------
# 4. VOLATILITY VISUALIZATION PLOTS
# ---------------------------------------------------------------------------

def save_risk_return_scatter(summary_df, output_dir):
    """Plots Mean NPV against NPV CV for a risk-return map."""
    plt.figure(figsize=(10, 8))
    
    # Scatter plot with regression line using seaborn
    sns.regplot(data=summary_df, x="NPV CV", y="Mean NPV", 
                scatter_kws={"s": 100, "alpha": 0.7, "color": UNIFORM_COLOR},
                line_kws={"color": "tab:orange", "alpha": 0.5, "ls": "--"})

    # Label points with Site Name
    for i, row in summary_df.iterrows():
        plt.text(row["NPV CV"] + 0.05, row["Mean NPV"], row["Site"], fontsize=9, va='center')

    plt.axhline(0, color='black', lw=1, ls='-') # Base NPV 0 line
    plt.grid(True, alpha=0.3, ls='--')
    
    plt.xlabel("Volatility (Coefficient of Variation of NPV)", fontweight='bold', fontsize=12)
    plt.ylabel("Mean NPV (M EUR)", fontweight='bold', fontsize=12)
    plt.title("Risk-Return Map: Profitability vs. Predictability", fontsize=15, fontweight='bold')
    
    # Add quadrant labels for emphasis
    x_lim = plt.xlim()
    y_lim = plt.ylim()
    plt.text(x_lim[0] + (x_lim[1]-x_lim[0])*0.1, y_lim[1]*0.9, "Safe\nHavens", fontsize=11, fontweight='bold', color='green', ha='center')
    plt.text(x_lim[1]*0.9, y_lim[1]*0.9, "High-Stakes\nGambles", fontsize=11, fontweight='bold', color='orange', ha='center')
    plt.text(x_lim[1]*0.9, y_lim[0]*0.9, "Avoid", fontsize=11, fontweight='bold', color='red', ha='center')

    plt.savefig(os.path.join(output_dir, "Risk_Return_ScatterMap.png"), dpi=200, bbox_inches='tight')
    plt.close()


def save_npv_distribution_boxplot(all_site_data_frames, output_dir):
    """Creates a boxplot showing the distribution of NPV across all sites."""
    plt.figure(figsize=(12, 7))
    
    # Combine all individual site DataFrames into one massive one for Seaborn
    combined_df = pd.concat(all_site_data_frames, ignore_index=True)
    
    # Find correct NPV column name in the combined DF
    npv_col = _get_npv_column(combined_df)
    
    if npv_col:
        # Boxplot overlaid with individual points (stripplot) for transparency
        sns.boxplot(data=combined_df, x='site_display', y=npv_col, palette='viridis', width=0.6)
        sns.stripplot(data=combined_df, x='site_display', y=npv_col, color='black', size=3, jitter=0.2, alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Site Location", fontweight='bold', fontsize=12)
        plt.ylabel("NPV (M EUR)", fontweight='bold', fontsize=12)
        plt.title("NPV Distribution Across Weather Years", fontsize=16, fontweight='bold')
        plt.grid(axis='y', alpha=0.3, ls='--')
        plt.axhline(0, color='red', lw=1.5, ls='--') # Mark the break-even line
        
        plt.savefig(os.path.join(output_dir, "NPV_Volatility_Distribution_Boxplot_ALL.png"), dpi=200, bbox_inches='tight')
    else:
        print("[Error] Could not identify NPV column for Boxplot.")
    plt.close()

def save_npv_capex_distribution_boxplot(all_site_data_frames, output_dir):
    """
    Creates an academic boxplot for NPV/CAPEX.
    - Y-axis in percentage format (e.g., 20%).
    - 6 distinct colors in a muted academic palette.
    - Black labels, titles, and full border.
    - Rotated X-axis labels for clarity.
    """
    # 1. Combine data
    combined_df = pd.concat(all_site_data_frames, ignore_index=True)
    
    target_col = "NPV_over_CAPEX"
    if target_col not in combined_df.columns:
        print(f"[Error] Column '{target_col}' not found.")
        return

    # 2. Setup Figure
    plt.figure(figsize=(12, 7))
    
    # EXTENDED ACADEMIC PALETTE (6 Distinct Muted Tones)
    # Teal, Blue, Copper, Sage Green, Sand/Gold, Muted Mauve
    site_colors_6 = [
        "#43D1D9", "#4B86C2", "#9C7667", 
        "#68A357", "#D4A373", "#B07BA1"
    ]

    # 3. Create Boxplot
    # fliersize=0 because dots are handled by stripplot
    ax = sns.boxplot(
        data=combined_df, 
        x='site_display', 
        y=target_col, 
        palette=site_colors_6, 
        width=0.5,
        fliersize=0,
        linewidth=1.5
    )
    
    # 4. Overlay stripplot (the dots)
    sns.stripplot(
        data=combined_df, 
        x='site_display', 
        y=target_col, 
        color='black', 
        size=4, 
        jitter=0.15, 
        alpha=0.3
    )
    
    # 5. PERCENTAGE Y-AXIS FORMATTING
    # PercentFormatter(1.0) converts a value like 0.2 to "20%"
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # 6. Styling & Formatting (Black labels and titles)
    plt.xticks(rotation=45, ha='right', fontsize=10, color='black')
    plt.yticks(color='black', fontsize=10)
    
    plt.xlabel("Site Location", fontweight='bold', fontsize=11, color='black')
    plt.ylabel("NPV / CAPEX [%]", fontweight='bold', fontsize=11, color='black')
    
    plt.title("Distribution of NPV/CAPEX Across Scenario Years", 
              fontsize=14, fontweight='bold', color='black', pad=25)
    
    # 7. Border and Grid
    plt.axhline(0, color='red', lw=1.2, ls='--', alpha=0.6) # Break-even line
    plt.grid(axis='y', alpha=0.3, linestyle=':')
    
    # Ensure the black edge is all around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)

    # 8. Save
    save_path = os.path.join(output_dir, "NPV_CAPEX_Boxplot_Percentage.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    
    plt.close()



# ---------------------------------------------------------------------------
# 5. MULTI-SITE COMPARISON (Unchanged, for context)
# ---------------------------------------------------------------------------

def save_multi_site_comparison(site_list, input_dir, output_dir):
    # Compressed height to keep layout tight
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # Academic palette: Teal, Blue, Copper
    site_colors = ["#43D1D9", "#4B86C2", "#9C7667"]
    # NEW TITLE COLOR based on your image
    title_text_color = "#334155" 
    
    x_coords, year_labels = None, None
    site_legend_handles = []

    for idx, site_key in enumerate(site_list):
        files = glob.glob(os.path.join(input_dir, f"{site_key}*eval_*.csv"))
        if not files: continue
        
        df = pd.read_csv(files[0])
        x, labels = _prepare_year_axis(df)
        if x_coords is None: x_coords, year_labels = x, labels
        
        full_site_name = SITE_DISPLAY_MAP.get(site_key, site_key)
        color = site_colors[idx % 3]

        # Site handle for bottom footer legend
        site_legend_handles.append(mlines.Line2D(
            [], [], color=color, marker='o', linestyle='-', 
            markersize=8, label=full_site_name
        ))

        for i, cfg in enumerate(METRICS):
            col = _find_column(df, cfg["candidates"])
            if col:
                data = pd.to_numeric(df[col], errors="coerce").dropna()
                avg = data.mean()
                sigma = data.std()
                cv = (sigma / avg) if avg != 0 else 0
                
                unit = cfg["ylabel"]
                y_plot = data * 100 if unit == "%" else data
                avg_plot = avg * 100 if unit == "%" else avg
                
                # CV only for the top NPV plot
                if i == 0:
                    label_str = f"Mean: {avg_plot:.1f} {unit}\nCV: {cv:.2f}"
                else:
                    label_str = f"Mean: {avg_plot:.2f} {unit}"

                axes[i].plot(x, y_plot, marker="o", markersize=5, 
                             label=label_str, color=color, alpha=0.8)
                
                # UPDATED: Subplot title color and position
                axes[i].set_title(cfg['title'], loc='left', fontweight='bold', 
                                  fontsize=12, pad=-12, color=title_text_color)
                
                axes[i].set_ylabel(f"[{unit}]", fontweight='bold', fontsize=10)
                axes[i].grid(True, alpha=0.3, linestyle=':')

    # X-axis formatting
    if x_coords is not None:
        axes[2].set_xticks(x_coords)
        year_num = pd.to_numeric(year_labels, errors="coerce")
        axes[2].set_xticklabels([f"{int(v) % 100:02d}" if pd.notna(v) else "" for v in year_num])
        axes[2].set_xlabel("Scenario Year", fontweight='bold', fontsize=12)

    for ax in axes:
        ax.legend(loc='upper right', fontsize=9, frameon=True, framealpha=0.9, labelspacing=1.0)

    # Maintain tight layout
    plt.subplots_adjust(bottom=0.12, top=0.94, hspace=0.15) 
    
    # Footer legend for Site Names
    fig.legend(
        handles=site_legend_handles, 
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(site_legend_handles), 
        frameon=False, 
        fontsize=12,
        columnspacing=3.0
    )

    plt.suptitle("Financial Performance Comparison", fontsize=16, fontweight='bold', color="#1e293b")
    
    save_file = os.path.join(output_dir, "Financial_Comparison_Thetys.png")
    fig.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------------------
# 6. MULTI-SITE COMPARISON Normalized
# ---------------------------------------------------------------------------

def save_multi_site_comparison_normalized(site_list, input_dir, output_dir):
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # Academic palette
    site_colors = ["#43D1D9", "#4B86C2", "#9C7667"]
    title_text_color = "#334155" 
    
    x_coords, year_labels = None, None
    site_legend_handles = []

    for idx, site_key in enumerate(site_list):
        files = glob.glob(os.path.join(input_dir, f"{site_key}*eval_*.csv"))
        if not files: continue
        
        df = pd.read_csv(files[0])
        x, labels = _prepare_year_axis(df)
        if x_coords is None: x_coords, year_labels = x, labels
        
        full_site_name = SITE_DISPLAY_MAP.get(site_key, site_key)
        color = site_colors[idx % 3]

        # 1. Create the clean handle for the FOOTER legend (No CV here)
        site_legend_handles.append(mlines.Line2D(
            [], [], color=color, marker='o', linestyle='-', 
            markersize=8, label=full_site_name
        ))

        for i, cfg in enumerate(METRICS):
            col = _find_column(df, cfg["candidates"])
            if col:
                data = pd.to_numeric(df[col], errors="coerce").dropna()
                avg = data.mean()
                
                # Normalize
                y_norm = data / avg if abs(avg) > 1e-6 else data
                
                # 2. Calculate CV for the legend text
                sigma = data.std()
                cv_val = (sigma / abs(avg)) if abs(avg) > 1e-6 else 0
                
                # 3. Define label: Only include "cv =" for the NPV subplot (index 0)
                if i == 0:
                    line_label = f"{full_site_name} cv = {cv_val:.3f}"
                else:
                    line_label = "_nolegend_" # Hide site names in other subplots to avoid redundancy

                axes[i].plot(x, y_norm, marker="o", markersize=5, 
                             color=color, alpha=0.8, label=line_label)
                
                axes[i].set_title(f"{cfg['title']} (Normalized)", loc='left', 
                                 fontweight='bold', fontsize=12, pad=-12, color=title_text_color)
                
                axes[i].set_ylabel("Ratio to Mean", fontweight='bold', fontsize=10)
                axes[i].grid(True, alpha=0.3, linestyle=':')
                axes[i].axhline(1.0, color='black', lw=0.8, ls='--', alpha=0.5)

    # X-axis formatting
    if x_coords is not None:
        axes[2].set_xticks(x_coords)
        year_num = pd.to_numeric(year_labels, errors="coerce")
        axes[2].set_xticklabels([f"{int(v) % 100:02d}" if pd.notna(v) else "" for v in year_num])
        axes[2].set_xlabel("Scenario Year", fontweight='bold', fontsize=12)

    # 4. Show the legend ONLY on the NPV subplot to show CVs
    axes[0].legend(loc='upper right', fontsize=9, frameon=True, framealpha=0.9)

    # Adjust layout to make room for footer
    plt.subplots_adjust(bottom=0.15, top=0.92, hspace=0.2) 
    
    # 5. Restore the SITE legend at the very bottom
    if site_legend_handles:
        fig.legend(
            handles=site_legend_handles, 
            loc='lower center', 
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(site_legend_handles), 
            frameon=False, 
            fontsize=12,
            columnspacing=3.0
        )

    plt.suptitle("Normalized Performance & Volatility Comparison", fontsize=16, fontweight='bold', color="#1e293b")
    
    save_file = os.path.join(output_dir, "Financial_Comparison_Thetys_Normalized.png")
    fig.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=INPUT_DIR_DEFAULT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    all_site_stats = []
    processed_dataframes = [] # Collect raw DFs for Boxplot

    for site in SITES_TO_PLOT:
        files = glob.glob(os.path.join(args.input_dir, f"{site}_HiFiEMS_HiFiEMS_eval_*.csv"))
        for f in files:
            print(f"[Processing] {site}")
            stats, cleaned_df = save_single_site_triple_stack(f, args.output_dir)
            all_site_stats.append(stats)
            processed_dataframes.append(cleaned_df)

    # Generate and save Summary Table
    if all_site_stats:
        summary_df = pd.DataFrame(all_site_stats)
        # Reorder columns for readability (removed redundant NPV/CAPEX CV)
        cols = ["Site", "Mean NPV", "NPV CV", "Mean NPV/CAPEX", "Mean IRR"]
        summary_df = summary_df[[c for c in cols if c in summary_df.columns]]
        
        summary_path = os.path.join(args.output_dir, "Financial_Summary_Statistics.csv")
        summary_df.to_csv(summary_path, index=False)
        
        #print("\n" + "="*30)
        #print("SUMMARY STATISTICS")
        #print("="*30)
        #print(summary_df.to_string(index=False))
        #print("="*30)

        # Generate the new Volatility Plots
        print(f"\n[Processing] Generating Volatility Visualizations...")
        #save_npv_distribution_boxplot(processed_dataframes, args.output_dir)
        #save_npv_capex_distribution_boxplot(processed_dataframes, args.output_dir)

    print(f"\n[Processing] Multi-site comparison overlay...")
    save_multi_site_comparison(SITES_TO_PLOT, args.input_dir, args.output_dir)
    save_multi_site_comparison_normalized(SITES_TO_PLOT, args.input_dir, args.output_dir)
    print(f"Execution Complete. All plots and summary saved to: {args.output_dir}")

if __name__ == "__main__":
    main()