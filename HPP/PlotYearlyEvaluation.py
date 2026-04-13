"""
Combined Visualization script for HPP Evaluation.
Enhanced font sizes for legends and axis labels.
"""

import argparse
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
}

UNIFORM_COLOR = "#5c8cbc" 

METRICS = [
    {"key": "npv", "title": "NPV", "ylabel": "M EUR", "color": "tab:orange", "style": "line", "candidates": ["NPV [MEuro]", "NPV"]},
    {"key": "npv_capex", "title": "NPV/CAPEX", "ylabel": "%", "color": "tab:blue", "style": "line", "candidates": ["NPV_over_CAPEX", "NPV/CAPEX"]},
    {"key": "revenue", "title": "Revenue", "ylabel": "M EUR", "color": "tab:red", "style": "bar", "candidates": ["Revenues [MEuro]", "Revenue"]},
    {"key": "irr", "title": "IRR", "ylabel": "%", "color": "tab:green", "style": "line", "candidates": ["IRR", "IRR [%]"]},
]

# --- USER SETTINGS ---
SITES_TO_PLOT = ["SicilySouth", "Golfe_du_Lion", "Sud_Atlantique", "Thetys", "NordsoenMidt", "Vestavind"] 
INPUT_DIR_DEFAULT = os.path.join("HPP", "Evaluations", "HiFiEMS") 
OUTPUT_DIR_DEFAULT = os.path.join("HPP", "Evaluations", "HiFiEMS", "plots")

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

# ---------------------------------------------------------------------------
# 3. SINGLE-SITE TRIPLE STACK (With Larger Text)
# ---------------------------------------------------------------------------

def save_single_site_triple_stack(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    site_id = os.path.basename(csv_path).split("_HiFiEMS_HiFiEMS_eval_")[0]
    display_name = SITE_DISPLAY_MAP.get(site_id, site_id)
    x, labels = _prepare_year_axis(df)
    
    triple_cfg = [m for m in METRICS if m['key'] in ['npv', 'npv_capex', 'irr']]
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True, sharex=True)
    
    for i, cfg in enumerate(triple_cfg):
        col = _find_column(df, cfg["candidates"])
        if col:
            data = pd.to_numeric(df[col], errors="coerce")
            y_plot = data * 100 if cfg["ylabel"] == "%" else data
            mean_val = y_plot.mean()
            
            axes[i].plot(x, y_plot, marker="o", linewidth=2.2, color=UNIFORM_COLOR, 
                         label=f"Mean: {mean_val:.2f} {cfg['ylabel']}")
            
            axes[i].set_ylabel(f"{cfg['title']} [{cfg['ylabel']}]", fontweight='bold', fontsize=12)
            axes[i].grid(True, alpha=0.2, linestyle='--')
            axes[i].set_title(f"Annual {cfg['title']}", loc='left', fontsize=11, fontweight='bold', alpha=0.7)
            
            # Increase axis number size
            axes[i].tick_params(axis='both', which='major', labelsize=11)
            
            # Increase legend text size
            axes[i].legend(loc="upper right", frameon=True, fontsize=11)

    axes[2].set_xticks(x)
    year_num = pd.to_numeric(labels, errors="coerce")
    axes[2].set_xticklabels([f"{int(v) % 100:02d}" if pd.notna(v) else "" for v in year_num])
    axes[2].set_xlabel("Scenario Year", fontweight='bold', fontsize=12)

    plt.suptitle(f"Financial Performance: {display_name}", fontsize=17, fontweight='bold')
    fig.savefig(os.path.join(output_dir, f"{site_id}_TripleStack_Summary.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------------------
# 4. MULTI-SITE COMPARISON (With Larger Text)
# ---------------------------------------------------------------------------

def save_multi_site_comparison(site_list, input_dir, output_dir):
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), constrained_layout=True, sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(site_list)))
    
    triple_cfg = [m for m in METRICS if m['key'] in ['npv', 'npv_capex', 'irr']]
    x_coords, year_labels = None, None

    for idx, site_key in enumerate(site_list):
        files = glob.glob(os.path.join(input_dir, f"{site_key}_HiFiEMS_HiFiEMS_eval_*.csv"))
        if not files: continue
        
        df = pd.read_csv(files[0])
        x, labels = _prepare_year_axis(df)
        if x_coords is None: x_coords, year_labels = x, labels
        pretty_name = SITE_DISPLAY_MAP.get(site_key, site_key)

        for i, cfg in enumerate(triple_cfg):
            col = _find_column(df, cfg["candidates"])
            if col:
                data = pd.to_numeric(df[col], errors="coerce")
                y_plot = data * 100 if cfg["ylabel"] == "%" else data
                axes[i].plot(x, y_plot, marker="o", label=pretty_name, color=colors[idx], alpha=0.8)
                axes[i].set_ylabel(f"{cfg['title']} [{cfg['ylabel']}]", fontweight='bold', fontsize=12)
                axes[i].grid(True, alpha=0.2, linestyle='--')
                axes[i].tick_params(axis='both', which='major', labelsize=11)

    if x_coords is not None:
        axes[2].set_xticks(x_coords)
        year_num = pd.to_numeric(year_labels, errors="coerce")
        axes[2].set_xticklabels([f"{int(v) % 100:02d}" if pd.notna(v) else "" for v in year_num])
        axes[2].set_xlabel("Scenario Year", fontweight='bold', fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=min(len(labels), 6), frameon=True, framealpha=0.95, fontsize=11)

    plt.suptitle("Financial Comparison - All Sites", fontsize=18, fontweight='bold')
    fig.savefig(os.path.join(output_dir, "Comparison_MultiSite_TripleStack_ALL.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=INPUT_DIR_DEFAULT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    for site in SITES_TO_PLOT:
        files = glob.glob(os.path.join(args.input_dir, f"{site}_HiFiEMS_HiFiEMS_eval_*.csv"))
        for f in files:
            print(f"[Processing] {site}")
            save_single_site_triple_stack(f, args.output_dir)

    print(f"[Processing] Multi-site comparison overlay...")
    save_multi_site_comparison(SITES_TO_PLOT, args.input_dir, args.output_dir)
    print("Execution Complete.")



if __name__ == "__main__":
    main()