import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# --- File paths ---
EVAL_DIR = r"HPP/Evaluations"
INPUT_TS_DIR = r"HPP/SiteConfig/_prepared_input_ts"
# Specific path for saving plots
PLOT_OUTPUT_DIR = r"C:\Users\malth\HPP\hydesign\HPP\DataStoreage\ScatterhistPlots"

# Create the output directory if it doesn't exist
if not os.path.exists(PLOT_OUTPUT_DIR):
    os.makedirs(PLOT_OUTPUT_DIR)

def read_input_ts(path):
    try:
        df = pd.read_csv(path, sep=',', encoding='utf-8-sig')
        if len(df.columns) <= 1:
            df = pd.read_csv(path, sep=';', encoding='utf-8-sig')
        return df
    except Exception as e:
        print(f"Error reading input file: {e}")
        return None

def scatterhist(df, x_col, y_col, xlabel, ylabel, title, filename):
    """Creates a scatter plot with marginal histograms."""
    sns.set(style="whitegrid")
    
    # Ensure columns are lowercase to match the standardized DF
    x_key = x_col.lower()
    y_key = y_col.lower()
    
    g = sns.JointGrid(data=df, x=x_key, y=y_key, space=0, height=7, ratio=5)
    g.plot_joint(sns.scatterplot, s=15, alpha=0.4, color='teal', edgecolor='none')
    g.plot_marginals(sns.histplot, kde=True, color='teal')
    g.set_axis_labels(xlabel, ylabel)
    
    plt.suptitle(title, y=1.02, fontsize=14)
    
    # Construct full save path
    save_path = os.path.join(PLOT_OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

# --- Multi-site logic ---
prod_files = [f for f in os.listdir(EVAL_DIR) if f.endswith('_hourly_production_1982_2015.csv') or f.endswith('_hourly_production_1982_1982.csv')]

if not prod_files:
    sys.exit("No production files found.")

for prod_file in prod_files:
    site = prod_file.split('_hourly_production')[0]
    prod_path = os.path.join(EVAL_DIR, prod_file)
    input_ts_path = os.path.join(INPUT_TS_DIR, f"{site}_input_ts.csv")

    print(f"\nProcessing site: {site}")
    
    if not os.path.exists(input_ts_path):
        print(f"Skipping {site}: Input file missing.")
        continue

    input_ts = read_input_ts(input_ts_path)
    prod = pd.read_csv(prod_path)

    # Standardize column names
    input_ts.columns = input_ts.columns.str.lower()
    prod.columns = prod.columns.str.lower()

    # Parse Datetime
    input_ts['time'] = pd.to_datetime(input_ts['time'], dayfirst=True, errors='coerce')
    prod['time'] = pd.to_datetime(prod['time'], errors='coerce')

    # Merge
    merged = pd.merge(input_ts, prod, on='time', how='inner')

    # --- FILTER FOR 1982 - 2015 ---
    # This ensures no matter what the file contains, the plot only shows this range
    mask = (merged['time'] >= '1982-01-01') & (merged['time'] <= '2015-12-31 23:59:59')
    merged = merged.loc[mask]

    if merged.empty:
        print(f"No data available for 1982-2015 for {site}.")
        continue

    # --- Generate Plots ---
    # Example: Wind Speed vs Wind Production
    scatterhist(
        merged, "ws_150", "wind_t", 
        "Wind Speed at 150m [m/s]", "Wind Production [MWh]", 
        f"{site}: Wind Correlation (1982-2015)", 
        f"{site}_ws150_windt.png"
    )

    # Example: GHI vs Solar Production
    scatterhist(
        merged, "ghi", "solar_t", 
        "Global Horizontal Irradiance [W/m²]", "Solar Production [MWh]", 
        f"{site}: Solar Correlation (1982-2015)", 
        f"{site}_ghi_solart.png"
    )

    # Example: Wind Production vs Price
    scatterhist(
        merged, "wind_t", "price", 
        "Wind Production [MWh]", "Price [€/MWh]", 
        f"{site}: Wind_Price Correlation (1982-2015)", 
        f"{site}_windt_price.png"
    )
    
    
    # Example: Solar Production vs Price
    scatterhist(
        merged, "solar_t", "price", 
        "Solar Production [MWh]", "Price [€/MWh]", 
        f"{site}: Solar_Price Correlation (1982-2015)", 
        f"{site}_solart_price.png"
    )
