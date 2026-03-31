import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# --- File paths ---
input_ts_path = r"HPP/SiteConfig/_prepared_input_ts/Golfe_du_Lion_input_ts.csv"
prod_path = r"HPP/Evaluations/Golfe_du_Lion_hourly_production_1982_1982.csv"

def read_input_ts(path):
    try:
        df = pd.read_csv(path, sep=',', encoding='utf-8-sig')
        if len(df.columns) <= 1:
            df = pd.read_csv(path, sep=';', encoding='utf-8-sig')
        return df
    except Exception as e:
        print(f"Error reading input file: {e}")
        return None

# Load data
input_ts = read_input_ts(input_ts_path)
prod = pd.read_csv(prod_path)

if input_ts is None or prod is None:
    sys.exit("Could not load dataframes.")

# 1. Standardize column names to lowercase immediately
input_ts.columns = input_ts.columns.str.lower()
prod.columns = prod.columns.str.lower()

# 2. Parse Datetime
# Note: input_ts uses DD-MM-YYYY based on your head() output
input_ts['time'] = pd.to_datetime(input_ts['time'], dayfirst=True, errors='coerce')
prod['time'] = pd.to_datetime(prod['time'], errors='coerce')

# 3. FIX: Align years if they don't match
# Your input_ts is 2012, but prod is 1982. 
# We shift input_ts to match the year in prod so the merge works.
year_prod = prod['time'].dt.year.iloc[0]
year_input = input_ts['time'].dt.year.iloc[0]

if year_prod != year_input:
    print(f"Aligning years: Shifting input_ts from {year_input} to {year_prod}")
    delta = year_prod - year_input
    input_ts['time'] = input_ts['time'] + pd.offsets.DateOffset(years=delta)

# 4. Merge on time
merged = pd.merge(input_ts, prod, on='time', how='inner')

print(f"Merged shape: {merged.shape}")
if merged.empty:
    sys.exit("Merge failed: No overlapping timestamps found even after alignment.")

# 5. Plotting function
def scatterhist(x_col, y_col, xlabel, ylabel, title, filename):
    """
    Creates a scatter plot with marginal histograms.
    Input strings are automatically lowercased to match dataframe.
    """
    sns.set(style="whitegrid")
    
    # JointGrid handles the layout for scatter + histograms
    g = sns.JointGrid(
        data=merged, 
        x=x_col.lower(), 
        y=y_col.lower(), 
        space=0, 
        height=7, 
        ratio=5
    )
    
    # Add scatter plot to center
    g.plot_joint(sns.scatterplot, s=15, alpha=0.4, color='teal', edgecolor='none')
    
    # Add histograms to the top and right
    g.plot_marginals(sns.histplot, kde=True, color='teal')
    
    g.set_axis_labels(xlabel, ylabel)
    plt.suptitle(title, y=1.02, fontsize=14)
    
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# --- Generate Plots ---

# 1. Wind speed vs wind production
scatterhist(
    x_col="ws_150",
    y_col="wind_t",
    xlabel="Wind Speed at 150m [m/s]",
    ylabel="Wind Production",
    title="Wind Resource vs Production Correlation",
    filename="scatterhist_ws150_windt.png"
)

# 2. GHI vs solar production
scatterhist(
    x_col="ghi",
    y_col="solar_t",
    xlabel="Global Horizontal Irradiance [W/m²]",
    ylabel="Solar Production",
    title="Solar Resource vs Production Correlation",
    filename="scatterhist_ghi_solart.png"
)