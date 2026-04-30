import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# --- File path ---
HIFIEMS_FILE = r"C:\Users\malth\HPP\hydesign\HPP\Evaluations\HiFiEMS\New\Thetys_HiFiEMS_eval_1982_2015_p42.0_hourly.csv"
# Specific path for saving plots
PLOT_OUTPUT_DIR = r"C:\Users\malth\HPP\hydesign\HPP\DataStoreage\ScatterhistPlots"

# Create the output directory if it doesn't exist
if not os.path.exists(PLOT_OUTPUT_DIR):
    os.makedirs(PLOT_OUTPUT_DIR)

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

# --- Read HiFiEMS data ---
print(f"Processing HiFiEMS file: {HIFIEMS_FILE}")

if not os.path.exists(HIFIEMS_FILE):
    sys.exit(f"File not found: {HIFIEMS_FILE}")

# Read the CSV file
df = pd.read_csv(HIFIEMS_FILE)

# Standardize column names
df.columns = df.columns.str.lower()

# Parse Datetime
df['time'] = pd.to_datetime(df['time'], errors='coerce')

# Add a price column (using SM_price_cleared)
df['price'] = df['sm_price_cleared']

# --- Generate Plots ---
# Wind Production vs Price
scatterhist(
    df, "wind_mw", "price", 
    "Wind Production [MW]", "Price [€/MWh]", 
    "Thetys: Wind_Price Correlation", 
    "Thetys_wind_price.png"
)

# Solar Production vs Price
scatterhist(
    df, "solar_mw", "price", 
    "Solar Production [MW]", "Price [€/MWh]", 
    "Thetys: Solar_Price Correlation", 
    "Thetys_solar_price.png"
)

# --- SOLAR CANNIBALISM ANALYSIS ---
print("\n" + "="*60)
print("SOLAR CANNIBALISM ANALYSIS")
print("="*60)

# Remove NaN values for analysis
df_clean = df.dropna(subset=['solar_mw', 'price'])

# --- 1. Price Degradation Curve (Binning) ---
print("\n1. PRICE DEGRADATION BY SOLAR OUTPUT BINS")
print("-" * 60)

# Create bins for solar output
n_bins = 6
df_clean['solar_bin'] = pd.cut(df_clean['solar_mw'], bins=n_bins)
binning_analysis = df_clean.groupby('solar_bin').agg({
    'price': ['mean', 'std', 'count'],
    'solar_mw': ['min', 'max']
}).round(2)

print("\nPrice by Solar Output Bins:")
print(binning_analysis)

# Plot binning analysis
fig, ax = plt.subplots(figsize=(10, 6))
bin_centers = df_clean.groupby('solar_bin')['solar_mw'].mean()
bin_prices = df_clean.groupby('solar_bin')['price'].mean()
ax.plot(bin_centers, bin_prices, marker='o', linewidth=2, markersize=8, color='darkorange')
ax.fill_between(bin_centers, bin_prices, alpha=0.3, color='darkorange')
ax.set_xlabel('Solar Production [MW]', fontsize=12)
ax.set_ylabel('Average Price [€/MWh]', fontsize=12)
ax.set_title('Thetys: Solar Cannibalism Effect - Price Degradation (2005-2010)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
save_path = os.path.join(PLOT_OUTPUT_DIR, "Thetys_solar_cannibalism_binning.png")
plt.savefig(save_path, dpi=200, bbox_inches='tight')
print(f"\nSaved: {save_path}")
plt.close()

# --- 2. Time-of-Day Pattern ---
print("\n2. TIME-OF-DAY PATTERN ANALYSIS")
print("-" * 60)

df_clean['hour'] = df_clean['time'].dt.hour
hourly_analysis = df_clean.groupby('hour').agg({
    'price': ['mean', 'std', 'count'],
    'solar_mw': 'mean'
}).round(2)

print("\nPrice and Solar Generation by Hour:")
print(hourly_analysis)

# Plot time-of-day analysis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Price by hour
hourly_price = df_clean.groupby('hour')['price'].mean()
ax1.bar(hourly_price.index, hourly_price.values, color='steelblue', alpha=0.7)
ax1.set_ylabel('Average Price [€/MWh]', fontsize=11)
ax1.set_title('Thetys: Average Price by Hour of Day', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_xlim(-0.5, 23.5)

# Solar production by hour
hourly_solar = df_clean.groupby('hour')['solar_mw'].mean()
ax2.bar(hourly_solar.index, hourly_solar.values, color='gold', alpha=0.7)
ax2.set_xlabel('Hour of Day', fontsize=11)
ax2.set_ylabel('Average Solar Production [MW]', fontsize=11)
ax2.set_title('Thetys: Average Solar Generation by Hour of Day', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xlim(-0.5, 23.5)

plt.tight_layout()
save_path = os.path.join(PLOT_OUTPUT_DIR, "Thetys_solar_cannibalism_hourly.png")
plt.savefig(save_path, dpi=200, bbox_inches='tight')
print(f"\nSaved: {save_path}")
plt.close()

# --- 2B. Wind Time-of-Day Pattern ---
print("\n2B. WIND TIME-OF-DAY PATTERN ANALYSIS")
print("-" * 60)

wind_hourly_analysis = df_clean.groupby('hour').agg({
    'price': ['mean', 'std', 'count'],
    'wind_mw': 'mean'
}).round(2)

print("\nPrice and Wind Generation by Hour:")
print(wind_hourly_analysis)

# Plot wind time-of-day analysis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Price by hour (same as above)
hourly_price = df_clean.groupby('hour')['price'].mean()
ax1.bar(hourly_price.index, hourly_price.values, color='steelblue', alpha=0.7)
ax1.set_ylabel('Average Price [€/MWh]', fontsize=11)
ax1.set_title('Thetys: Average Price by Hour of Day', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_xlim(-0.5, 23.5)

# Wind production by hour
hourly_wind = df_clean.groupby('hour')['wind_mw'].mean()
ax2.bar(hourly_wind.index, hourly_wind.values, color='lightcoral', alpha=0.7)
ax2.set_xlabel('Hour of Day', fontsize=11)
ax2.set_ylabel('Average Wind Production [MW]', fontsize=11)
ax2.set_title('Thetys: Average Wind Generation by Hour of Day', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xlim(-0.5, 23.5)

plt.tight_layout()
save_path = os.path.join(PLOT_OUTPUT_DIR, "Thetys_wind_cannibalism_hourly.png")
plt.savefig(save_path, dpi=200, bbox_inches='tight')
print(f"\nSaved: {save_path}")
plt.close()

# --- 3. Elasticity Statistics ---
print("\n3. ELASTICITY & STATISTICAL ANALYSIS")
print("-" * 60)

# Split data into high and low solar quintiles
high_solar = df_clean[df_clean['solar_mw'] >= df_clean['solar_mw'].quantile(0.75)]
low_solar = df_clean[df_clean['solar_mw'] <= df_clean['solar_mw'].quantile(0.25)]

avg_price_high = high_solar['price'].mean()
avg_price_low = low_solar['price'].mean()
avg_solar_high = high_solar['solar_mw'].mean()
avg_solar_low = low_solar['solar_mw'].mean()

# Calculate elasticity: % price change per % solar change
pct_price_change = ((avg_price_high - avg_price_low) / avg_price_low) * 100
pct_solar_change = ((avg_solar_high - avg_solar_low) / avg_solar_low) * 100
elasticity = pct_price_change / pct_solar_change if pct_solar_change != 0 else 0

# Price penalty factor
price_penalty = ((avg_price_low - avg_price_high) / avg_price_low) * 100

# Correlation
correlation = df_clean['solar_mw'].corr(df_clean['price'])

print(f"\nAverage Price (Low Solar Quartile): €{avg_price_low:.2f}/MWh")
print(f"Average Price (High Solar Quartile): €{avg_price_high:.2f}/MWh")
print(f"Price Penalty Factor: {price_penalty:.2f}% (price reduction when solar is high)")
print(f"\nAverage Solar (Low Quartile): {avg_solar_low:.2f} MW")
print(f"Average Solar (High Quartile): {avg_solar_high:.2f} MW")
print(f"\nPrice Elasticity: {elasticity:.3f} (% price change per % solar change)")
print(f"Correlation (Solar vs Price): {correlation:.3f}")

# --- 3B. Wind Elasticity Statistics ---
print("\n3B. WIND CANNIBALISM ANALYSIS - ELASTICITY & STATISTICS")
print("-" * 60)

# Split data into high and low wind quintiles
high_wind = df_clean[df_clean['wind_mw'] >= df_clean['wind_mw'].quantile(0.75)]
low_wind = df_clean[df_clean['wind_mw'] <= df_clean['wind_mw'].quantile(0.25)]

avg_price_high_wind = high_wind['price'].mean()
avg_price_low_wind = low_wind['price'].mean()
avg_wind_high = high_wind['wind_mw'].mean()
avg_wind_low = low_wind['wind_mw'].mean()

# Calculate elasticity: % price change per % wind change
pct_price_change_wind = ((avg_price_high_wind - avg_price_low_wind) / avg_price_low_wind) * 100
pct_wind_change = ((avg_wind_high - avg_wind_low) / avg_wind_low) * 100
elasticity_wind = pct_price_change_wind / pct_wind_change if pct_wind_change != 0 else 0

# Price penalty factor
price_penalty_wind = ((avg_price_low_wind - avg_price_high_wind) / avg_price_low_wind) * 100

# Correlation
correlation_wind = df_clean['wind_mw'].corr(df_clean['price'])

print(f"\nAverage Price (Low Wind Quartile): €{avg_price_low_wind:.2f}/MWh")
print(f"Average Price (High Wind Quartile): €{avg_price_high_wind:.2f}/MWh")
print(f"Price Penalty Factor: {price_penalty_wind:.2f}% (price reduction when wind is high)")
print(f"\nAverage Wind (Low Quartile): {avg_wind_low:.2f} MW")
print(f"Average Wind (High Quartile): {avg_wind_high:.2f} MW")
print(f"\nPrice Elasticity: {elasticity_wind:.3f} (% price change per % wind change)")
print(f"Correlation (Wind vs Price): {correlation_wind:.3f}")

print("\n" + "="*60)
print("SOLAR vs WIND CANNIBALISM COMPARISON")
print("="*60)
print(f"Solar Price Penalty: {price_penalty:.2f}%")
print(f"Wind Price Penalty: {price_penalty_wind:.2f}%")
print(f"Solar Correlation: {correlation:.3f}")
print(f"Wind Correlation: {correlation_wind:.3f}")
print("\n" + "="*60)

print("\n" + "="*60)

# --- 4. FINANCIAL IMPACT ANALYSIS ---
print("\n4. FINANCIAL IMPACT - SOLAR CANNIBALISM COST")
print("-" * 60)

# Cost parameters (EUR/MW and EUR/MWh)
solar_cost_per_mw = 140_000 + 350_000 + 21_000  # Total CapEx per MW
solar_fixed_om = 8.635  # EUR/MW (annually, divided by 8760 hours)
solar_variable_om = 0.0  # Assume included in fixed O&M

# Calculate revenues under two scenarios
total_solar_output = df_clean['solar_mw'].sum()
hours_in_period = len(df_clean)
years_in_period = hours_in_period / (365.25 * 24)

# Scenario 1: ACTUAL (with cannibalism)
actual_revenue = (df_clean['solar_mw'] * df_clean['price']).sum()

# Scenario 2: NO CANNIBALISM (constant baseline price = average price when no solar)
baseline_price = df_clean[df_clean['solar_mw'] <= df_clean['solar_mw'].quantile(0.1)]['price'].mean()
no_cannibalism_revenue = (df_clean['solar_mw'] * baseline_price).sum()

# Calculate loss
revenue_loss_absolute = no_cannibalism_revenue - actual_revenue
revenue_loss_percentage = (revenue_loss_absolute / no_cannibalism_revenue) * 100

# Annual figures (extrapolate from period)
annual_factor = (365.25 * 24) / hours_in_period if hours_in_period > 0 else 1
annual_actual_revenue = actual_revenue * annual_factor
annual_no_cannibalism = no_cannibalism_revenue * annual_factor
annual_revenue_loss = revenue_loss_absolute * annual_factor

# Per MW metrics (assuming 500 MW total capacity as per the table)
capacity_mw = 500
revenue_per_mw = annual_actual_revenue / capacity_mw
revenue_per_mw_no_cannibalism = annual_no_cannibalism / capacity_mw

# Calculate impact on project economics
total_capex = solar_cost_per_mw * capacity_mw
annual_om = (solar_fixed_om / 1000) * capacity_mw  # Convert EUR/MW to EUR/1000MW for calculation

# Levelized Cost of Electricity impact
lcoe_factor = 1.0  # Baseline
lcoe_with_cannibalism = lcoe_factor * (annual_om + (total_capex / 25)) / (total_solar_output * annual_factor / 1e6)

print(f"\nCapacity Analyzed: {capacity_mw} MW Solar")
print(f"\nRevenue Analysis (Based on {hours_in_period:,} hours of data):")
print(f"  Actual Revenue (WITH cannibalism):     €{actual_revenue:,.0f}")
print(f"  Potential Revenue (NO cannibalism):    €{no_cannibalism_revenue:,.0f}")
print(f"  Revenue LOSS due to cannibalism:       €{revenue_loss_absolute:,.0f} ({revenue_loss_percentage:.1f}%)")

print(f"\nAnnualized Figures (extrapolated):")
print(f"  Annual Revenue Loss per {capacity_mw} MW:         €{annual_revenue_loss:,.0f}")
print(f"  Revenue per MW (actual):               €{revenue_per_mw:,.0f}/MW/year")
print(f"  Revenue per MW (no cannibalism):       €{revenue_per_mw_no_cannibalism:,.0f}/MW/year")

print(f"\nProject Economics Impact ({capacity_mw} MW):")
print(f"  Total CapEx (Solar):                   €{total_capex:,.0f}")
print(f"  Payback period reduction:              {(total_capex / annual_revenue_loss):.1f} years longer due to cannibalism")

print(f"\nAverage Price Metrics:")
print(f"  Baseline Price (No solar hours):       €{baseline_price:.2f}/MWh")
print(f"  Actual Average Price:                  €{df_clean['price'].mean():.2f}/MWh")
print(f"  Price reduction due to cannibalism:    €{baseline_price - df_clean['price'].mean():.2f}/MWh")

print("\n" + "="*60)
print("ANALYSIS COMPLETE - Plots saved to DataStoreage/ScatterhistPlots/")
print("="*60)
