import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. File Paths ---
base_path = r"C:\Users\malth\HPP\hydesign\hydesign\examples\Europe\GWA2\\"
save_path = r"C:\Users\malth\HPP\hydesign\HPP\DataStoreage\PVCompare"

# Create the save directory if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

# --- 2. Load Data ---
try:
    # Load onshore as strings to handle the dot-separators (thousands)
    onshore_all = pd.read_csv(base_path + "GHI.csv", sep=';', dtype=str)
    
    # Load individual offshore site files
    offshore_sites = {
        'Golfe_Du_Lion': pd.read_csv(base_path + "input_ts_Golfe_du_Lion.csv", sep=';', decimal=','),
        'Thetys': pd.read_csv(base_path + "input_ts_Thetys.csv", sep=';', decimal=','),
        'NordsoenMidt': pd.read_csv(base_path + "input_ts_NordsoenMidt.csv", sep=';', decimal=','),
        'SicilySouth': pd.read_csv(base_path + "input_ts_Siciliy_South.csv", sep=';', decimal=','),
        'Vestavind': pd.read_csv(base_path + "input_ts_Vestavind.csv", sep=';', decimal=','),
        'Sud_Atlantique': pd.read_csv(base_path + "input_ts_Sud_Atlantique.csv", sep=';', decimal=',')
    }
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# --- 3. Preprocessing & Universal Cleaning ---

# A. Clean Onshore Data
onshore_all['time'] = pd.to_datetime(onshore_all['time'], dayfirst=True)
for col in onshore_all.columns:
    if col != 'time':
        # Remove thousands dots, convert to numeric, and scale by 10,000
        onshore_all[col] = pd.to_numeric(onshore_all[col].str.replace('.', '', regex=False), errors='coerce')
        onshore_all[col] = onshore_all[col] / 10000

# B. Clean Offshore Data
for site_name, df in offshore_sites.items():
    df.columns = [c.upper() if c.lower() == 'ghi' else c for c in df.columns]
    df['GHI'] = pd.to_numeric(df['GHI'], errors='coerce')
    df['time'] = pd.to_datetime(df['time'], dayfirst=True)

# --- 4. Statistics Comparison Table ---
# This prints the Mean, Min, Max, and Std Dev to your console
print(f"\n{'Site & Type':<25} | {'Mean':<10} | {'Max':<10} | {'Std Dev':<10}")
print("-" * 75)

for site_name, off_df in offshore_sites.items():
    # Offshore Stats
    off_stats = off_df['GHI'].describe()
    print(f"{site_name + ' (Off)':<25} | {off_stats['mean']:10.2f} | {off_stats['max']:10.2f} | {off_stats['std']:10.2f}")
    
    # Onshore Stats
    if site_name in onshore_all.columns:
        on_stats = onshore_all[site_name].describe()
        print(f"{site_name + ' (On)':<25} | {on_stats['mean']:10.2f} | {on_stats['max']:10.2f} | {on_stats['std']:10.2f}")
    
    print("-" * 75) # Separator between sites

# --- 5. Generate and Save Hourly Plots ---
print(f"\nGenerating plots and saving to: {save_path}...")

for site_name, off_df in offshore_sites.items():
    if site_name in onshore_all.columns:
        plt.figure(figsize=(10, 5))
        
        # Calculate Mean GHI for every hour of the day (0-23)
        off_hourly = off_df.groupby(off_df['time'].dt.hour)['GHI'].mean()
        on_hourly = onshore_all.groupby(onshore_all['time'].dt.hour)[site_name].mean()
        
        # Plotting
        plt.plot(off_hourly.index, off_hourly.values, label='Offshore (Avg)', 
                 color='royalblue', linewidth=2, marker='o', markersize=4)
        plt.plot(on_hourly.index, on_hourly.values, label='Onshore (Avg)', 
                 color='darkorange', linewidth=2, linestyle='--', marker='x', markersize=4)
        
        plt.title(f'Mean Diurnal GHI Profile: {site_name}')
        plt.xlabel('Hour of Day')
        plt.ylabel('GHI (W/m²)')
        plt.xticks(range(0, 24))
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        file_name = f"GHI_Comparison_{site_name}.png"
        plt.savefig(os.path.join(save_path, file_name))
        plt.close() # Close to free up memory

print("Process complete. All stats displayed and plots saved.")