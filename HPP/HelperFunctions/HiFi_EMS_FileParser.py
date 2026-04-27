import pandas as pd
import os

# --- Configuration ---
input_file = r"C:\Users\malth\HPP\hydesign\hydesign\examples\Europe\GWA2\input_ts_Siciliy_South_DA.csv"
output_path_power = r"C:\Users\malth\HPP\hydesign\hydesign\examples\HiFiEMS_inputs\Power"
output_path_market = r"C:\Users\malth\HPP\hydesign\hydesign\examples\HiFiEMS_inputs\Market"

power_cols_to_zero = ["DA_2", "DA_3", "HA", "RT", "HA_ub", "HA_lb"]
market_cols = [
    "SM_forecast_1", "SM_forecast_2", "SM_forecast_3", "SM_cleared",
    "BM_Up_cleared", "BM_Down_cleared", "reg_cleared", "reg_forecast_1",
    "reg_forecast_2", "reg_forecast_3", "reg_vol_Up", "reg_vol_Down"
]
suffix = "_ITda"

os.makedirs(output_path_power, exist_ok=True)
os.makedirs(output_path_market, exist_ok=True)

def split_data():
    print("Reading source file (detecting semicolons)...")
    # Added sep=';' to handle your specific file format
    df = pd.read_csv(input_file, sep=';')
    
    # dayfirst=True ensures 01-01 is read as Jan 1st, not Jan 1st (ambiguous) 
    # or failing on 31-01 (where it thinks 31 is a month).
    df['time'] = pd.to_datetime(df['time'], dayfirst=True)
    
    price_col = 'price' if 'price' in df.columns else 'Price'
    years = df['time'].dt.year.unique()
    
    for year in years:
        if pd.isna(year): continue # Skip any empty rows if they exist
        
        print(f"Processing year: {int(year)}...")
        year_df = df[df['time'].dt.year == year].copy()
        
        # --- 1. Wind Data ---
        wind_export = pd.DataFrame()
        wind_export['time'] = year_df['time']
        wind_export['Measurement'] = year_df['WP_150']
        wind_export['DA_1'] = year_df['WP_150_DA']
        for col in power_cols_to_zero:
            wind_export[col] = 0
        wind_export.to_csv(os.path.join(output_path_power, f"Winddata{int(year)}{suffix}.csv"), index=False)
        
        # --- 2. Solar Data ---
        solar_export = pd.DataFrame()
        solar_export['time'] = year_df['time']
        solar_export['Measurement'] = year_df['SP']
        solar_export['DA_1'] = year_df['SP_DA']
        for col in power_cols_to_zero:
            solar_export[col] = 0
        solar_export.to_csv(os.path.join(output_path_power, f"Solardata{int(year)}{suffix}.csv"), index=False)
        
        # --- 3. Market Data ---
        market_export = pd.DataFrame()
        market_export['time'] = year_df['time']
        for col in market_cols:
            market_export[col] = year_df[price_col]
        market_export.to_csv(os.path.join(output_path_market, f"Market{int(year)}{suffix}.csv"), index=False)

    print(f"\nSuccess! Check your folders for the {suffix}.csv files.")

if __name__ == "__main__":
    split_data()