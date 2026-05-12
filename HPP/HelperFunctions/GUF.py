import pandas as pd
import os

# Define the file paths
file_paths = [
    r"C:\Users\malth\HPP\hydesign\HPP\Evaluations\HiFiEMS\P25\Sud_Atlantique_HiFiEMS_eval_1982_2015_p25.0.csv",
    r"C:\Users\malth\HPP\hydesign\HPP\Evaluations\HiFiEMS\P25\Sud_Atlantique_Solar_HiFiEMS_eval_1982_2015_p25.0.csv",
    r"C:\Users\malth\HPP\hydesign\HPP\Evaluations\HiFiEMS\P25\Sud_Atlantique_Wind_HiFiEMS_eval_1982_2015_p25.0.csv"
]

def calculate_mean_guf(paths: list) -> pd.DataFrame:
    results = []
    
    for path in paths:
        if os.path.exists(path):
            # Load the dataframe
            df = pd.read_csv(path)
            
            # Extract site name from filename for clarity
            site_name = os.path.basename(path).replace('_eval_1982_2015_p25.0.csv', '')
            
            # Calculate mean GUF (assuming the column name is 'GUF')
            # If your column name differs (e.g., 'GUF' or 'gen_unit_factor'), adjust here:
            if 'GUF' in df.columns:
                mean_guf = df['GUF'].mean()
                results.append({'Site/Configuration': site_name, 'Mean GUF': mean_guf})
            else:
                print(f"Warning: 'GUF' column not found in {site_name}")
        else:
            print(f"File not found: {path}")
            
    return pd.DataFrame(results)

# Execute and display comparison
comparison_df = calculate_mean_guf(file_paths)

print("--- GUF Comparison Results ---")
print(comparison_df.to_string(index=False))

# Optional: Identify the highest performing configuration
if not comparison_df.empty:
    best_site = comparison_df.loc[comparison_df['Mean GUF'].idxmax()]
    print(f"\nHighest Mean GUF: {best_site['Site/Configuration']} ({best_site['Mean GUF']:.4f})")