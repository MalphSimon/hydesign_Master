
import os
from pathlib import Path

import pandas as pd

input_ts_file = r"C:\Users\malth\HPP\hydesign\hydesign\examples\Europe\GWA2\input_ts_Sud_Atlantique_onshore.csv"
output_ts_file = r"C:\Users\malth\HPP\hydesign\hydesign\examples\Europe\GWA2\input_ts_Sud_Atlantique_onshore.csv"


def main():
	source = Path(input_ts_file)
	target = Path(output_ts_file)

	if not source.is_file():
		raise FileNotFoundError(f"Input file not found: {source}")

	# input_ts convention in this project: semicolon-separated, timestamp index in column 0.
	df = pd.read_csv(source, sep=";", index_col=0)

	# Print before replacement
	if 'temp_air_1' in df.columns:
		print("Before replacement:")
		print(df['temp_air_1'].head())
		df['temp_air_1'] = df['temp_air_1'].astype(str).str.replace(',', '.', regex=False)
		print("After replacement:")
		print(df['temp_air_1'].head())

	 # Filter for year 2012 if present
	#try:
	   # timestamps = pd.to_datetime(df.index, errors="coerce", dayfirst=True)
	   # year_mask = timestamps.year == 2012
	   # df = df.loc[year_mask]
	   # print(f"Filtered to year 2012: {len(df)} rows")
	#except Exception as e:
	    #print(f"Could not filter by year: {e}")

	target.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(target, sep=";")

	print(f"Rows written: {len(df)}")
	print(f"Overwritten file: {target}")


if __name__ == "__main__":
	main()



