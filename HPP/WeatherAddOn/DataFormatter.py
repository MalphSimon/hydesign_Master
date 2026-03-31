
import os
from pathlib import Path

import pandas as pd

input_ts_file = r"C:\Users\malth\Master_Thesis_HPP\hydesign\hydesign\examples\Europe\GWA2\input_ts_Sud_Atlantique.csv"
output_ts_file = r"C:\Users\malth\Master_Thesis_HPP\hydesign\hydesign\examples\Europe\GWA2\input_ts_Sud_Atlantique_Small.csv"


def main():
	source = Path(input_ts_file)
	target = Path(output_ts_file)

	if not source.is_file():
		raise FileNotFoundError(f"Input file not found: {source}")

	# input_ts convention in this project: semicolon-separated, timestamp index in column 0.
	df = pd.read_csv(source, sep=";", index_col=0)

	timestamps = pd.to_datetime(df.index, errors="coerce", dayfirst=True)
	year_mask = timestamps.year == 2012
	df_2012 = df.loc[year_mask].copy()

	if df_2012.empty:
		raise ValueError("No rows found for year 2012 in source file.")

	target.parent.mkdir(parents=True, exist_ok=True)
	df_2012.to_csv(target, sep=";")

	print(f"Source rows: {len(df)}")
	print(f"2012 rows : {len(df_2012)}")
	print(f"Overwritten file: {target}")


if __name__ == "__main__":
	main()



