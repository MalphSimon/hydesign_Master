# Analyse the difference in SP and SP_DA in these two files: 
# C:\Users\malth\HPP\hydesign\hydesign\examples\Europe\GWA2\input_ts_Sud_Atlantique_DA_Offshore.csv
# C:\Users\malth\HPP\hydesign\hydesign\examples\Europe\GWA2\input_ts_Sud_Atlantique_DA.csv
# The column name is "SP" and "SP_DA" in both files, compare them against each other and plot the difference.
# The data is hourly from 1982-2015. So plot the hourly avg SP and SP_DA for both files, and also plot the difference between them.

import pandas as pd
df_offshore = pd.read_csv("C:\\Users\\malth\\HPP\\hydesign\\hydesign\\examples\\Europe\\GWA2\\input_ts_Sud_Atlantique_DA_Offshore.csv", sep=';')
df_onshore = pd.read_csv("C:\\Users\\malth\\HPP\\hydesign\\hydesign\\examples\\Europe\\GWA2\\input_ts_Sud_Atlantique_DA.csv", sep=';')

import matplotlib.pyplot as plt
# Convert the 'time' column to datetime format
df_offshore['time'] = pd.to_datetime(df_offshore['time'], format='%d-%m-%Y %H:%M')
df_onshore['time'] = pd.to_datetime(df_onshore['time'], format='%d-%m-%Y %H:%M')
# Set 'time' as the index
df_offshore.set_index('time', inplace=True)
df_onshore.set_index('time', inplace=True)
# Resample to hourly average (if not already hourly)
hourly_offshore = df_offshore.resample('h').mean()
hourly_onshore = df_onshore.resample('h').mean()
# Calculate the difference between SP and SP_DA for both datasets
hourly_offshore['SP_diff'] = hourly_offshore['SP'] - hourly_offshore['SP_DA']
hourly_onshore['SP_diff'] = hourly_onshore['SP'] - hourly_onshore['SP_DA']
# Plot the mean diurnal profile for SP
plt.figure(figsize=(14, 7))
offshore_sp_hourly = df_offshore.groupby(df_offshore.index.hour)['SP'].mean()
onshore_sp_hourly = df_onshore.groupby(df_onshore.index.hour)['SP'].mean()
plt.plot(offshore_sp_hourly.index, offshore_sp_hourly.values, label='Offshore SP', marker='o', linewidth=2)
plt.plot(onshore_sp_hourly.index, onshore_sp_hourly.values, label='Onshore SP', marker='o', linewidth=2, linestyle='--')
plt.title('Mean Diurnal SP Profile: Sud_Atlantique')
plt.xlabel('Hour of Day')
plt.ylabel('SP (W/m²)')
plt.legend()
plt.grid()
plt.xticks(range(0, 24))
plt.show()

# Plot the mean diurnal profile for SP_DA
plt.figure(figsize=(14, 7))
offshore_sp_da_hourly = df_offshore.groupby(df_offshore.index.hour)['SP_DA'].mean()
onshore_sp_da_hourly = df_onshore.groupby(df_onshore.index.hour)['SP_DA'].mean()
plt.plot(offshore_sp_da_hourly.index, offshore_sp_da_hourly.values, label='Offshore SP_DA', marker='o', linewidth=2)
plt.plot(onshore_sp_da_hourly.index, onshore_sp_da_hourly.values, label='Onshore SP_DA', marker='o', linewidth=2, linestyle='--')
plt.title('Mean Diurnal SP_DA Profile: Sud_Atlantique')
plt.xlabel('Hour of Day')
plt.ylabel('SP_DA (W/m²)')
plt.legend()
plt.grid()
plt.xticks(range(0, 24))
plt.show()

# make a table of the mean dinural profile for SP ofshore against sp onshore and for SP_DA offshore against SP_DA onshore. 
diurnal_profile = pd.DataFrame({
    'Hour': range(24),
    'Offshore_SP': offshore_sp_hourly.values,
    'Onshore_SP': onshore_sp_hourly.values,
    'SP_difference_percentage': ((offshore_sp_hourly.values - onshore_sp_hourly.values) / onshore_sp_hourly.values * 100),
    'Offshore_SP_DA': offshore_sp_da_hourly.values,
    'Onshore_SP_DA': onshore_sp_da_hourly.values,
    'SP_DA_difference_percentage': ((offshore_sp_da_hourly.values - onshore_sp_da_hourly.values) / onshore_sp_da_hourly.values * 100)
})
print(diurnal_profile)