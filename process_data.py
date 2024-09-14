
#%%
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# Function to get file last modified time
def get_last_modified_time(file_path):
    return datetime.fromtimestamp(Path(file_path).stat().st_mtime)

# Define the folder path where the file is located
folder_path = os.path.join(os.path.dirname(__file__), "Waveforms", "PCB#1", "avg")

# Define the file path for temp_log
templog_path = os.path.join(folder_path, "temp_log.txt")

# Read the temp log
temp_log = pd.read_csv(templog_path, delimiter='\t')
print(temp_log.head())

# Set the file export path
export_path = folder_path.replace("Waveforms", "Data")

# Create the subfolder if it doesn't exist
if not os.path.exists(export_path):
    os.makedirs(export_path)

#%%######### SET READ METHOD ###########
######### CHOOSE AVG OR ALL HERE ##########
#from Functions.func_read_signals_avg import read_signals
from Functions.func_read_signals import read_signals_single, read_signals_avg
from Functions.func_calc_features import calculate_features
###########################################

# set which receiver channels to read
# PCB#1 has 1x receiver, PCB#2/3 have 2x receivers.
# 2 = recA, 3 = recB

# Define the folder path and channel numbers
ch_nums = [2]  # Define the channel numbers

# Create an empty dict to store dfs
all_features_dfs = {}

# Loop over each ch_num
for ch_num in ch_nums:
    print(f'Loading signals for ch_num {ch_num}...')

    if "single" in folder_path:
        signals = read_signals_single(folder_path, ch_num)
    elif "avg" in folder_path:
        signals = read_signals_avg(folder_path, ch_num)
    else:
        raise ValueError("The folder path must contain either 'single' or 'avg'.")

    signals_df, filename_df, time_df = signals
    
    # Plot a signal
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(time_df[0], signals_df[0], linewidth=1)
    plt.title(f'First signal from channel {ch_num}')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (us)')
    plt.grid(True, linestyle='--')
    plt.show()
    
    print('Calculating features...')
    bin_num = 10
    amp_peak_num = 5
    fft_peak_num = 5
    
    all_features_df = calculate_features(signals_df, time_df, bin_num, amp_peak_num, fft_peak_num)
    time_stamp = filename_df['CreationTime']
    filename = filename_df['Filename']
    time_stamp.rename('time_stamp', inplace=True)
    all_features_df = pd.concat([filename, time_stamp, all_features_df], axis=1)
    all_features_df['time_stamp'] = pd.to_datetime(all_features_df['time_stamp'])
    
    print('Extracting hotspot number and temperature...')
    num_of_hotspots = 5
    hotspot_nums = []
    #temps = []
    for index, row in temp_log.iterrows():
        if (row.iloc[1:num_of_hotspots+1].max() - row.iloc[1:num_of_hotspots+1].min()) <= 5:
            hotspot_num = 0
            #temp = row.iloc[1:6].mean()
        else:
            max_column_index = np.argmax(row.iloc[1:num_of_hotspots+1].values) + 1
            hotspot_num = max_column_index
            #temp = row.iloc[1:6].max()
        hotspot_nums.append(hotspot_num)
        #temps.append(temp)
    temp_log['hotspot_num'] = hotspot_nums
    #temp_log['temp'] = temps
    
    print('Combining with features...')
    all_features_df['time_stamp'] = pd.to_datetime(all_features_df['time_stamp'])
    temp_log['time_stamp'] = pd.to_datetime(temp_log['time_stamp'])
    closest_indices = []
    for timestamp in all_features_df['time_stamp']:
        closest_index = temp_log['time_stamp'].sub(timestamp).abs().idxmin()
        closest_indices.append(closest_index)    
    # Identify the index of 'time_stamp' in temp_log
    time_stamp_idx = temp_log.columns.get_loc('time_stamp')
    # Get all columns past 'time_stamp'
    columns_to_copy = temp_log.columns[time_stamp_idx + 1:]
    # Create dictionaries to hold the values for each column
    columns_data = {col: [] for col in columns_to_copy}
    # Populate the dictionaries with data from temp_log based on closest_indices
    for idx in closest_indices:
        for col in columns_to_copy:
            columns_data[col].append(temp_log.loc[idx, col])
    # Find the index to insert the new columns after 'time_stamp' in all_features_df
    index_to_insert_after = all_features_df.columns.get_loc('time_stamp')
    # Insert each column into all_features_df
    for i, col in enumerate(columns_to_copy, start=1):
        all_features_df.insert(index_to_insert_after + i, col, columns_data[col])
    # Sort and reset index
    all_features_df = all_features_df.sort_values(by='time_stamp', ascending=True)
    all_features_df = all_features_df.reset_index(drop=True)
    # Store the DataFrame with suffix based on ch_num in the dictionary
    all_features_dfs[f'all_features_ch{ch_num}'] = all_features_df
    print(f'Combined features for ch_num {ch_num} stored in DataFrame\n')

# Create an empty list to store DataFrames
dfs_with_ch_num = []

# Loop over each DataFrame in all_features_dfs
for ch_num, df in all_features_dfs.items():
    # Extract just the number from the DataFrame name
    ch_num = ch_num.split('_')[2]  # Splitting by '_' and extracting the last element
    # Remove the "ch" prefix
    ch_num = ch_num[2:] if ch_num.startswith('ch') else ch_num
    # Convert to numeric
    ch_num = int(ch_num)
    # Add a new column containing the ch_num
    df.insert(df.columns.get_loc('time_stamp') + 1, 'ch_num', ch_num)
    # Append the DataFrame to the list
    dfs_with_ch_num.append(df)

# Concatenate all DataFrames in dfs_with_ch_num into a single DataFrame
all_features_df = pd.concat(dfs_with_ch_num, ignore_index=True)

# drop 'temp 6' column if present
if 'temp 6' in all_features_df.columns:
    all_features_df.drop('temp 6', axis=1, inplace=True)

#%%##################### PLOT MEASUREMENT TEMPERATURES ######################

# Get unique values in 'hotspot_num'
hotspot_values = all_features_df['hotspot_num'].unique()

# Filter columns that contain the word 'temp' and convert them to float
temp_df = all_features_df.filter(regex='[Tt]emp').astype(float)
time_stamps = pd.to_datetime(all_features_df['time_stamp'])

# Define the filename
filename = os.path.join(export_path, "temperature of signals.pdf")

# Plot
plt.figure(figsize=(8, 6))
for col in temp_df.columns:
    plt.scatter(time_stamps, temp_df[col], s=10, label=f'{col}')

plt.xlabel('Time of Day')
plt.ylabel('Temperature')
plt.title('Temperature of signals')
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))  # Format x-axis ticks to show only time
plt.grid(True, linestyle='--')
plt.legend(markerscale=5)
plt.tight_layout()
plt.savefig(filename)

plt.show()

#%% ################# SAVE FEATURES ##########################

# Extract everything past "Data" in export_path and remove backslashes
filename_str = export_path.split("Data")[-1].replace("\\", " ").strip()
# Define the filename
filename = os.path.join(export_path, f"{filename_str} features.csv")
# Save calculated features to file with the specified filename
all_features_df.to_csv(filename, index=False)
print("Features saved to csv at:", filename)

# %%
