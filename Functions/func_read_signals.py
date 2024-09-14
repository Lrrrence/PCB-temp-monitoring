# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:52:33 2024

@author: Lrrr
"""

def read_signals_single(folder_path, ch_num):

    import pandas as pd
    import numpy as np
    import os
    
    def read_txt_files(folder_path):
        txt_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    txt_files.append(file_path)
        return txt_files
    
    def get_ctime(file_path):
        # Get creation time in nanoseconds
        ctime = os.stat(file_path).st_ctime
        return ctime
    
    # Read all .txt files within the folder and its subfolders
    txt_files = read_txt_files(folder_path)
    
    # Sort the list of files by datetime modified
    txt_files.sort(key=lambda x: get_ctime(x))
    
    #remove temp_log
    txt_files = list(filter(lambda x: "temp_log.txt" not in x and "DataLogSession" not in x and "CombinedDataLog" not in x, txt_files))
          
    ########## calculate sample rate ###############
    
    # Read the first file into a DataFrame
    df = pd.read_csv(txt_files[0], delimiter='\t', header=1)
    # Convert entire DataFrame to numeric
    df = df.apply(pd.to_numeric)
    time = df.iloc[:, 0].values
    # Calculate the difference between the first two rows
    time_difference = time[1] - time[0]
    # Calculate the sample rate
    fs = (1 / time_difference) * 1e6
    print("Sample Rate: {:.2f}".format(fs))
    
    ################ high pass filter #############
    
    from scipy import signal
    
    # Desired cutoff frequencies (kHz)
    low_cutoff = 50000  # 50 kHz
    high_cutoff = 1e6 # 1 MHz
    
    # Calculate Nyquist frequency
    nyquist_frequency = fs / 2
    
    # Normalize cutoff frequencies with respect to the Nyquist frequency
    low_cutoff_normalized = low_cutoff/ nyquist_frequency
    high_cutoff_normalized = high_cutoff/ nyquist_frequency
    
    # Define filter order
    N = 1  # First-order filter
    
    # Design bandpass filter
    b, a = signal.butter(N, [low_cutoff_normalized, high_cutoff_normalized], btype='band')
    
    import datetime
    import pandas as pd
    
    #init
    data_list=[]
    filename_list=[]
    creation_datetime_list=[]
    time_df_list=[]
    
    # Loop through each file, extract and store data, filename, and last modified time
    for file in txt_files:
        # Extract filename
        filename = file.split('\\')[-1].split('Waveforms\\')[-1].replace('.txt', '')
        
        # Get creation time 
        creation_time = os.stat(file).st_mtime
        
        # Convert to a datetime object
        creation_datetime = datetime.datetime.fromtimestamp(creation_time)

        # Print filename and last modified time being read
        print("Reading file:", filename, "Last modified time:", creation_datetime)
        
        # Read the file into a DataFrame
        df = pd.read_csv(file, delimiter='\t', header=1)  # Adjust delimiter as needed
        
        # Convert entire DataFrame to numeric, coerce errors to NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Check if there are any NaN values in the DataFrame
        if df.isna().any().any():
            print(f"Skipping file {file} due to NaN values")
            continue
        
        # Convert entire DataFrame to numeric
        df = df.apply(pd.to_numeric)

        time = df.iloc[:, 0].values
        rec = df.iloc[:, ch_num].values
        
        # Apply high-pass filter
        rec = signal.filtfilt(b, a, rec)
        
        # Remove DC offset by subtracting the mean
        rec -= rec.mean()
        
        # Append to the data list
        data_list.append(rec)
        
        # Append the filename to the filename list
        filename_list.append(filename)
        
        # Append the last modified time to the last modified time list
        creation_datetime_list.append(creation_datetime)
        
        # Create a DataFrame for time and append it to the list
        time_df_list.append(pd.DataFrame({'Time': time}))
    
    # Step 5: Create DataFrames
    signals_df = pd.DataFrame(data_list).T  # Transpose to convert rows to columns
    filename_df = pd.DataFrame({'Filename': filename_list, 'CreationTime': creation_datetime_list})
    time_df = pd.concat(time_df_list, axis=1)
            
    # Set column names to integers
    signals_df.columns = range(len(signals_df.columns))
    time_df.columns = range(len(time_df.columns))
    
    # Display the resulting DataFrames
    print("Data DataFrame:")
    print(signals_df)
    print("\nFilename DataFrame:")
    print(filename_df)
    
    return signals_df, filename_df, time_df



def read_signals_avg(folder_path, ch_num):
    
    import pandas as pd
    import os
    from datetime import datetime, timedelta
    
    def read_txt_files(folder_path):
        txt_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    txt_files.append(file_path)
        return txt_files
    
    def get_ctime(file_path):
        # Get creation time in nanoseconds
        ctime = os.stat(file_path).st_ctime
        # Convert nanoseconds to a datetime object
        ctime_datetime = datetime.fromtimestamp(ctime)
        # Add one hour to the creation time
        ctime_datetime += timedelta(hours=1)
        # Convert the updated datetime object back to nanoseconds
        ctime = ctime_datetime.timestamp()
        return ctime
    
    # Read all .txt files within the folder and its subfolders
    txt_files = read_txt_files(folder_path)
    
    # Sort the list of files by datetime modified
    txt_files.sort(key=lambda x: get_ctime(x))
    
    #remove temp_log
    txt_files = list(filter(lambda x: "temp_log.txt" not in x and "DataLogSession" not in x and "CombinedDataLog" not in x, txt_files))
          
    ########## calculate sample rate ###############
    
    # Read the first file into a DataFrame
    df = pd.read_csv(txt_files[0], delimiter='\t', header=1)
    # Convert entire DataFrame to numeric
    df = df.apply(pd.to_numeric)
    time = df.iloc[:, 0].values
    # Calculate the difference between the first two rows
    time_difference = time[1] - time[0]
    # Calculate the sample rate
    fs = (1 / time_difference) * 1e6
    print("Sample Rate: {:.2f}".format(fs))
    
    ################ high pass filter #############
    
    from scipy import signal
    
    # Desired cutoff frequencies (kHz)
    low_cutoff = 50000 # 50 kHz
    high_cutoff = 1e6 # 1 MHz
    
    # Calculate Nyquist frequency
    nyquist_frequency = fs / 2
    
    # Normalize cutoff frequencies with respect to the Nyquist frequency
    low_cutoff_normalized = low_cutoff/ nyquist_frequency
    high_cutoff_normalized = high_cutoff/ nyquist_frequency
    
    # Define filter order
    N = 1  # First-order filter
    
    # Design bandpass filter
    b, a = signal.butter(N, [low_cutoff_normalized, high_cutoff_normalized], btype='band')
    
    ###########################################
    
    # Dictionary to store signals for each subfolder
    subfolder_signals = {}
    subfolder_counts = {}
    subfolder_creation_times = {}
    
    # Iterate through each subfolder
    for txt_file in txt_files:
        folder = os.path.dirname(txt_file)
        if folder not in subfolder_signals:
            subfolder_signals[folder] = pd.DataFrame()  # Initialize DataFrame for the subfolder
            subfolder_counts[folder] = 0
            subfolder_creation_times[folder] = get_ctime(txt_file)  # Store creation time of the first file
                
        # Read the file into a DataFrame
        df = pd.read_csv(txt_file, delimiter='\t', header=1)
        
        # Convert entire DataFrame to numeric
        df = df.apply(pd.to_numeric)
        
        # Add signals from the selected column to the DataFrame for the subfolder
        subfolder_signals[folder][os.path.basename(txt_file)] = df.iloc[:, ch_num]
        subfolder_counts[folder] += 1
    
    # Compute the row-wise averages for each subfolder
    averaged_signals = {}
    for folder, df in subfolder_signals.items():
        averaged_signals[folder] = df.mean(axis=1)
    
    # Apply high-pass filter to remove DC offset and subtract the mean for each averaged signal
    for folder, averaged_signal in averaged_signals.items():
        # Apply high-pass filter
        averaged_signals[folder] = signal.filtfilt(b, a, averaged_signal)
        # Remove DC offset by subtracting the mean
        averaged_signals[folder] -= averaged_signals[folder].mean()
    
    # Create a DataFrame containing the averages for each subfolder
    averaged_df = pd.DataFrame(averaged_signals)
    
    # Rename columns to 'Average_chB_i'
    averaged_df.columns = [f'Average_chB_{i}' for i in range(len(averaged_df.columns))]
    
    # Compute the total count of files averaged in each subfolder
    subfolder_counts_df = pd.DataFrame(list(subfolder_counts.items()), columns=['Subfolder', 'Files_Averaged'])
    
    # Create a DataFrame containing subfolder names and creation times of the first file in each subfolder
    subfolder_creation_times_df = pd.DataFrame(subfolder_creation_times.items(), columns=['Subfolder', 'First_File_Creation_Time'])
    # Extract the deepest folder name as the subfolder name
    subfolder_creation_times_df['Subfolder'] = subfolder_creation_times_df['Subfolder'].apply(lambda x: os.path.basename(x))
    # Convert creation time to datetime format
    subfolder_creation_times_df['First_File_Creation_Time'] = pd.to_datetime(subfolder_creation_times_df['First_File_Creation_Time'], unit='s')
    # Rename columns
    subfolder_creation_times_df = subfolder_creation_times_df.rename(columns={"Subfolder": "Filename", "First_File_Creation_Time": "CreationTime"})

    # Display the DataFrame containing subfolder names and creation times of the first file
    print(subfolder_creation_times_df)
    
    # Display the averaged DataFrame
    print(averaged_df)
    
    # Display the DataFrame containing subfolder names and file count
    print(subfolder_counts_df)
    
    # Duplicate the time array to match the width of averaged_df
    num_columns = averaged_df.shape[1]  # Get the number of columns in averaged_df
    time_df = pd.DataFrame({f'Time_{i}': time for i in range(num_columns)})
    
    # Set column names to integers
    averaged_df.columns = range(len(averaged_df.columns))
    time_df.columns = range(len(time_df.columns))
    
    # get values
    signals_df = averaged_df
    time_df = time_df
    
    return signals_df, subfolder_creation_times_df , time_df

