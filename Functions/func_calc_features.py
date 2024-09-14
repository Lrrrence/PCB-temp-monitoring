# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:00:27 2024

@author: Lrrr
"""

def calculate_features(signals_df, time_df, bin_num=10, amp_peak_num=5, fft_peak_num=5):

    import numpy as np
    import pandas as pd
    import scipy.stats
    from scipy.signal import hilbert, find_peaks

    ################ main features first ##############
        
    main_features_df = pd.DataFrame()
    
    # Iterate over each column in the DataFrame
    for column_name, column_values in signals_df.items(): 
    
        # Calculate features for the current column
        features = {
            'mean': np.mean(column_values),
            'std_dev': np.std(column_values),
            'skewness': scipy.stats.skew(column_values),
            'kurtosis': scipy.stats.kurtosis(column_values),
            'median': np.median(column_values),
            'min': np.min(column_values),
            'max': np.max(column_values),
            #'range': np.max(column_values) - np.min(column_values),
            'sum': np.sum(column_values),
            #'variance': np.var(column_values),
            # 'percentile_10': np.percentile(column_values, 10),
            # 'percentile_25': np.percentile(column_values, 25),
            # 'percentile_50': np.percentile(column_values, 50),  # Median
            # 'percentile_75': np.percentile(column_values, 75),
            # 'percentile_90': np.percentile(column_values, 90),
            # 'interquartile_range': np.percentile(column_values, 75) - np.percentile(column_values, 25),
            # 'quartile_1': np.percentile(column_values, 25),  # Q1
            # 'quartile_2': np.percentile(column_values, 50),  # Q2 (Median)
            # 'quartile_3': np.percentile(column_values, 75),  # Q3
            # 'decile_10': np.percentile(column_values, 10),
            # 'decile_20': np.percentile(column_values, 20),
            # 'decile_30': np.percentile(column_values, 30),
            # 'decile_40': np.percentile(column_values, 40),
            # 'decile_50': np.percentile(column_values, 50),
            # 'decile_60': np.percentile(column_values, 60),
            # 'decile_70': np.percentile(column_values, 70),
            # 'decile_80': np.percentile(column_values, 80),
            # 'decile_90': np.percentile(column_values, 90),
            # Add more features as needed
        }
    
        # Create DataFrame with features for the current column
        features_df = pd.DataFrame([features], index=[column_name])  # Set the index to column_name
    
        # Concatenate features_df with main_features_df
        main_features_df = pd.concat([main_features_df, features_df]) 
        
    # Rename the index name to 'Signal'
    main_features_df.index.name = 'Signal'
    
    ############## FFT features ################
    
    # number of peaks
    fft_peak_num = 5
    
    # Prepare the result DataFrame with appropriate columns
    fft_result_columns = [f'fft_peak_{i+1}_freq' for i in range(fft_peak_num)] + [f'fft_peak_{i+1}_mag' for i in range(fft_peak_num)]
    fft_result_df = pd.DataFrame(columns=fft_result_columns)
    
    # get sample rate
    time_difference = time_df.iloc[1, 0] - time_df.iloc[0, 0]
    fs = (1 / time_difference) * 1e6
    
    for column in signals_df.columns:
        
        chB = signals_df[column].values
        
        # Zero-pad the signal to double its length
        chB_pad = np.pad(chB, (0, len(chB)), 'constant')
        
        # Compute FFT of the zero-padded signal
        fft_vals = np.fft.fft(chB_pad)
        fft_freqs = np.fft.fftfreq(len(chB_pad), 1.0 / fs)
        
        # Only consider the positive half of the spectrum
        positive_freq_indices = np.where(fft_freqs >= 0)
        positive_fft_vals = fft_vals[positive_freq_indices]
        positive_fft_freqs = fft_freqs[positive_freq_indices]
        
        # Compute magnitude spectrum
        magnitude_spectrum = np.abs(positive_fft_vals)
        
        # Find peaks in the magnitude spectrum
        peaks, _ = find_peaks(magnitude_spectrum)
        
        # Get the magnitudes of the peaks
        peak_magnitudes = magnitude_spectrum[peaks]
        
        # Find the indices of the largest peaks
        top_indices = np.argsort(peak_magnitudes)[-fft_peak_num:]
        
        # Get the corresponding frequencies and magnitudes of the top 5 peaks
        top_freqs = positive_fft_freqs[peaks][top_indices]
        top_magnitudes = peak_magnitudes[top_indices]
        
        # Sort the results by frequency
        sorted_indices = np.argsort(top_freqs)
        top_freqs = top_freqs[sorted_indices]
        top_magnitudes = top_magnitudes[sorted_indices]
        
        # Concatenate the arrays vertically
        result_df = pd.DataFrame(np.concatenate((top_freqs, top_magnitudes)))
        result_df = result_df.transpose() # Transpose 
        result_df.columns = fft_result_columns  # Set column names 
        
        # Concatenate the original DataFrame with the new row DataFrame
        fft_result_df = pd.concat([fft_result_df, result_df], ignore_index=True)
    
    ############## amplitudes per peak ####################
        
    num_top_peaks= amp_peak_num
    prominence_threshold=0.5
    peak_dfs = {}
    for col in signals_df.columns:
        signal = signals_df.iloc[:,col]
        time_col = time_df.iloc[:,col]
        # Compute the analytical signal
        analytical_signal = hilbert(signal)
        # Compute the envelope (absolute value of the analytical signal)
        envelope = np.abs(analytical_signal)
        # Find peaks in the envelope and get their properties
        peaks, properties = find_peaks(envelope, prominence=prominence_threshold)
        # Sort the peak indices based on their prominences
        sorted_peak_indices = sorted(peaks, key=lambda x: properties["prominences"][list(peaks).index(x)], reverse=True)
        # Select the indices of the top peaks
        top_peak_indices = sorted_peak_indices[:num_top_peaks]
        # Get the amplitudes of the top peaks
        top_peak_amplitudes = envelope[top_peak_indices]
        # Get the time corresponding to the top peak indices
        top_peak_times = time_col[top_peak_indices]
        # Combine the results into a dictionary
        peak_data = {
            "Peak Times": top_peak_times.values,
            "Peak Indices": top_peak_indices,
            "Peak Prominences": [round(properties["prominences"][i], 2) for i in range(num_top_peaks)],
            "Peak Amplitudes": [round(top_peak_amplitudes[i], 2) for i in range(num_top_peaks)]
        }
        # Create a DataFrame from the dictionary
        peak_df = pd.DataFrame(peak_data)
        peak_dfs[col] = peak_df

    # Initialize an empty list to store 'Peak Amplitudes' from each DataFrame
    peak_amplitudes_list = []
    
    # Iterate over the dictionary entries
    for key, df in peak_dfs.items():
        # Extract 'Peak Amplitudes' column and append to the list
        peak_amplitudes_list.append(df['Peak Amplitudes'])
    
    # Concatenate the list of 'Peak Amplitudes' into a single DataFrame
    peak_amplitudes_df = pd.concat(peak_amplitudes_list, axis=1).T
    
    # reset index
    peak_amplitudes_df.reset_index(drop=True, inplace=True)
    
    # Get the number of columns in the DataFrame
    n = peak_amplitudes_df.shape[1]

    # Generate new column names dynamically
    new_columns = {i: f'peak{i+1}amp' for i in range(n)}

    # Rename columns
    peak_amplitudes_df.rename(columns=new_columns, inplace=True)
    
    ############# equally spaced bins ##################
    
    # Number of bins
    #num_bins = 10
    
    # Dictionary to store split data for each column
    split_data = {}
    
    # Loop through each column
    for column in signals_df.columns:
        # Extract the column data
        ch_data = signals_df[column].values
    
        # Calculate the number of samples per bin
        #samples_per_bin = len(ch_data) // bin_num
    
        # Split signal into bins
        bins = np.array_split(ch_data, bin_num)
    
        # Store the split data in the dictionary
        split_data[column] = bins
    
    # Function to calculate features for each bin
    def calculate_bin_features(column_values):
        bin_features = {
            'mean': np.mean(column_values),
            'std_dev': np.std(column_values),
            'skewness': scipy.stats.skew(column_values),
            'kurtosis': scipy.stats.kurtosis(column_values),
            'median': np.median(column_values),
            'min': np.min(column_values),
            'max': np.max(column_values),
            #'range': np.max(column_values) - np.min(column_values),
            'sum': np.sum(column_values),
            #'variance': np.var(column_values)
        }
        return bin_features
    
    # Initialize a list to store feature rows
    feature_rows = []
    
    # Iterate over each signal (row) in split_data
    for signal_name, signal_bins in split_data.items():
        print(f"Calculating features for signal: {signal_name}")
        
        # Initialize a list to store features for this signal
        signal_features = []
        
        # Iterate over each bin in the signal
        for bin_data in signal_bins:
            # Calculate features for the bin
            bin_features = calculate_bin_features(bin_data)
            # Extract feature values and append them to the list of features for this signal
            signal_features.extend(bin_features.values())
        
        # Append the features for this signal to the list of feature rows
        feature_rows.append(signal_features)
    
    # Create column names for the features
    columns = [f'bin{i}_{feature}' for i in range(1, bin_num+1) for feature in bin_features.keys()]
    
    # Create a DataFrame from the feature rows
    bin_features_df = pd.DataFrame(feature_rows, columns=columns)
    
    # Assuming all signals have the same number of bins
    num_bins = len(next(iter(split_data.values())))
    
    # Create column names for the features
    columns = [f'bin{i}_{feature}' for i in range(1, num_bins+1) for feature in bin_features.keys()]
    
    # Initialize a list to store feature rows
    feature_rows = []
    
    # Iterate over each signal (row) in split_data
    for signal_bins in split_data.values():
        # Initialize a list to store features for this signal
        signal_features = []
        
        # Iterate over each bin in the signal
        for bin_data in signal_bins:
            # Calculate features for the bin
            bin_features = calculate_bin_features(bin_data)
            # Extract feature values and append them to the list of features for this signal
            signal_features.extend(bin_features.values())
        
        # Append the features for this signal to the list of feature rows
        feature_rows.append(signal_features)
    
    # Create a DataFrame from the feature rows
    bin_features_df = pd.DataFrame(feature_rows, columns=columns)
    
    ############ combine all features ###########
    combined_features_df = pd.concat([main_features_df, peak_amplitudes_df, bin_features_df, fft_result_df], axis=1)
        
    return combined_features_df




