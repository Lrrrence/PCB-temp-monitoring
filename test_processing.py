# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:26:37 2024

@author: Lrrr
"""
#%%

import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import find_peaks, hilbert, find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the folder path where the file is located
file = r"C:\Users\yulel\Documents\Waveforms\Set 6 (PCB#4)\20240705 (5).txt"

# Define the folder path where the file is located
folder_path = os.path.join(os.path.dirname(__file__), "Waveforms", "PCB#1", "single")

# Get the first .txt file in the folder
file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
file_list.sort()  # Sort the files to ensure consistent order
file = os.path.join(folder_path, file_list[0]) if file_list else None

if file is None:
    raise FileNotFoundError("No .txt files found in the specified folder.")

# Read the first file into a DataFrame
df = pd.read_csv(file, delimiter='\t', header=1)

# Convert entire DataFrame to numeric
df = df.apply(pd.to_numeric)

time = df.iloc[:, 0].values
chA = df.iloc[:, 1].values
chB = df.iloc[:, 2].values

# Calculate the difference between the first two rows
time_difference = time[1] - time[0]

# Calculate the sample rate
fs = (1 / time_difference) * 1e6

print("Sample Rate: {:.2f}".format(fs))

#%% ################ high pass filter #############

# Desired cutoff frequency (kHz)
nyquist_frequency = fs / 2

# Define filter order
N = 1  # Fourth-order filter

# Define lower and upper cutoff frequencies (Hz)
lowcut = 50000  # Lower cutoff frequency (Hz)
highcut = 1e6  # Upper cutoff frequency (Hz)

# Normalize the cutoff frequencies
lowcut_normalized = lowcut / nyquist_frequency
highcut_normalized = highcut / nyquist_frequency

# Create band-pass filter
b, a = signal.butter(N, [lowcut_normalized, highcut_normalized], btype='band')

# Apply high-pass filter
chA = signal.filtfilt(b, a, chA)
chB = signal.filtfilt(b, a, chB)

# Remove DC offset by subtracting the mean
chA -= chA.mean()
chB -= chB.mean()

#%% calculate envelope and find peaks

# Compute the analytical chB
analytical_chB = hilbert(chB)
envelope = np.abs(analytical_chB)

# Specify the number of top peaks to find
num_top_peaks = 5  # You can change this number as needed

# Find peaks in the envelope and get their prominences
peaks, properties = find_peaks(envelope, prominence=0.5)  # Adjust prominence threshold as needed

# Sort the peak indices based on their prominences
sorted_peak_indices = sorted(peaks, key=lambda x: properties["prominences"][list(peaks).index(x)], reverse=True)

# Select the indices of the top peaks
top_peak_indices = sorted_peak_indices[:num_top_peaks]

# Get the amplitudes of the top peaks
top_peak_amplitudes = envelope[top_peak_indices]

# Get the time corresponding to the top peak indices
top_peak_times = time[top_peak_indices].round(2)

# Combine the results into a dictionary
peak_data = {
    "Peak Times": top_peak_times,
    "Peak Indices": top_peak_indices,
    "Peak Prominences": [round(properties["prominences"][i], 2) for i in range(num_top_peaks)],
    "Peak Amplitudes": [round(top_peak_amplitudes[i], 2) for i in range(num_top_peaks)]
}

# Create a DataFrame from the dictionary
peak_df = pd.DataFrame(peak_data)

print(peak_df)

# Plot the original chB and its envelope with the top peaks
plt.figure(figsize=(10, 6))
plt.plot(time, chB, label='Original chB')
plt.plot(time, envelope, label='Envelope')
plt.plot(top_peak_times, top_peak_amplitudes, 'ro', label='Top Peaks')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Original chB and Hilbert Envelope with Top Peaks')
plt.legend()
plt.grid(True)
plt.show()

#%% create equally spaced bins

# Number of bins
num_bins = 10

# Adjust the length of the signal to ensure it's divisible by num_bins
length_adjusted = len(chB) - (len(chB) % num_bins)

# Truncate the signal to the adjusted length
chB_adjusted = chB[:length_adjusted]
time_adjusted = time[:length_adjusted]

# Calculate the number of samples per bin
samples_per_bin = length_adjusted // num_bins

# Split signal into bins
bins = np.array_split(chB_adjusted, num_bins)
time_bins = np.array_split(time_adjusted, num_bins)

#%% FFT

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

# Find the indices of the 5 largest peaks
top_5_indices = np.argsort(peak_magnitudes)[-5:]

# Get the corresponding frequencies and magnitudes of the top 5 peaks
top_5_freqs = positive_fft_freqs[peaks][top_5_indices]
top_5_magnitudes = peak_magnitudes[top_5_indices]

# Sort the results by frequency
sorted_indices = np.argsort(top_5_freqs)
top_5_freqs = top_5_freqs[sorted_indices]
top_5_magnitudes = top_5_magnitudes[sorted_indices]

# Display the results
print("Top 5 frequencies: ", top_5_freqs)
print("Top 5 magnitudes: ", top_5_magnitudes)

#%% plot bin splits

palette = sns.color_palette()
#sns.palplot(palette) 

# define colours
chA_col = palette[0] #blue
chB_col = palette[1] #orange

# Create a figure and the primary axis
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 8))

# excitation signal
ax1.plot(time, chA, label='Excitation signal', color=chA_col)
ax1.set_xlim(time_adjusted[0], time_adjusted[-1])
ax1.set_ylim(-10, 10)
ax1.grid(True, linestyle=':', linewidth=0.5)  # Add a dotted grid
ax1.tick_params(axis='y', labelcolor=chA_col)  # Increase y-axis tick label size
ax1.set_ylabel('Amplitude (V)', color=chA_col)
ax1.set_xlabel('Time (µs)')

# Create a secondary y-axis sharing the same x-axis
ax2 = ax1.twinx()

# Plot an invisible dashed line for the legend entry
ax2.plot([], [], color=chB_col, linestyle='-', label='Received signal')
ax2.plot([], [], color='gray', linestyle='--', label='Bin boundaries (1-10)')

# Plot each bin sequentially and add vertical dashed lines for bin boundaries
for i, (bin_data, bin_time) in enumerate(zip(bins, time_bins)):
    if i > 0:
        # Connect the last point of the previous bin to the first point of the current bin
        ax2.plot([prev_time[-1], bin_time[0]], [prev_data[-1], bin_data[0]], color=chB_col)
    ax2.plot(bin_time, bin_data, color=chB_col, linewidth=1.25)
    
    # Add a vertical dashed line at the start of the bin
    if i > 0:
        ax2.axvline(x=bin_time[0], color='gray', linestyle='--')
    
    prev_data = bin_data
    prev_time = bin_time

ax2.tick_params(axis='y', labelcolor=chB_col)  # Increase y-axis tick label size

# Plot the envelope and top peaks
ax2.plot(time, envelope, label='Envelope', color='k')
ax2.plot(top_peak_times, top_peak_amplitudes,'o', label='Top peaks (prominence)', color='#e80a2a')

# Add annotations for each peak
for idx, (peak_time, peak_amplitude) in enumerate(zip(top_peak_times, top_peak_amplitudes)):
    ax2.annotate(f'Peak {idx + 1}', xy=(peak_time, peak_amplitude), 
                 xytext=(peak_time, peak_amplitude + 3),
                 fontsize=8, color='black', ha='center')

ax2.set_xlabel('Time (µs)')
ax2.set_ylabel('Amplitude (mV)', color=chB_col)

#legend
# ask matplotlib for the plotted objects and their labels
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

# Set x-axis limits to match the min/max of the time values
ax2.set_xlim(time_adjusted[0], time_adjusted[-1])
ax2.set_ylim(-80,80)

# Plot FFT Magnitude Spectrum
ax3.plot(positive_fft_freqs, magnitude_spectrum)
ax3.plot(top_5_freqs, top_5_magnitudes, 'ro')  # Mark the peaks
ax3.set_title('FFT Magnitude Spectrum')
ax3.set_xlabel('Frequency [Hz]')
ax3.set_ylabel('Magnitude')
ax3.set_xlim(0, 1e6)  # Set x-axis limits from 0 to 1 MHz
ax3.grid(True, linestyle=':', linewidth=0.5)

plt.tight_layout()

# Save the plot as a PDF file
#plt.savefig('binned_signal.pdf')

plt.show()

# %%
