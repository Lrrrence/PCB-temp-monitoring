# PCB temperature monitoring using ultrasonic guided waves and machine learning

This repository contains the raw data and associated python scripts to predict the temperature of PCB components from ultrasonic guided wave signals. The process is demonstrated on three different PCBs, each with five hotspot positions (temperature controlled resistors). Piezo Wafer Active Sensors (PWAS) are used to generate and detect transmitted signals.

## Method

An arbitrary waveform generator (GW Instek MFG-2230M) is used to generate a 300 kHz, five-cycle Hann windowed pulse, internally triggered at a rate of 100 ms. A Picoscope 3406D MSO USB oscilloscope is used to digitise the signals, at a sampling rate of 5 MHz. Signals are captured on a trigger from the excitation pulse, 200 μs with a 5% pre-trigger. A high-pass filter is used to remove any DC offset. The temperature of the components and the boards themselves are monitored using an infrared camera (Xinfrared T2S Plus), placing measurement points at the centre of each hotspot position.

All data processing is carried out in python, as described below. 

## Setup the virtual environment in VSCode

- install python 3.12
- clone the repo and open the folder in VSCode
- open a terminal window
- create a virtual environment with: `py -3.12 -m venv .venv`
- activate it with: `.venv/scripts/activate`
- install the required packages with: `pip install -r requirements.txt`

## Processing

The first stage of the processing (`process_data.py`) takes the raw data ("Waveforms" folder), combines it with traditional temperature monitoring data from thermocouples/IR cameras (for model training/validation), and then generates a selection of features from the receiver waveforms. These features are then stored in a `.csv` file for the ML stage. 

There are three test PCBs:
- PCB#1
    - One receiver, five spread out resistor hotspots.
    - Two datasets, `single` and `avg`.
        - `single` contains one capture per trigger.
        - `avg` contains 64 captures (buffer size of DAQ) per trigger.  
- PCB#2
    - Two receivers, five inline resistor hotspots.
    - One dataset, `avg`.
- PCB#3
    - Two receivers, five hotspots applied externally via resistors to board ICs.
    - One dataset, `single`.

The function for reading in signals is changed automatically, depending on the type. `func_read_signals` or `func_read_signals_average`.

Within each data folder there is a `temp_log.txt` file which contains the temperature of each component (as measured by thermocouple or IR camera). This file is read in to `PCB_ML_main.py` and used at around line 74, where the closest measurement by timestamp from `temp_log.txt` is matched up and stored with the signal data. Temp 1-5 are the component temperatures, temp 6 is the minimum temperature of the board (not used).

Processing stages of `process_data.py`:
1. Import dependencies and set file paths.
2. Choose number of receivers.
3. Loop by `ch_num`:
    1. Plot the first signal to check it's as expected.
    2. Calculate features using `calculate_features`, where the number of bins and peaks can be set.
    3. Extract hotspot number and temperature from `temp_log.txt`.
    4. Combine temperature measurements with calculated features.
4. Export the final dataset.
5. Plot temperature over time to visualise the experimental process.

## Functions

 - `process_data.py` reads in raw data, combines with temperature
   measurements, and generates features. 
 - `test_processing.py` to see how
   peaks are found, bins are split, and FFT is calculated on a single
   file, outside of the function. 
- `regression_multi.py` to compare
   regression models and find the most effective.
- `regression_ExtraTrees.py` trains a single model
   (ExtraTreesRegressor). 
- `func_read_signals.py` contains the functions
   that read either buffered (`read_signals_avg`) or single (`read_signals_single`) waveforms.
- `func_calc_features.py` contains the function `calculate_features` that generates
   features.
- `univariate_feature_selection.py` uses SelectKBest to determine the most important features, based on F-test and Mutual Information score. These are plotted alongside a correlation matrix, which can be subset for easier analysis. If there are pairs of highly correlated features, one can be removed. The second half of the script trains multiple regression models with an increasing number of the best features, as determined by mutual information score, and plots the change in RMSE. 
- `spearman_correlation` use this to generate a dendrogram based on Spearman correlation, and see the effect on an ExtraTreesRegressor model of dropping highly correlated features.
