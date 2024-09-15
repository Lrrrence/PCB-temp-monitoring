# PCB temperature monitoring using ultrasonic guided waves and machine learning

This repository contains the raw data, python scripts, and results, of predicting the temperature of PCB components from ultrasonic guided wave signals using machine learning algorithms. The process is demonstrated on three different PCBs, each with five hotspot positions (temperature controlled resistors).

## Method

Piezo Wafer Active Sensors (PWAS) are used to generate and detect ultrasonic guided wave signals. The sensors are placed at the edges of the test PCBs, away from the components. They measure 6.50 × 0.27 mm with wrap-around electrode patterns (APC 851/Navy Type II/PZT-5A: $d_{33}$ $400×10^{−12}$ m/V, $d_{31}$ $-210×10^{−12}$ m/V, $Q_m$ 80).

An arbitrary waveform generator (GW Instek MFG-2230M) is used to generate 300 kHz, five-cycle Hann windowed pulses, internally triggered at a rate of 100 ms. A Picoscope 3406D MSO USB oscilloscope is used to digitise the signals, at a sampling rate of 5 MHz. Signals are captured on a trigger from the excitation pulse, 200 μs with a 5% pre-trigger. The temperature of the components and the boards themselves are monitored using an infrared camera (Xinfrared T2S Plus), placing measurement points at the centre of each hotspot position.

In all cases the voltage across each resistor (PCB#1/2: 0805 type (2012 metric), 200Ω, 0.5W. PCB#3: 2512-type (6432 metric), 200Ω, 0.5W.) is incremented by 0.5 V, which corresponds to ∼5◦C, from 3.5–9.0 V, or 30–90◦C. The temperature of each hotspot is allowed to stabilise for 3 mins before signals are captured. Each hotspot position is ramped up in temperature over the full range, and then allowed to return to room temperature, before moving to the next hotspot position.

A large number of features (103) are calculated from each signal, which can be split into four sections, overall features, binned features, peak amplitudes, and peak frequency components. The overall features are made up of the mean, standard deviation, skewness, kurtosis, median, min, max, range, sum, and variance. These features are then calculated for different parts of the signal, after splitting into equally spaced bins (default 10). Finally, the envelope of the original signal is calculated, and the amplitude of the most prominent peaks (default 5) are measured. The frequency and magnitude of the most prominent peaks (default 5) are derived from FFT.

**There are three test PCBs:**

<img src="Data/PCB%231/PCB1_edit.png" width="30%"> <img src="Data/PCB%232/PCB2_edit.png" width="30%"></img> <img src="Data/PCB%233/PCB3_edit.png" width="30%"></img>

- PCB#1
    - One receiver, five spread out resistor hotspots.
    - Two datasets, `single` and `avg`.
        - `single` contains one capture per trigger. 2696 samples.
        - `avg` contains 64 captures (buffer size of DAQ) per trigger. 75 samples.
- PCB#2
    - Two receivers, five inline resistor hotspots.
    - One dataset, `avg`. 332 samples.
- PCB#3
    - Two receivers, five hotspots applied externally via resistors to board ICs.
    - One dataset, `single`. 2002 samples.

All data processing is carried out in python, as described below. The [sci-kit learn package](https://scikit-learn.org/stable/index.html) is used for feature selection, model training, and evaluation. A range of regression models are trained and ranked based on root mean squared error (RMSE) and R$^2$. 

## Setup the python virtual environment in VSCode

- install python 3.12
- clone the repo and open the folder in VSCode
- open a terminal window
- create a virtual environment with: `py -3.12 -m venv .venv`
- activate it with: `.venv/scripts/activate`
- install the required packages with: `pip install -r requirements.txt`

## Processing

The first stage of the processing (`process_data.py`) takes the raw data ("Waveforms" folder), combines it with IR camera (Xinfrared T2S Plus) measurements for model training/validation, and then generates a selection of features from the receiver waveforms. These features are then stored in a `.csv` file, to be passed to the ML stage. The `test_processing.py` script can be used to understand how the features are calculated on a single sample.

The function for reading in signals is changed automatically, depending on the type. `func_read_signals` or `func_read_signals_average`. The average version takes a mean average of the 64 signals within each buffer capture folder (present for PCB#1 & #2), resulting in significantly less data than the `single` method. Note that for PCB#1 the datasets are unique for each method.

Within each data folder there is a `temp_log.txt` file which contains the temperature of each component, as measured by IR camera. This file is read in to `process_data.py` and used at line 85, where the closest measurement by timestamp from `temp_log.txt` is matched up and stored with the signal data. Temp 1-5 are the component temperatures, temp 6 is the minimum temperature of the board (not used).

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

## Results

Overall, `ExtraTreesRegressor' is the best performing regression model on all three PCBs, achieving an average RMSE of <3.5◦C, and an R2 of ≥0.95. This is based on the use of the 20 most important features, as determined by mutual information score.
