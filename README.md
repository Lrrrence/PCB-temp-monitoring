# PCB-temp-monitoring
Temperature monitoring of PCB components using ultrasonic guided waves and machine learning.

This repository contains the raw data and associated python scripts to predict the temperature of PCB components from ultrasonic guided wave signals. The process is demonstrated on three different PCBs, each with five hotspot positions (temperature controlled resistors). 

Setup the virtual environment in VSCode:
- open a terminal window
- cd into the repository folder
- create the virtual environment with this command: `py -3.12 -m venv .venv`
- activate it with this command: `.venv/scripts/activate`
- Install the required packages with `pip install -r requirements.txt`

The first stage of the processing takes the raw data ("Waveforms" folder), combines it with traditional temperature monitoring data from thermocouples/IR cameras (for model training/validation), and then generates a selection of features from the receiver waveforms. These features are then stored in a `.csv` file for the ML stage. 

The data from PCB's #1 & #3 are stored in the same way, one set of signals (1x excitation signal, 2x receivers) are stored in a `.csv` file per trigger/capture. The data from PCB #2 is stored differently, each folder contains 64 signals (the buffer size of the DAQ), which are averaged before features are calculated. The function for reading in signals should be changed accordingly, `func_read_signals` or `func_read_signals_average`.

PCB#1 has 1x receiver, PCB#2/3 have 2x receivers.

Within each data folder there is a `temp_log.txt` file which contains the temperature of each component (as measured by thermocouple or IR camera). This file is read in to `PCB_ML_main.py` and used at around line 74, where the closest measurement by timestamp from `temp_log.txt` is matched up and stored with the signal data. Temp 1-5 are the component temperatures, temp 6 is the minimum temperature of the board (not used).

Processing stages of `PCB_ML_main.py`:
1. Import dependencies and set file paths.
2. Choose number of receivers and methdo fo reading signals.
3. Loop by `ch_num`:
    1. Plot the first signal to check it's as expected.
    2. Calculate features using `calculate_features`, where the number of bins and peaks can be set.
    3. Extracting hotspot number and temperature from `temp_log.txt`.
    4. Combine temperature measurements with calculated features.
4. Export the final dataset.
5. Plot temperature over time to visualise the experimental process.

Functions:
 - `process_data` reads in raw data, combines with temperature
   measurements, and generates features. 
 - `test_processing.py` to see how
   peaks are found, bins are split, and FFT is calculated on a single
   file, outside of the function. 
- `regression_multi.py` to compare
   regression models and find the most effective.
- `regression_ExtraTrees.py` trains a single model
   (ExtraTreesRegressor). 
- `func_read_signals.py` contains the functions
   that read either buffered or single waveforms.
- `func_calc_features.py` contains the function that generates
   features.
- `univariate_feature_selection.py` uses SelectKBest to determine the most important features, based on F-test and Mutual Information score. These are plotted alongside a correlation matrix, which can be subset for easier analysis. If there are pairs of highly correlated features, one can be removed. The second half of the script trains multiple regression models with an increasing number of the best features, as determined by mutual information score, and plots the change in RMSE. 
- `spearman_correlation` use this to generate a dendrogram based on Spearman correlation, and see the effect on an ExtraTreesRegressor model of dropping highly correlated features.
