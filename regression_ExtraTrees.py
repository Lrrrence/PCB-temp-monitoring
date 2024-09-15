# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:00:19 2024

@author: Lrrr
"""

#%% load features from file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the folder path where the file is located
folder_path = os.path.join(os.path.dirname(__file__), "Data", "PCB#3", "single")

# Find any file that has 'features.csv' in the name in the specified folder
for file in os.listdir(folder_path):
    if 'features.csv' in file:
        path = os.path.join(folder_path, file)
        break
else:
    raise FileNotFoundError("No file with 'features.csv' in the name found in the specified folder")

# get filename
filename = os.path.splitext(os.path.basename(path))[0]
folder_path = os.path.dirname(path)
folder_name = os.path.basename(os.path.dirname(path))

# load features 
features_df = pd.read_csv(path)

# extract target variables
y_clf = features_df['hotspot_num']
y_reg = features_df[['temp 1', 'temp 2', 'temp 3', 'temp 4', 'temp 5']]

# select columns to drop
dropped_columns = [col for col in features_df.columns if 'temp' in col.lower()] + ['hotspot_num', 'time_stamp', 'Filename', 'ch_num']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(features_df.drop(columns=dropped_columns), # input features
                                                    y_reg,  # target
                                                    test_size=0.2) 

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", Y_train.shape)
print("Shape of y_test:", Y_test.shape)

#%% train model with default parameters

from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import ExtraTreesRegressor

# scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# select best features by mutual information score
selector = SelectKBest(score_func=mutual_info_regression, k=20)
selector.fit(X_train_scaled, Y_train['temp 1'])
selected_indices = selector.get_support(indices=True)
selected_feature_names = X_train.columns[selected_indices]
X_train = pd.DataFrame(selector.transform(X_train_scaled), columns=selected_feature_names)
X_test = pd.DataFrame(selector.transform(X_test_scaled), columns=selected_feature_names)

# Define kfold method
cv = KFold(n_splits=5, shuffle=True)

# Train model with default parameters
model = MultiOutputRegressor(ExtraTreesRegressor())
model.fit(X_train, Y_train)

# Print the hyperparameters of the model
print("Hyperparameters of the model:")
print(model.get_params())

# Perform cross-validation and calculate R² for each target
r2_scores = cross_val_score(model, X_train, Y_train, cv=cv, scoring='r2')
avg_r2 = r2_scores.mean().round(2)
std_r2 = np.std(r2_scores).round(2)

# Calculate RMSE for each target variable
y_pred_train = model.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(Y_train, y_pred_train, multioutput='raw_values'))

# Print results
print(f"Cross-validated R²: {avg_r2} ± {std_r2}")
print(f"Train RMSE: {rmse_train}")

#%% predict on test set
from sklearn.metrics import r2_score

# Predict on the test data using the trained model
y_pred = model.predict(X_test)

# Calculate RMSE for each target variable
rmse = np.sqrt(mean_squared_error(Y_test, y_pred, multioutput='raw_values'))

# Calculate R^2 for each target variable
r2 = r2_score(Y_test, y_pred, multioutput='raw_values')

# store in df
accuracy_df = pd.DataFrame(columns=['Target', 'RMSE', 'R^2'])
for i in range(5):
    target_name = f"Target {i+1}"
    accuracy_df.loc[i] = [target_name, round(rmse[i], 2), round(r2[i], 2)]
print(accuracy_df)

# get overall RMSE
test_rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
print(f"Overall test RMSE: {test_rmse:.2f}")
r2 = r2_score(Y_test, y_pred)
print(f"Overall R^2 score: {r2:.2f}")

# to df
y_pred = pd.DataFrame(y_pred)
indices = Y_test.index
y_pred = y_pred.set_index(indices)

#%% plot

import seaborn as sns

num_columns = Y_test.shape[1]
colors = sns.color_palette('husl', num_columns)

for i in range(num_columns):
    plt.figure(figsize=(6, 6))
    sns.regplot(x=Y_test.iloc[:, i], y=y_pred.iloc[:, i],
                scatter_kws={'alpha': 0.5, 'color': colors[i]},
                line_kws={'alpha': 0.5, 'ls': '--', 'color': 'grey'})
    
    target_key = f'Target {i+1}'
    rmse_value = accuracy_df.loc[accuracy_df['Target'] == target_key, 'RMSE'].values[0]
    r2_value = accuracy_df.loc[accuracy_df['Target'] == target_key, 'R^2'].values[0]
    
    plt.xlabel('True Values (°C)')
    plt.ylabel('Predictions (°C)')
    plt.title(f'ExtraTreesRegressor (RMSE: {rmse_value}, R²: {r2_value}) - Hotspot {i+1}')
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    
    plt.show()
# %%
