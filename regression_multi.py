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

# load features 
features_df = pd.read_csv(path)

# remove rows where ch_num = 2
# features_df = features_df[features_df['ch_num'] != 3]

# extract target variables
y_clf = features_df['hotspot_num']
y_reg = features_df[['temp 1', 'temp 2', 'temp 3', 'temp 4', 'temp 5']]

# select columns to drop
dropped_columns = [col for col in features_df.columns if 'temp' in col.lower()] + ['hotspot_num', 'time_stamp', 'Filename', 'ch_num']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(features_df.drop(columns=dropped_columns), # input features
                                                   	 						y_reg,  # target
                                                    						test_size=0.2, # split
                                                    						#random_state=42
                                                                            ) 

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", Y_train.shape)
print("Shape of y_test:", Y_test.shape)

#%% test

from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import set_config
set_config(transform_output="pandas")

from sklearn import linear_model, ensemble, gaussian_process, isotonic, kernel_ridge, neural_network, svm, tree, neighbors
from sklearn.dummy import DummyRegressor

# scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# select best features by mutual information score
# Initialize SelectKBest with mutual_info_regression
selector = SelectKBest(score_func=mutual_info_regression, k=20)
# Fit selector to training data
selector.fit(X_train_scaled, Y_train['temp 1'])
# Get indices of selected features
selected_indices = selector.get_support(indices=True)
# Get selected feature names
selected_feature_names = X_train.columns[selected_indices]
# Transform training and test data to selected features
X_train = pd.DataFrame(selector.transform(X_train_scaled), columns=selected_feature_names)
X_test = pd.DataFrame(selector.transform(X_test_scaled), columns=selected_feature_names)

# select models
reg_models = [
    # Linear models
    linear_model.LinearRegression(),
    linear_model.LassoCV(max_iter=1000000, tol=1e-4),  
    linear_model.ElasticNetCV(max_iter=100000),  
    linear_model.BayesianRidge(),  
    # linear_model.ARDRegression(),  
    # #linear_model.SGDRegressor(max_iter=10000),  

    # # # Model for large-scale learning
    # linear_model.PassiveAggressiveRegressor(),  

    # # # Robust-to-outliers models
    # linear_model.RANSACRegressor(),  
    # linear_model.HuberRegressor(max_iter=10000),  
    # linear_model.TheilSenRegressor(),  

    # # Tree based
    tree.DecisionTreeRegressor(),  

    # # # Ensemble methods
    ensemble.RandomForestRegressor(),  
    ensemble.ExtraTreesRegressor(),  
    #ensemble.AdaBoostRegressor(),  
    ensemble.GradientBoostingRegressor(),  
    ensemble.HistGradientBoostingRegressor(),  

    # # # Support Vector Machines
    # svm.LinearSVR(dual="auto"),  
    # svm.NuSVR(),  
    # svm.SVR(),  

    # # # K-Nearest Neighbors
    neighbors.KNeighborsRegressor(),  

    # # # Neural network
    neural_network.MLPRegressor(max_iter=100000),  

    # # # Dummy regressor - just predicts the mean for all rows. Acts as a good benchmark
    # DummyRegressor(strategy="mean")
]

# define kfold method
cv = KFold(n_splits=5, shuffle=True)

# initialise outputs
results_list = []
best_model = None
best_rmse = np.inf  # Start with a very high value

# Train all models with pipeline
all_results = pd.DataFrame(columns=["Model", "RMSE", "RMSE st.dev", "R²", "R² st.dev", "k_value"])

# Compute metrics for each model
avg_rmse = []
std_rmse = []
avg_r2 = []
std_r2 = []

for model in reg_models:
    pipeline = Pipeline([
        # ('scaler', StandardScaler()),
        # ('selector', SelectKBest(mutual_info_regression, k=k)),
        ('regressor', MultiOutputRegressor(model))
    ])

    # Perform cross-validation and calculate RMSE for each target
    rmse_scores = abs(cross_val_score(pipeline, X_train, Y_train, cv=cv, scoring='neg_root_mean_squared_error'))
    avg_rmse_i = rmse_scores.mean().round(2)  # Compute mean RMSE across targets
    std_rmse_i = np.std(rmse_scores).round(2)  # Compute std dev of RMSE across targets
    avg_rmse.append(avg_rmse_i)
    std_rmse.append(std_rmse_i)

    # Perform cross-validation and calculate R² for each target
    r2_scores = cross_val_score(pipeline, X_train, Y_train, cv=cv, scoring='r2')
    avg_r2_i = r2_scores.mean().round(2)  # Compute mean R² across targets
    std_r2_i = np.std(r2_scores).round(2)  # Compute std dev of R² across targets
    avg_r2.append(avg_r2_i)
    std_r2.append(std_r2_i)

    # Print and store results for each model
    print(f"{model.__class__.__name__} - RMSE {avg_rmse_i:.2f} +/- {std_rmse_i:.2f}, R² {avg_r2_i:.2f} +/- {std_r2_i:.2f}")
    results_list.append({"Model": model.__class__.__name__, "RMSE": avg_rmse_i, "RMSE st.dev": std_rmse_i, "R²": avg_r2_i, "R² st.dev": std_r2_i})

    # Update best model if the current model performs better
    if avg_rmse_i < best_rmse:
        best_rmse = avg_rmse_i
        best_model = pipeline  # Update the best model

# Convert results_list to a DataFrame and sort
all_results = pd.DataFrame(results_list)
all_results["RMSE Rank"] = all_results["RMSE"].rank(method='first')
all_results["R² Rank"] = all_results["R²"].rank(ascending=False, method='first')
sorted_results = all_results.sort_values(by="RMSE Rank")
sorted_results = sorted_results.reset_index(drop=True) # Resetting the index
print(sorted_results)

best_model_name = sorted_results.loc[0, 'Model']
print(f"Best performing model: {best_model_name} (RMSE: {best_rmse:.2f})")
print()

# save results to xlsx
filename = os.path.join(folder_path, 'reg_results.xlsx')
sorted_results.to_excel(filename, index=False)
print(f"DataFrame saved to '{folder_path}'")

#%% predict on test set
from sklearn.metrics import r2_score

# Fit the best model on the entire training data
best_model.fit(X_train, Y_train)

# get the features names of the best performing model
feature_names = best_model[:-1].get_feature_names_out()

# Predict on the test data using the best model
y_pred = best_model.predict(X_test)

# Calculate RMSE for each target variable
rmse = root_mean_squared_error(Y_test, y_pred, multioutput='raw_values')

# Calculate R^2 for each target variable
r2 = r2_score(Y_test, y_pred, multioutput='raw_values')

# store in df
accuracy_df = pd.DataFrame(columns=['Target', 'RMSE', 'R^2'])
for i in range(5):
    target_name = f"Target {i+1}"
    accuracy_df.loc[i] = [target_name, round(rmse[i], 2), round(r2[i], 2)]
print(accuracy_df)

# get overall RMSE
test_rmse = root_mean_squared_error(Y_test, y_pred)
print(f"Overall test RMSE using the best model ({best_model_name}): {test_rmse:.2f}")
r2 = r2_score(Y_test, y_pred)
print(f"Overall R^2 score using the best model ({best_model_name}): {r2:.2f}")

# to df
y_pred = pd.DataFrame(y_pred)
# Extract the indices from y_test
indices= Y_test.index
# combine indices
y_pred = y_pred.set_index(indices)


#%% plot
# plot regression for each hotspot position

import seaborn as sns
import os

num_columns = Y_test.shape[1]  # Number of columns in Y_test 
colors = sns.color_palette('husl', num_columns)  # Generate a list of colors for each column pair

for i in range(num_columns):
    plt.figure(figsize=(6, 6))
    sns.regplot(x=Y_test.iloc[:, i], y=y_pred.iloc[:, i],
                scatter_kws={'alpha': 0.5, 'color': colors[i]},
                line_kws={'alpha': 0.5, 'ls': '--', 'color': 'grey'})
    
    target_key = f'Target {i+1}'  # Construct key based on loop index
    rmse_value = accuracy_df.loc[accuracy_df['Target'] == target_key, 'RMSE'].values[0]
    r2_value = accuracy_df.loc[accuracy_df['Target'] == target_key, 'R^2'].values[0]
    
    plt.xlabel('True Values (°C)')
    plt.ylabel('Predictions (°C)')
    plt.title(f'{best_model_name} (RMSE: {rmse_value}, R²: {r2_value}) - Hotspot {i+1}')
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    
    # Save each plot with model name and column index in the filename
    plot_filename = os.path.join(folder_path, f'regression_{best_model_name}_hotspot_{i+1}.pdf')
    plt.savefig(plot_filename)
    plt.show(block=False)

# %%
