# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:29:22 2024

@author: Lrrr
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler

# load features from file

# Define the folder path where the file is located
folder_path = os.path.join(os.path.dirname(__file__), "Data", "PCB#1", "single")

# Find any file that has 'features.csv' in the name in the specified folder
for file in os.listdir(folder_path):
    if 'features.csv' in file:
        path = os.path.join(folder_path, file)
        break
else:
    raise FileNotFoundError("No file with 'features.csv' in the name found in the specified folder")

# Load features DataFrame from CSV file
features_df = pd.read_csv(path)

# # which channels to include
# ch_nums = [2,3]
# features_df = features_df[features_df['ch_num'].isin(ch_nums)]
# # Reset the index
# features_df.reset_index(drop=True, inplace=True)

#%% scale features

# Initialize the scaler
scaler = StandardScaler()

# Extract the features and target variables
dropped_columns = ['hotspot_num', 'time_stamp', 'Filename','ch_num']
temp_cols = [col for col in features_df.columns if 'Temp' in col or 'temp' in col]

# Combine the two lists
combined_drop = dropped_columns + temp_cols

X = features_df.drop(columns=combined_drop)
dropped_df = features_df[combined_drop]

# Fit the scaler to the features and transform them
scaled_features = scaler.fit_transform(X)
scaled_df = pd.DataFrame(scaled_features, columns=X.columns)

# re-add descriptor columns
X = pd.concat([scaled_df, dropped_df], axis=1)

# select target variables
y_class = X['hotspot_num']
y_reg = X['temp 1']

# drop descriptors
X = X.drop(columns=combined_drop)

print("Original number of features:", X.shape[1])

#%% drop highly correlated

# use this to select if one in a pair of highly correlated features should be dropped
use_correlation_reduction = False

# Calculate the correlation matrix
correlation_matrix = X.corr()
# Calculate absolute correlation values
abs_correlation_matrix = correlation_matrix.abs()
# Sort the columns based on the mean absolute correlation values with other columns
sorted_columns = abs_correlation_matrix.mean().sort_values(ascending=False).index
# Rearrange the correlation matrix based on sorted columns
sorted_correlation_matrix = correlation_matrix.reindex(sorted_columns).reindex(sorted_columns, axis=1)

# Plot the rearranged correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(sorted_correlation_matrix.iloc[:20, :20], annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Rearranged Correlation Matrix (Most Correlated Features First)')
plt.show()

if use_correlation_reduction:

    # Identify highly correlated features (above the threshold)
    threshold = 0.98
    upper_tri = sorted_correlation_matrix.where(np.triu(np.ones(sorted_correlation_matrix.shape), k=1).astype(np.bool_))

    # Find and remove one of each pair of highly correlated features
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column].abs() > threshold)]
    reduced_data = X.drop(columns=to_drop)

    # Replot the subset correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(reduced_data.iloc[:20, :20].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Subset Correlation Matrix after Removing Highly Correlated Features')
    plt.show()

    # Output which features were dropped
    print(f'Dropped features: {to_drop}')
    
    # Count the number of columns
    after_count = reduced_data.shape[1]
    print("Reduced number of features:", after_count)
    
    # Update X with the reduced data
    X = reduced_data

#%% SelectKBest using F-tests and Mutual Information

# f_regression
selector_reg = SelectKBest(score_func=f_regression, k='all')  # You can specify the number of features to select with k parameter
selector_reg.fit(X, y_reg)
feature_scores_reg = pd.DataFrame({'Feature': X.columns, 'Score': selector_reg.scores_})
selected_features_reg = feature_scores_reg.sort_values(by='Score', ascending=False)

# mutual_info
selector_mi_reg = SelectKBest(score_func=mutual_info_regression, k='all')
selector_mi_reg.fit(X, y_reg)
feature_scores_mi_reg = pd.DataFrame({'Feature': X.columns, 'Score': selector_mi_reg.scores_})
selected_features_mi_reg = feature_scores_mi_reg.sort_values(by='Score', ascending=False)


def plot_top_features(features_df, title):
    plt.figure(figsize=(10, 6))
    bars = plt.barh(features_df['Feature'][:10], features_df['Score'][:10], color="#024b7a", alpha = 0.5, zorder=2)
    for i, bar in enumerate(bars):
        if i % 2 == 1:  # Every other bar
            bar.set_color("#44a5c2")  # Set a different color
    plt.xlabel('Score')
    plt.ylabel('Feature')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', axis='x', zorder=1)  # Add grid lines underneath bars
    plt.show()

# Plot top features for SelectKBest with F-test (Regression)
plot_top_features(selected_features_reg, 'Top 10 Features - SelectKBest (F-test) for Regression')

# Plot top features for SelectKBest with mutual information (Regression)
plot_top_features(selected_features_mi_reg, 'Top 10 Features - SelectKBest (Mutual Information) for Regression')

#%% plot and save
# regression correlation matrix

# how many features to include
feat_num = 10

# Get top features selected using SelectKBest with mutual information for regression
top_features_mi_reg = selected_features_mi_reg['Feature'][:feat_num]
# Extract the top features 
top_features_data = X[top_features_mi_reg]
# Calculate the correlation matrix
correlation_matrix = top_features_data.corr()

# Plot the correlation matrix using seaborn
plt.figure(figsize=(5, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=False)
#plt.title('Correlation Matrix for Top Features (SelectKBest - Mutual Information - Regression)')
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plot_filename = os.path.join(folder_path, f'correlation matrix (top {feat_num} MutInf).pdf')
#plt.savefig(plot_filename)
plt.show()

#%% test linear regression model based on mutual information regression

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, HuberRegressor, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

selected_features_mi_reg = selected_features_mi_reg.sort_values(by='Score', ascending=False)

# Extract feature names after setting the index
selected_features = selected_features_mi_reg['Feature'].tolist()

# Splitting X into features (X) and target variable (y)
X = X[selected_features]

# number of features to test
feat_n = 40  

# Initialize result storage
feature_dict = {}
all_results = []

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'BayesianRidge': BayesianRidge(),
    'HuberRegressor': HuberRegressor(max_iter=10000),
    'SGDRegressor': SGDRegressor(max_iter=10000),
    'RandomForestRegressor': RandomForestRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'KNeighborsRegressor': KNeighborsRegressor(),
    'SVR': SVR(),
    'MLPRegressor': MLPRegressor(max_iter=100000),
    'ExtraTreesRegressor': ExtraTreesRegressor(),  
}

# Loop over models
for model_name, model in models.items():
    num_features = []
    rmse_scores = []
    
    # Loop over increasing number of top features
    for i in range(1, feat_n + 1):
        # Select top i features
        features_subset = selected_features[:i]
        # Store feature subset
        num_features.append(i)
        # Store feature names in dictionary
        feature_dict[i] = features_subset
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X[features_subset], y_reg, test_size=0.2, random_state=42)
        # Fit the regression model
        model.fit(X_train, y_train)
        # Predict on test set
        y_pred = model.predict(X_test)
        # Calculate RMSE
        rmse = root_mean_squared_error(y_test, y_pred)
        # Store RMSE score
        rmse_scores.append(rmse)

    # Store results for this model
    model_results = pd.DataFrame({
        'Number of Features': num_features,
        'Model': model_name,
        'RMSE': rmse_scores
    })
    all_results.append(model_results)

# Combine all results into a single DataFrame
result_df = pd.concat(all_results, ignore_index=True)

#%% plot rmse change by features

# Plotting
plt.figure(figsize=(16, 9))
for model_name in result_df['Model'].unique():
    model_results = result_df[result_df['Model'] == model_name]
    #marker = marker_symbols[model_name]
    plt.plot(model_results['Number of Features'], model_results['RMSE'], label=model_name)

plt.xlabel('Number of Features')
plt.ylabel('RMSE')
plt.legend(loc='upper right')
#plt.title('RMSE vs. Number of Features for Different Regression Models')
plt.grid(True, linestyle='--')
plt.tight_layout()

#plot_filename = os.path.join(folder_path,'change in RMSE (MutInf).pdf')
#plt.savefig(plot_filename)

plt.show()

#%% subset of models

# drop from plot
#result_df = result_df[result_df['Model'] != 'SVR']
result_subset_df = result_df[result_df['Model'].isin(['LinearRegression', 
                                               'MLPRegressor', 
                                               'RandomForestRegressor', 
                                               'KNeighborsRegressor',
                                               'ExtraTreesRegressor'])]

# Plotting
plt.figure(figsize=(10, 10))
for model_name in result_subset_df['Model'].unique():
    model_results = result_subset_df[result_subset_df['Model'] == model_name]
    #marker = marker_symbols[model_name]
    plt.plot(model_results['Number of Features'], model_results['RMSE'], label=model_name, linewidth=2)

plt.xlabel('Number of Features')
plt.ylabel('RMSE')
plt.yticks(np.arange(0, max(result_subset_df['RMSE']), 2.5))  # Set y-axis ticks every 2.5
plt.legend(loc='upper right')
#plt.title('RMSE vs. Number of Features for Different Regression Models')
plt.grid(True, linestyle='--')
plt.tight_layout()

plot_filename = os.path.join(folder_path,'change in RMSE (MutInf).pdf')
plt.savefig(plot_filename)
plt.show()
