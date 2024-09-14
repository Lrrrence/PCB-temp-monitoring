# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:01:14 2024

@author: Lrrr
"""
#%% load features from file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

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

# columns to drop
dropped_columns = ['temp 1', 'temp 2', 'temp 3', 'temp 4', 'temp 5', 'ch_num', 'hotspot_num', 'time_stamp', 'Filename']
dropped_df = features_df[dropped_columns]

X = features_df.drop(columns=dropped_columns)
y = features_df['temp 1']

# number of top features to subset 
top_feat_num = 15

#%% scale features

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler to the features and transform them
scaled_features = scaler.fit_transform(X)
scaled_df = pd.DataFrame(scaled_features, columns=X.columns)

print("Original number of features:", X.shape[1])

#%% # dendrogram

corr = spearmanr(X).correlation
corr = np.nan_to_num(corr)
corr_condensed = squareform(1-corr, checks=False)

# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr_condensed)
# Ensure the distance matrix is symmetric
distance_matrix = (distance_matrix + distance_matrix.T) / 2
# Perform hierarchical clustering using Ward's linkage
dist_linkage = hierarchy.ward(squareform(distance_matrix))

# Create the dendrogram in the first figure
plt.figure(figsize=(12, 8))
dendro = hierarchy.dendrogram(
    dist_linkage, labels=X.columns.to_list(), leaf_rotation=90
)
plt.title('Dendrogram')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

#%%#################################

# manually pick a threshold (1) by visual inspection of the dendrogram to group our features into clusters and choose a feature from each cluster to keep
# lower number = more features included

threshold = 2

from collections import defaultdict
from sklearn.model_selection import cross_val_score

cluster_ids = hierarchy.fcluster(dist_linkage, threshold, criterion="distance")
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
selected_features_names = X.columns[selected_features]

X_sel = X[selected_features_names]

# Initialize the model
mdl = ExtraTreesRegressor(n_estimators=100, random_state=42)

# Evaluate the model using cross-validation with all features
scores_og = cross_val_score(mdl, X, y, cv=5, scoring='neg_root_mean_squared_error')
rmse_og = -np.mean(scores_og)

# Evaluate the model using cross-validation with selected features
scores_sel = cross_val_score(mdl, X_sel, y, cv=5, scoring='neg_root_mean_squared_error')
rmse_final = -np.mean(scores_sel)

print("Original number of features:", X.shape[1])
print("Remaining number of features:", X_sel.shape[1])
print(f"Cross-validated RMSE with all features: {rmse_og:.2f}")
print(f"Cross-validated RMSE with selected features: {rmse_final:.2f}")

# %%
