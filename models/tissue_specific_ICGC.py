#%%
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.svm import OneClassSVM
from collections import Counter
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from numpy import asarray
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import sys 
import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

# plot feature importance using built-in function
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot

import seaborn as sns

import importlib
from sklearn.inspection import permutation_importance

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import LeaveOneGroupOut
import numpy as np
from matplotlib import pyplot
sys.path.append('/Users/uw20204/Documents/scripts/CanDrivR/models')
from prepare_training_data import prepare_data
from train_evaluate import train_and_evaluate_model
from metric_results_table import get_results_table
from train_classifier import train_classifier
from DeepFFN import *
from ANN import *
import time
#%%
from tensorflow.keras.models import clone_model

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score

import pickle
from tensorflow.keras import layers, models
#%%
sys.path.append('/Users/uw20204/Documents/scripts/CanDrivR/optimisation')
from selected_features import features
from params import *
from typing import *
from best_donor_count import cancer_best_donor_count
#%%
len(features)
#%%
def merge_ICGC_donor_count(all_features_file, output_dir, features):
    df = []
    for chunk in pd.read_csv(all_features_file, sep = "\t", chunksize = 500000):
        df.append(chunk)

    df = pd.concat(df)

    features_updated = list(set(df.columns.tolist()) & set(features))
    print(len(features))
    cols_to_keep = ["chrom", "pos", "ref_allele", "alt_allele", "driver_stat", "study_id", "donor_count", "grouping"] + features_updated
    df = df[cols_to_keep]

    df.to_csv(f'{output_dir}/ICGC_with_donor_count_restricted_features.txt.gz', sep = "\t", index = None, compression='gzip')

    return df
#%%

# Run simple gradient boosting model on full balanced ICGC dataset
def run_baseline_model(df, study_id = None, positive_dataset_donor_count = 3, features = features):
    #try:
    if study_id:
        ICGC = df[df["study_id"] == study_id]
    else:
        ICGC = df

    # Filter so that positive data is variants present in > "x" donors in the ICGC dataset
    ICGC = pd.concat([ICGC[ICGC["donor_count"] == 1].sample(len(ICGC[ICGC["donor_count"] > positive_dataset_donor_count])), ICGC[ICGC["donor_count"] > positive_dataset_donor_count]], axis = 0)

    #ICGC = ICGC.drop(ICGC.columns[ICGC.isna().any()].tolist(), axis = 1)
    
    ICGC = ICGC.reset_index(drop = True)

    # Train only with features found in the top features list and in the ICGC dataset
    features_updated = list(set(ICGC.columns.tolist()) & set(features))

    # Train and evaluate classifier
    actual_predicted_targets, feature_importance, final_model, results = train_classifier(ICGC, features = features_updated, classifier = XGBClassifier(random_state=42))
    XGB_results_metrics = get_results_table(results[0], model_name = "XGB")
    print(study_id)
    print(len(ICGC))
    print(XGB_results_metrics)
    return XGB_results_metrics, ICGC, features_updated, results[1]
    #except:
     #   print("dataset too small")
from keras import models, layers, regularizers

def build_model(input_shape, l2_reg=0.001):
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg), input_shape=(input_shape,)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
def build_general_ANN(df, features, donor_count_positives = 2):
    #Filter above dataset again
    ICGC_balanced = pd.concat([df[df["donor_count"] == 1].sample(len(df[df["donor_count"] > donor_count_positives])), df[df["donor_count"] > donor_count_positives]], axis = 0).reset_index(drop = True)

    ICGC_balanced = ICGC_balanced.fillna(0)

    features = list(set(ICGC_balanced.columns.tolist()) & set(features))

    X_train_val, y_train_val, X_test, y_test, X, y= prepare_data(ICGC_balanced, features)

    # Standardise features
    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)

    input_shape = X_train_val.shape[1]
    model_general = build_model(input_shape)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, mode='min')

    history = model_general.fit(X_train_val, y_train_val, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])

    test_loss, test_acc = model_general.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Predictions
    predictions = model_general.predict(X_test)
    binary_predictions = np.round(predictions).astype(int)

    # Calculating metrics
    precision = precision_score(y_test, binary_predictions)
    recall = recall_score(y_test, binary_predictions)
    f1 = f1_score(y_test, binary_predictions)
    auc_roc = roc_auc_score(y_test, predictions)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")

    return model_general, features, f1, precision, recall

def split_dataset_cancer(df):
    cancer_types = df['study_id'].unique()
    cancer_datasets = {cancer: df[df['study_id'] == cancer] for cancer in cancer_types}
    # Save the cancer_datasets dictionary
    with open('/Volumes/Samsung_T5/data/cancer_datasets_2.pkl', 'wb') as f:
        pickle.dump(cancer_datasets, f)

def run_fine_tuning_cancer_specific(model_general, features):
    with open('/Volumes/Samsung_T5/data/cancer_datasets_2.pkl', 'rb') as f:
        cancer_datasets = pickle.load(f)

    # Initialise an empty DataFrame to store metrics
    metrics_df = pd.DataFrame()

    for cancer_type, data in cancer_datasets.items():
        try:
            donor_count_positives = cancer_best_donor_count[cancer_type] # Gets the optimised donor count for positives for each cancer type from saved dictionary
            data = pd.concat([data[data["donor_count"] == 1].sample(len(data[data["donor_count"] > donor_count_positives])), data[data["donor_count"] > donor_count_positives]], axis=0).reset_index(drop = True)
            data = data.fillna(0)
            dataset_size = len(data)

            if dataset_size > 100:
                groups = data['grouping']

                # Filter and prepare cancer-specific dataset
                X_cancer = data[features]
                y_cancer = data['driver_stat']

                # Lists to collect metrics for each fold
                accuracies, balanced_accs, precisions, recalls, f1_scores, auc_rocs = [], [], [], [], [], []

                logo = LeaveOneGroupOut()
                
                # Callbacks
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)

                for train_idx, test_idx in logo.split(X_cancer, y_cancer, groups):
                    X_train, X_test = X_cancer.iloc[train_idx], X_cancer.iloc[test_idx]
                    y_train, y_test = y_cancer.iloc[train_idx], y_cancer.iloc[test_idx]

                    # Standardise features
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                    model_specific = clone_model(model_general)
                    model_specific.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    
                    # Just early stopping
                    early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
                    
                    # Training without a separate validation set since we're using LOGO
                    model_specific.fit(X_train, y_train, epochs=5, callbacks=[early_stopping])

                    predictions = model_specific.predict(X_test)
                    binary_predictions = np.round(predictions).astype(int)

                    # Collect metrics for this fold
                    accuracies.append(accuracy_score(y_test, binary_predictions))
                    balanced_accs.append(balanced_accuracy_score(y_test, binary_predictions))
                    precisions.append(precision_score(y_test, binary_predictions, zero_division=0))
                    recalls.append(recall_score(y_test, binary_predictions))
                    f1_scores.append(f1_score(y_test, binary_predictions))
                    auc_rocs.append(roc_auc_score(y_test, predictions))

                # Calculate mean and std of metrics across all folds
                metrics_summary = {
                    'Cancer Type': cancer_type,
                    'Positive Donor Count': donor_count_positives,
                    'Dataset Size': dataset_size,
                    'Mean Accuracy': np.mean(accuracies),
                    'Std Accuracy': np.std(accuracies),
                    'Mean Balanced Accuracy': np.mean(balanced_accs),
                    'Std Balanced Accuracy': np.std(balanced_accs),
                    'Mean Precision': np.mean(precisions),
                        'Std Precision': np.std(precisions),
                        'Mean Recall': np.mean(recalls),
                        'Std Recall': np.std(recalls),
                        'Mean F1 Score': np.mean(f1_scores),
                        'Std F1 Score': np.std(f1_scores),
                        'Mean AUC-ROC': np.mean(auc_rocs),
                        'Std AUC-ROC': np.std(auc_rocs),
                    }

                # Using pd.concat to add the new row to metrics_df
                metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics_summary])], ignore_index=True)

            print(metrics_df)
        except Exception as e:
            print(f"Failed to process {cancer_type}: {e}")

    return metrics_df

def get_plot(cancer, features, positive, negative):
    # Ensure 'features' is a list even if a single feature string is passed
    if not isinstance(features, list):
        features = [features]
    
    # Initialize a DataFrame to hold the combined data
    combined_df_list = []

    # Process each feature
    for feature_to_plot in features:
        # Copy dataframes to avoid changing original data
        pos_copy = positive[[feature_to_plot]].copy()
        neg_copy = negative[[feature_to_plot]].copy()

        # Add a new column to each DataFrame to denote the case (positive or negative) and the feature
        pos_copy['Case'] = 'Positive'
        neg_copy['Case'] = 'Negative'
        pos_copy['Feature'] = feature_to_plot
        neg_copy['Feature'] = feature_to_plot

        # Rename the value column uniformly to allow seaborn to plot all features together
        pos_copy.rename(columns={feature_to_plot: 'Value'}, inplace=True)
        neg_copy.rename(columns={feature_to_plot: 'Value'}, inplace=True)

        # Concatenate the positive and negative DataFrames and then append to the list
        combined_df_list.append(pd.concat([pos_copy, neg_copy]))
    
    # Concatenate all dataframes in the list to form the final combined DataFrame
    combined_df = pd.concat(combined_df_list)

    # Plot using seaborn
    plt.figure(figsize=(12, 8))  # Adjusted for potentially better visualization
    box_plot = sns.boxplot(x='Feature', y='Value', hue='Case', data=combined_df, palette="Set2")

    # Enhance the visual appearance
    box_plot.set_title(f'Comparison of Features by Case for {cancer}', fontsize=16, fontweight='bold')
    box_plot.set_xlabel('Feature', fontsize=14, fontweight='bold')
    box_plot.set_ylabel('Value', fontsize=14, fontweight='bold')
    box_plot.tick_params(axis='x', labelrotation=45)  # Rotate x labels for better readability
    box_plot.tick_params(axis='y', labelsize=12)

    sns.despine(offset=11, trim=True)
    plt.legend(title='Case', title_fontsize='13', fontsize='12')

    # Show the plot
    plt.show()
#%%
# Specify location of all features file
#donor_count_file = "/Volumes/Samsung_T5/data/final_ICGC_tissue_specific_data/ICGC_with_studyID.bed" # File that contains donor count information
all_features_file = "/Volumes/Samsung_T5/data/final_ICGC_tissue_specific_data/all_features.bed.gz" # Feature file for ICGC
output_dir = "/Volumes/Samsung_T5/data/final_ICGC_tissue_specific_data" # Where to save feature file with donor count merge
df_tissue_spec = merge_ICGC_donor_count(all_features_file, output_dir, features)

# Just removing these columns for now until these features stop running
df_tissue_spec.drop(list(df_tissue_spec.filter(regex='_w|_x|_y|_z')), axis=1, inplace=True)
df_tissue_spec
#%%
df
#%%
# Test different values for the donor count poisitive sample filtering
# Save the results to select best count
best_donor_count = []
for i in range(1, 10):
    # Run a standard XGB baseline model
    XGB_results_metrics, ICGC, features = run_baseline_model(df, positive_dataset_donor_count = i)
    best_donor_count.append([i, len(ICGC), XGB_results_metrics])
    print(best_donor_count)

#%%
metrics_to_merge = pd.concat([i[2] for i in best_donor_count]).reset_index(drop = True)
donor_count_to_merge = pd.DataFrame(data = {"donor_count": [i[0] for i in best_donor_count]})
df_size_to_merge = pd.DataFrame(data = {"size": [i[1] for i in best_donor_count]})
results = pd.concat([donor_count_to_merge, df_size_to_merge, metrics_to_merge], axis = 1)
results
#%%
# Plot donor count vx accuracy/ size
# Shows that best performance seen with donor count > 2
sys.path.append('/Users/uw20204/Documents/scripts/CanDrivR/plots')
from donor_count_vs_accuracy import plot_curve_with_dual_y_axis
plot_curve_with_dual_y_axis(results, "/Users/uw20204/Documents/figures/")
#%%
df_tissue_spec1 = df_tissue_spec
import random

# Define a list of groups and ensure each is used at least once with duplicates to reach 22 assignments
groups = list(range(1, 7)) * 4  # Duplicating group range to cover 22 assignments
random.shuffle(groups)  # Shuffle to randomize distribution

# List of chromosomes
chromosomes = [
    'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9',
    'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
    'chr18', 'chr19', 'chr20', 'chr21', 'chr22'
]

# Assign first 22 shuffled groups to chromosomes to ensure even distribution
chromosome_to_group = dict(zip(chromosomes, groups[:22]))

# Now apply the mapping to your DataFrame
df_tissue_spec1['grouping'] = df_tissue_spec1['chrom'].map(chromosome_to_group)

# Handle any NaN values, though there shouldn't be any if all chromosomes are covered
df_tissue_spec1['grouping'] = df_tissue_spec1['grouping'].fillna(0)

# Optionally, remove columns with all NaN values
df_tissue_spec1 = df_tissue_spec1.dropna(axis=1, how='all')
#%%
df_tissue_spec1.to_csv("/Volumes/Samsung_T5/data/final_ICGC_tissue_specific_data/tissue_specific_dataset_final.txt.gz", sep = "\t", index = None, compression='gzip')
#%%
######################
df = pd.read_csv("/Volumes/Samsung_T5/data/final_ICGC_tissue_specific_data/tissue_specific_dataset_final.txt.gz", sep = "\t", compression='gzip')
#%%
# Get cross validation and test results for optimised donor count
XGB_results_metrics, ICGC, features, test_results = run_baseline_model(df, positive_dataset_donor_count = 2)
#%%
df_tissue_spec1
#%%

# Split dataset into cancer-specific and save as separate .pkl files
split_dataset_cancer(df_tissue_spec1)
#%%
# Load the cancer datasets file
with open('/Volumes/Samsung_T5/data/cancer_datasets_2.pkl', 'rb') as f:
    cancer_datasets = pickle.load(f)
#%%
optimised_cancer_results = []
cancer_dict_donor_counts = Counter() # Initialise an empty dictionary for storing optimised donor counts for each cancer
# Run a general cancer-specific model using XGB
# Optimise the donor count threshold for each cancer type then run XGB
for cancer_dataset in list(cancer_datasets.keys()):
    cancer_dict_donor_counts[cancer_dataset] = 0
    tracker = 0
    for dc in range(1, 10):
        try:
            XGB_results_metrics, ICGC, features, _ = run_baseline_model(df_tissue_spec1, positive_dataset_donor_count = dc, study_id = cancer_dataset)
            f1 = XGB_results_metrics["f1"].str.split(" ± ", expand=True)[0].astype("float").item()
            if f1 > tracker:
                cancer_dict_donor_counts[cancer_dataset] = dc # Update the donor count if the f1 score is higher
                tracker = f1 # Update f1 tracker
            optimised_cancer_results.append([cancer_dataset, dc, len(ICGC), XGB_results_metrics]) # Create a list for plotting
        except:
            print("dataset too small")
#%%
optimised_cancer_results[0]
#%%
cancers = [optimised_cancer_results[i][0] for i in range(len(optimised_cancer_results))]
donor_counts = [optimised_cancer_results[i][1] for i in range(len(optimised_cancer_results))]
dataset_size = [optimised_cancer_results[i][2] for i in range(len(optimised_cancer_results))]
F1 = [optimised_cancer_results[i][3]["f1"].item().split(" ± ")[0] for i in range(len(optimised_cancer_results))]
results_cancer_spec = pd.DataFrame(data = {"cancer": cancers, "donor_count": donor_counts, "size": dataset_size, "f1": F1})
# Plot optimised donor count for each cancer dataset
plot_grid_of_curves([results_cancer_spec[results_cancer_spec["cancer"] == cancer] for cancer in results_cancer_spec["cancer"].unique()], "/Users/uw20204/Documents/figures/", results_cancer_spec["cancer"].unique())
#%%
F1 = [optimised_cancer_results[i][3]["f1"].item() for i in range(len(optimised_cancer_results))]
F1
#%%
res = [optimised_cancer_results[i][3] for i in range(len(optimised_cancer_results))]

res = pd.concat([results_cancer_spec.reset_index(drop = True).drop("f1", axis = 1), pd.concat(res).reset_index(drop = True)], axis = 1)
#%%
optimised_res = []
for cancer_name in list(cancer_dict_donor_counts.keys()):
    print(cancer_name)
    opt_donor_count = cancer_dict_donor_counts[cancer_name]
    result = res[(res["cancer"] == cancer_name) & (res["donor_count"] == opt_donor_count)]
    optimised_res.append(result)
optimised_metrics = pd.concat(optimised_res).reset_index(drop = True)
optimised_metrics
#%%
# Only keep cancers where dataset size is > 100
optimised_metrics = optimised_metrics[optimised_metrics["size"] > 100]
#%%

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_barplot(data, outputDir, fileName):
    """
    Plot barplot with standard deviations using a custom color palette.

    Parameters:
        data (pd.DataFrame): DataFrame containing cancer type, donor count, and evaluation metrics.
        outputDir (str): Directory to save the plot.
        fileName (str): Name of the file.
    """
    # Define metrics for plotting
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    # Define a custom color palette
    custom_palette = sns.color_palette("Dark2", len(metrics))

    # Reformat the data
    df = []
    for i, row in data.iterrows():
        for metric in metrics:
            value, std_dev = row[metric].split(' ± ')
            df.append({
                'Metric': metric.capitalize(),
                'Value': float(value),
                'Std Dev': float(std_dev),
                'Cancer Type': row['cancer']
            })

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(df)

    # Plotting
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 5))
    ax = sns.barplot(x="Cancer Type", y="Value", hue="Metric", data=df, palette=custom_palette, ci=None, capsize=.2, edgecolor = "black")
    
    # Iterate over the bars to manually add errorbars
    for i, bar in enumerate(ax.patches):
        # Calculate the number of metrics * number of cancer types to find the bar index correctly
        num_metrics = len(metrics)
        num_cancer_types = len(data['cancer'].unique())
        index = i % num_metrics + (i // (num_metrics * num_cancer_types)) * num_metrics
        # Set error bar for each bar based on the Std Dev
        plt.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                     yerr=df.loc[index, 'Std Dev'], fmt='none', capsize=2, color='black')

    ax.set_title("Evaluation Metrics for Different Cancer Types", fontsize=16)
    ax.set_xlabel("Cancer Type", fontsize=14)
    ax.set_ylabel("Metric Value", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(title='Metrics', fontsize=12, title_fontsize=12)

    # Save the figure with adjustments
    plt.tight_layout()
    plt.savefig(f"{outputDir}/{fileName}.png", format="png", dpi=300, bbox_inches="tight")
    plt.show()

plot_barplot(optimised_metrics.sort_values("f1", ascending = False)[:20], "/Users/uw20204/Documents/figures/", "cancer_specific_xgb_metrics")
#%%
optimised_metrics
# Assuming 'df' is your DataFrame
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
# Create a new DataFrame for heatmap data
heatmap_data = pd.DataFrame()

for metric in metrics:
    # Split the metric and its standard deviation, and only keep the metric values
    heatmap_data[metric] = optimised_metrics[metric].apply(lambda x: float(x.split(' ± ')[0]))

# Set the index to cancer types for better visualization
heatmap_data.set_index(optimised_metrics['cancer'], inplace=True)
plt.figure(figsize=(10, 12))  # Adjust size as necessary
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Evaluation Metrics Heatmap by Cancer Type')
plt.ylabel('Cancer Type')
plt.xlabel('Metrics')
plt.xticks(rotation=45)  # Rotate metric labels for better readability
plt.tight_layout()  # Adjust layout to make sure nothing is clipped
plt.savefig('/Users/uw20204/Documents/figures/metrics_heatmap.png', dpi=300)  # Save the figure
plt.show()
#%%
# Train a general neural network model on the pooled cancer dataset
model_general, features, f1, precision, recall = build_general_ANN(df, features, donor_count_positives = 3)
#%%

#%%
# Fine-tune the general model on cancer-specific datasets
# The donor count is specified in a saved dictionary after optimising for the model
metrics_df = run_fine_tuning_cancer_specific(model_general, features)
#%%
# Sort the metrics based on the top models
metrics_df.sort_values("Mean F1 Score", ascending = False)
#%%
len(features)
#%%

#%%

def plot_features(cancer_datasets, cancer):
    dataset = cancer_datasets[cancer]
    donor_count_positives = cancer_best_donor_count[cancer]
    positive = dataset[dataset["donor_count"] > donor_count_positives][features]
    negative = dataset[dataset["donor_count"] == 1][features]

    print(positive)
    print(negative)

    results = []

    for feature in features:
        # Perform the t-test:
        t_stat, p_value = stats.ttest_ind(positive[feature], negative[feature])
        results.append([feature, t_stat, p_value])

    results = pd.DataFrame(results).sort_values(2).reset_index(drop = True)

    # Example usage
    #features_to_plot = ["1_Direction", "2_Direction", "3_Direction", "4_Direction"] # Adjust this line as per your dataframe
    #features_to_plot =["7_ProT", "8_ProT", "9_ProT", "10_ProT", "11_ProT", "12_ProT", "13_ProT", "14_ProT", "15_ProT"]
    #features_to_plot =["20GCContent", "40GCContent", "60GCContent", "80GCContent", "100GCContent", "200GCContent", "500GCContent", "1000GCContent", "2000GCContent"]
#    features_to_plot =["mutant_AA_Hydropathy_scale_based_on_self-information_values_in_the_two-state_model_(50%_accessibility)_(Naderi-Manesh_et_al.,_2001)", "WT_AA_Optical_rotation_(Fasman,_1976)", "WT_AA_The_Kerr-constant_increments_(Khanarian-Moore,_1980)", "WT_AA_Relative_mutability_(Jones_et_al.,_1992)", "WT_AA_Activation_Gibbs_energy_of_unfolding,_pH9.0_(Yutani_et_al.,_1987)", "WT_AA_Transfer_free_energy_from_chx_to_oct_(Radzicka-Wolfenden,_1988)", "mutant_AA_Heat_capacity_(Hutchens,_1970)", "WT_AA_alpha-NH_chemical_shifts_(Bundi-Wuthrich,_1979)"]
    features_to_plot = [str(i) + "_EP" for i in range(3, 19)]
    features_to_plot = [str(i) + "_ProT" for i in range(3, 19)]
    features_to_plot = [str(i) + "_MGW" for i in range(3, 19)]
    features_to_plot = results.iloc[:20, 0].tolist()
    get_plot(cancer, features_to_plot, positive, negative)

    return results

results = plot_features(cancer_datasets, "READ")
#%%


#%%
p_value
t_stat
#%%
# Interpret the results:
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis; there is a significant difference between the petal lengths of Iris setosa and Iris versicolor.")
else:
    print("Fail to reject the null hypothesis; there is no significant difference between the petal lengths of Iris setosa and Iris versicolor.")

#%%
results = []
for i in range(10, 0, -1):
    # First run XGBoost model on the full dataset
    XGB_results_metrics, ICGC, features = run_model(df, study_id = None, positive_dataset_donor_count = i, features = features)     
    print(XGB_results_metrics)
    results.append([i, XGB_results_metrics, ICGC, features])
#%%
XGB_results_metrics
#%%
# Then run XGBoost on each dataset individually for a cancer specific model
cancer_types = df['study_id'].unique()

#%%
metrics_df

#%%

#%%


#%%
with open('/Volumes/Samsung_T5/data/cancer_datasets.pkl', 'rb') as f:
    cancer_datasets = pickle.load(f)
#%%
# Pool the cancer datasets for the general model
df = pd.concat(cancer_datasets)
#%%
df = df.dropna().reset_index(drop = True)
df
#%%
#%%
# Fine tune the neural network model on cancer-specific datasets, using already optimised donor counts for each cancer
metrics_df = run_fine_tuning_cancer_specific(model_general, features)
#%%

#%%
df = pd.concat(loaded_cancer_datasets)
#%%
df = df.dropna()
#%%
results = []
for i in range(1, 20):
    model_general, features, f1, precision, recall = build_general_ANN(df, features, donor_count_positives = i)
    results.append(model_general, features, f1, precision, recall)
#%%
#split_dataset_cancer(p;;[''''']\df)
#%%

#%%

#%%
mean_abs_diff[:30]
#%%


#%%



#%%



#%%
len(features)
#%%
positive.dropna()
#%%
run_fine_tuning_cancer_specific(model_general)
#%%
if __name__ == "__main__":
#%%
donor_count_file = "/Volumes/Samsung_T5/data/ICGC_with_studyID.bed"
all_features_file = "/Volumes/Samsung_T5/data/all_features.bed"
output_dir = "/Volumes/Samsung_T5/data/"
#%%
#merge_ICGC_donor_count(donor_count_file, all_features_file, output_dir)
[run_model(i, features) for i in donor_counts_study_id[4].unique()]
XGB_results_metrics, ICGC_balanced, features = run_model(df, donor_counts_study_id)

#%%

df = []
for chunk in pd.read_csv(f'{output_dir}ICGC_donor_count.txt.gz', compression='gzip', sep = "\t", chunksize = 500000):
    df.append(chunk)

df = pd.concat(df).reset_index(drop = True)
#%%
df_with_kernel = df
df = df[df.columns.drop(list(df.filter(regex='kernel')))]
#%%
df[df["donor_count"] > 1]
#%%
ICGC_balanced = pd.concat([df[df["donor_count"] == 1].sample(len(df[df["donor_count"] > 2])), df[df["donor_count"] > 2]], axis = 0)
#%%
df_with_kernel[df_with_kernel["donor_count"] > 1]
#%%

model_general = build_general_ANN(df, features, donor_count_positives = 2)
#%%

with open('/Volumes/Samsung_T5/data/cancer_datasets.pkl', 'rb') as f:
    loaded_cancer_datasets = pickle.load(f)
#%%
sum([len(loaded_cancer_datasets[i]) for i in loaded_cancer_datasets.keys()])
#%%

#%%
# Display the DataFrame
metrics_df.sort_values("Mean F1 Score", ascending = False)
#%%
metrics_df
# %%
[metrics_df[metrics_df["Cancer Type"] == i].sort_values("F1 Score", ascending = False) for i in metrics_df["Cancer Type"].unique()]
# %%
metrics_df[metrics_df["F1 Score"] > 0.8]
# %%
metrics_df = metrics_df.sort_values("Mean F1 Score", ascending = False)
#%%
metrics_df[metrics_df["Mean F1 Score"] > 0.78]
# %%
metrics_df.groupby(["Cancer Type"])["Mean F1 Score"]
# %%
metrics_df.groupby('Cancer Type')['Mean F1 Score'].max()

# %%
# Step 1: Compute the max score within each group
metrics_df['Max F1'] = metrics_df.groupby('Cancer Type')['Mean F1 Score'].transform('max')

# Step 2: Filter rows where the Score equals the Max_Score within that group
to_map = metrics_df[metrics_df['Mean F1 Score'] == metrics_df['Max F1']][["Cancer Type", "Positive Donor Count"]].reset_index(drop = True)

#%%
metrics_df[metrics_df['Mean F1 Score'] == metrics_df['Max F1']]


filtered_columns = [col for col in max_f1.columns if 'Std' not in col]

max_f1[max_f1["Max F1"] > 0.7][filtered_columns]
# %%
# Create a dictionary that maps the cancer to the corresponding donor count best F1 score
for i in range(0, len(to_map)):
    dictionary[to_map["Cancer Type"][i]] = to_map["Positive Donor Count"][i]

# %%
dictionary
# %%
dictionary
from Bio import SeqIO
import os
import pandas as pd
from strkernel.mismatch_kernel import MismatchKernel
from strkernel.mismatch_kernel import preprocess
from Bio import SeqIO
from Bio.Seq import Seq
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report # classfication summary
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import sys
from itertools import islice
import glob
import multiprocessing
from config import *
from functools import reduce

def process_kmer(windowSize, kmerSize):
    # kmer of window size 5 and kmer size 3
    # get sequences given a set of variants and specified sequence lengths
    variantsSequences = getSequences(variants, windowSize, kmerSize)
    spectrumFeatureList = [getSpectrumFeatures(list(variantsSequences.loc[i, :])[0], list(variantsSequences.loc[i, :])[1], kmerSize) for i in range(0, len(variantsSequences))]
    spectrumList = [list(spectrumFeatureList[x].loc[0, :]) + list(spectrumFeatureList[x].loc[1, :]) for x in range(0, len(spectrumFeatureList))]
    spectrumdf = pd.DataFrame(spectrumList)
    spectrumdf = pd.concat([variants, spectrumdf], axis=1)
    spectrumdf["chrom"] = spectrumdf["chrom"]
    spectrumdf["pos"] = spectrumdf["pos"].astype(int)
    spectrumdf = spectrumdf.rename(columns={0: str(windowSize*2) + "_" + str(kmerSize) + "_w", 1: str(windowSize*2) + "_" + str(kmerSize) + "_x", 2: str(windowSize*2) + "_" + str(kmerSize) + "_y", 3: str(windowSize*2) + "_" + str(kmerSize) + "_z"})
    spectrumdf = spectrumdf.drop("pos2", axis = 1)

    # Save each result to CSV file
    output_file_path = os.path.join(outputDir, f"{windowSize*2}_{kmerSize}_kernel.txt")

    # Check if the output file already exists
    if os.path.exists(output_file_path):
        # Append to the existing file
        spectrumdf.to_csv(output_file_path, sep="\t", index=False, mode='a', header=False)
    else:
        # Create a new file
        spectrumdf.to_csv(output_file_path, sep="\t", index=False)

# unput is the variant dataset and window size either side of the variants (e.g. w of 100 = 200 bp in total)
def getSequences(dataset, window_size, k):
    kmerDf = pd.DataFrame()
    similarityArray = []

    def getSeqFun(i):

        # generates wild type sequence
        # by refering to the reference sequence, it gets the sequences flanking 100bp either side of the variant position
        wildType = str(record_dict[dataset.loc[i, "chrom"]].seq[int(dataset.loc[i, "pos"]-1-window_size):int(dataset.loc[i, "pos"]-1+window_size)]).upper()

        # mutant sequence
        # repeats the same as above but replaces WT variant with the mutant variant
        mutant = str(record_dict[dataset.loc[i, "chrom"]].seq[int(dataset.loc[i, "pos"]-1-window_size):int(dataset.loc[i, "pos"]-1)]) + dataset.loc[i, "alt_allele"] + str(record_dict[dataset.loc[i, "chrom"]].seq[int(dataset.loc[i, "pos"]):int(dataset.loc[i, "pos"]-1+window_size)]).upper()

        kmerDf.loc[i, "wildType"] = wildType.upper()
        kmerDf.loc[i, "mutant"] = mutant.upper()

    # Carries out function for each variant in the dataset
    [getSeqFun(i) for i in range(0, len(variants))]
    return kmerDf


# generator function
# generates all combinations of k-mers for the sequences
def over_slice(test_str, K):
    itr = iter(test_str)
    res = tuple(islice(itr, K))
    if len(res) == K:
        yield res   
    for ele in itr:
        res = res[1:] + (ele,)
        yield res

def getSpectrumFeatures(seq1, seq2, k): 
    # initializing string
    test_str = seq1 + seq2

    # initializing K
    K = k
    
    # calling generator function
    res = ["".join(ele) for ele in over_slice(test_str, K)]
    dfMappingFunction = pd.DataFrame(columns = ["seq"] + res)
    dfMappingFunction.loc[0, "seq"] = seq1
    dfMappingFunction.loc[1, "seq"] = seq2

    # generate mapping function
    def getMappingFunction(res, seq):
        if res in seq: # if the k-mer is present in the sequence, count the number of occurences in the sequence
            dfMappingFunction.loc[dfMappingFunction["seq"] == seq, res] = dfMappingFunction.loc[dfMappingFunction["seq"] == seq, "seq"].reset_index(drop=True)[0].count(res) 
        else: # if its not in the sequence, fill with a 0
            dfMappingFunction.loc[dfMappingFunction["seq"] == seq, res] = 0

    for seq in [seq1, seq2]:
        [getMappingFunction(x, seq) for x in res]

    # generate p-spectra
    pSpectrumKernel = pd.DataFrame(columns=[seq1, seq2])
    pSpectrumKernel.insert(0, "seq", [seq1, seq2])

    # to derive p-spectra, take product of each column and sum together
    products_WT_mut = [np.prod(dfMappingFunction.iloc[:, i]) for i in range(1, len(dfMappingFunction.columns))]
    sum_WT_mut  = np.sum(products_WT_mut)
    pSpectrumKernel.iloc[0, 2] = sum_WT_mut
    pSpectrumKernel.iloc[1, 1] = sum_WT_mut
    # to derive the diagonal of the kernel matrix, we take the sum of the squares for the corresponding row
    squares_WT = [np.square(dfMappingFunction.iloc[0, i]) for i in range(1, len(dfMappingFunction.columns))]
    squares_mutant = [np.square(dfMappingFunction.iloc[1, i]) for i in range(1, len(dfMappingFunction.columns))]
    pSpectrumKernel.iloc[0, 1] = np.sum(squares_WT)
    pSpectrumKernel.iloc[1, 2] = np.sum(squares_mutant)
    return pSpectrumKernel.drop("seq", axis = 1)
    
def getFinalSpectrumDf(windowSize, kmerSize):
    # kmer of window size 5 and kmer size 3
    # get sequences given a set of variants and specified sequence lengths
    variantsSequences = getSequences(variants, windowSize, kmerSize)
    spectrumFeatureList = [getSpectrumFeatures(list(variantsSequences.loc[i, :])[0], list(variantsSequences.loc[i, :])[1], kmerSize) for i in range(0, len(variantsSequences))]
    spectrumList = [list(spectrumFeatureList[x].loc[0, :]) + list(spectrumFeatureList[x].loc[1, :]) for x in range(0, len(spectrumFeatureList))]
    spectrumdf = pd.DataFrame(spectrumList)
    spectrumdf = pd.concat([variants, spectrumdf], axis = 1)
    spectrumdf["chrom"] = spectrumdf["chrom"].str.replace("chr", "").astype(str)
    spectrumdf["pos"] = spectrumdf["pos"].astype(int)
    spectrumdf = spectrumdf.rename(columns = {0: str(windowSize*2) + "_" + str(kmerSize) + "_w", 1: str(windowSize*2) + "_" + str(kmerSize) + "_x", 2: str(windowSize*2) + "_" + str(kmerSize) + "_y", 3: str(windowSize*2) + "_" + str(kmerSize) + "_z"})
    spectrumdf = spectrumdf.drop("pos2", axis = 1)    


if __name__ == "__main__":
    variantDir = sys.argv[1]
    variants = variantDir + sys.argv[2]
    outputDir = sys.argv[3]

    # Reads in the human GRCh38 genome in fasta format
    record_dict = SeqIO.to_dict(SeqIO.parse(hg38_seq, "fasta"))

    chunk_size = 100000
    for chunk in pd.read_csv(variants, sep = "\t", names = ['chrom', 'pos', 'pos2', 'ref_allele', 'alt_allele'], chunksize = chunk_size):
        # Reading in the variant file
        variants = chunk
        # Removes sex chromosomes
        variants = variants[(variants['chrom'] != "chrX") & (variants['chrom'] != "chrY")]
        variants = variants.reset_index(drop = True)

        # List of window sizes and kmer sizes for which you want to compute the kernels
        window_sizes = [1, 2, 3, 4, 5]
        kmer_sizes_list = [
            list(range(1, 2)),
            list(range(1, 4, 1)),
            list(range(1, 6, 1)),
            list(range(1, 8, 1)),
            list(range(1, 10, 1))
        ]

        # Create a pool of worker processes with the number of cores you want to use
        num_cores = multiprocessing.cpu_count()  # Use all available cores
        pool = multiprocessing.Pool(processes=num_cores)

        # Use the pool of processes to parallelize the computation
        for window_size, kmer_sizes in zip(window_sizes, kmer_sizes_list):
            pool.starmap(process_kmer, [(window_size, kmer_size) for kmer_size in kmer_sizes])

        # Close the pool of processes
        pool.close()
        pool.join()

##############

dictionary
from Bio import SeqIO
import os
import pandas as pd
from strkernel.mismatch_kernel import MismatchKernel
from strkernel.mismatch_kernel import preprocess
from Bio import SeqIO
from Bio.Seq import Seq
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report # classfication summary
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import sys
from itertools import islice
import glob
import multiprocessing
from config import *
from functools import reduce

def process_kmer(windowSize, kmerSize):
    # kmer of window size 5 and kmer size 3
    # get sequences given a set of variants and specified sequence lengths
    variantsSequences = getSequences(variants, windowSize, kmerSize)
    spectrumFeatureList = [getSpectrumFeatures(list(variantsSequences.loc[i, :])[0], list(variantsSequences.loc[i, :])[1], kmerSize) for i in range(0, len(variantsSequences))]
    spectrumList = [list(spectrumFeatureList[x].loc[0, :]) + list(spectrumFeatureList[x].loc[1, :]) for x in range(0, len(spectrumFeatureList))]
    spectrumdf = pd.DataFrame(spectrumList)
    spectrumdf = pd.concat([variants, spectrumdf], axis=1)
    spectrumdf["chrom"] = spectrumdf["chrom"]
    spectrumdf["pos"] = spectrumdf["pos"].astype(int)
    spectrumdf = spectrumdf.rename(columns={0: str(windowSize*2) + "_" + str(kmerSize) + "_w", 1: str(windowSize*2) + "_" + str(kmerSize) + "_x", 2: str(windowSize*2) + "_" + str(kmerSize) + "_y", 3: str(windowSize*2) + "_" + str(kmerSize) + "_z"})
    spectrumdf = spectrumdf.drop("pos2", axis = 1)

    # Save each result to CSV file
    output_file_path = os.path.join(outputDir, f"{windowSize*2}_{kmerSize}_kernel.txt")

    # Check if the output file already exists
    if os.path.exists(output_file_path):
        # Append to the existing file
        spectrumdf.to_csv(output_file_path, sep="\t", index=False, mode='a', header=False)
    else:
        # Create a new file
        spectrumdf.to_csv(output_file_path, sep="\t", index=False)

# unput is the variant dataset and window size either side of the variants (e.g. w of 100 = 200 bp in total)
def getSequences(dataset, window_size, k):
    kmerDf = pd.DataFrame()
    similarityArray = []

    def getSeqFun(i):

        # generates wild type sequence
        # by refering to the reference sequence, it gets the sequences flanking 100bp either side of the variant position
        wildType = str(record_dict[dataset.loc[i, "chrom"]].seq[int(dataset.loc[i, "pos"]-1-window_size):int(dataset.loc[i, "pos"]-1+window_size)]).upper()

        # mutant sequence
        # repeats the same as above but replaces WT variant with the mutant variant
        mutant = str(record_dict[dataset.loc[i, "chrom"]].seq[int(dataset.loc[i, "pos"]-1-window_size):int(dataset.loc[i, "pos"]-1)]) + dataset.loc[i, "alt_allele"] + str(record_dict[dataset.loc[i, "chrom"]].seq[int(dataset.loc[i, "pos"]):int(dataset.loc[i, "pos"]-1+window_size)]).upper()

        kmerDf.loc[i, "wildType"] = wildType.upper()
        kmerDf.loc[i, "mutant"] = mutant.upper()

    # Carries out function for each variant in the dataset
    [getSeqFun(i) for i in range(0, len(variants))]
    return kmerDf


# generator function
# generates all combinations of k-mers for the sequences
def over_slice(test_str, K):
    itr = iter(test_str)
    res = tuple(islice(itr, K))
    if len(res) == K:
        yield res   
    for ele in itr:
        res = res[1:] + (ele,)
        yield res

def getSpectrumFeatures(seq1, seq2, k): 
    # initializing string
    test_str = seq1 + seq2

    # initializing K
    K = k
    
    # calling generator function
    res = ["".join(ele) for ele in over_slice(test_str, K)]
    dfMappingFunction = pd.DataFrame(columns = ["seq"] + res)
    dfMappingFunction.loc[0, "seq"] = seq1
    dfMappingFunction.loc[1, "seq"] = seq2

    # generate mapping function
    def getMappingFunction(res, seq):
        if res in seq: # if the k-mer is present in the sequence, count the number of occurences in the sequence
            dfMappingFunction.loc[dfMappingFunction["seq"] == seq, res] = dfMappingFunction.loc[dfMappingFunction["seq"] == seq, "seq"].reset_index(drop=True)[0].count(res) 
        else: # if its not in the sequence, fill with a 0
            dfMappingFunction.loc[dfMappingFunction["seq"] == seq, res] = 0

    for seq in [seq1, seq2]:
        [getMappingFunction(x, seq) for x in res]

    # generate p-spectra
    pSpectrumKernel = pd.DataFrame(columns=[seq1, seq2])
    pSpectrumKernel.insert(0, "seq", [seq1, seq2])

    # to derive p-spectra, take product of each column and sum together
    products_WT_mut = [np.prod(dfMappingFunction.iloc[:, i]) for i in range(1, len(dfMappingFunction.columns))]
    sum_WT_mut  = np.sum(products_WT_mut)
    pSpectrumKernel.iloc[0, 2] = sum_WT_mut
    pSpectrumKernel.iloc[1, 1] = sum_WT_mut
    # to derive the diagonal of the kernel matrix, we take the sum of the squares for the corresponding row
    squares_WT = [np.square(dfMappingFunction.iloc[0, i]) for i in range(1, len(dfMappingFunction.columns))]
    squares_mutant = [np.square(dfMappingFunction.iloc[1, i]) for i in range(1, len(dfMappingFunction.columns))]
    pSpectrumKernel.iloc[0, 1] = np.sum(squares_WT)
    pSpectrumKernel.iloc[1, 2] = np.sum(squares_mutant)
    return pSpectrumKernel.drop("seq", axis = 1)
    
def getFinalSpectrumDf(windowSize, kmerSize):
    # kmer of window size 5 and kmer size 3
    # get sequences given a set of variants and specified sequence lengths
    variantsSequences = getSequences(variants, windowSize, kmerSize)
    spectrumFeatureList = [getSpectrumFeatures(list(variantsSequences.loc[i, :])[0], list(variantsSequences.loc[i, :])[1], kmerSize) for i in range(0, len(variantsSequences))]
    spectrumList = [list(spectrumFeatureList[x].loc[0, :]) + list(spectrumFeatureList[x].loc[1, :]) for x in range(0, len(spectrumFeatureList))]
    spectrumdf = pd.DataFrame(spectrumList)
    spectrumdf = pd.concat([variants, spectrumdf], axis = 1)
    spectrumdf["chrom"] = spectrumdf["chrom"].str.replace("chr", "").astype(str)
    spectrumdf["pos"] = spectrumdf["pos"].astype(int)
    spectrumdf = spectrumdf.rename(columns = {0: str(windowSize*2) + "_" + str(kmerSize) + "_w", 1: str(windowSize*2) + "_" + str(kmerSize) + "_x", 2: str(windowSize*2) + "_" + str(kmerSize) + "_y", 3: str(windowSize*2) + "_" + str(kmerSize) + "_z"})
    spectrumdf = spectrumdf.drop("pos2", axis = 1)    


def get_chunk_variants(filename, start_line, end_line):
    # Read only the chunk of variants specified
    df_chunk = pd.read_csv(filename, sep='\t', skiprows=range(1, start_line), nrows=end_line-start_line+1, header=None, names=['chrom', 'pos', 'pos2', 'ref_allele', 'alt_allele'])
    return df_chunk[(df_chunk['chrom'] != "chrX") & (df_chunk['chrom'] != "chrY")]

def main(start_line, end_line, variants_file, output_dir):
    variants = get_chunk_variants(variants_file, start_line, end_line)
    variants = variants.reset_index(drop=True)

    record_dict = SeqIO.to_dict(SeqIO.parse("hg38_seq.fasta", "fasta"))

    window_sizes = [1, 2, 3, 4, 5]
    kmer_sizes_list = [list(range(1, 2)), list(range(1, 4)), list(range(1, 6)), list(range(1, 8)), list(range(1, 10))]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        for window_size, kmer_sizes in zip(window_sizes, kmer_sizes_list):
            result = pool.starmap(process_kmer, [(window_size, kmer_size, variants) for kmer_size in kmer_sizes])
            results.extend(result)

if __name__ == "__main__":
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    input_variants = sys.argv[3]
    output_directory = sys.argv[4]

    main(start, end, input_variants, output_directory)