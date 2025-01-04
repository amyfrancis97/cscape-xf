#%%
import os 
home_dir = "/user/home/uw20204/CanDrivR_data/cscape-xf/"

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
#%%
sys.path.append(home_dir + 'models')
from prepare_training_data import prepare_data
from train_evaluate import train_and_evaluate_model
from metric_results_table import get_results_table
from train_classifier import train_classifier
from DeepFFN import *
from ANN import *
import time
sys.path.append(home_dir + 'optimisation')
from selected_features import *
from params import xgb_params, ann_params
from typing import *
#%%
df
#%%
import sys
sys.path.append('/Users/uw20204/Documents/scripts/CanDrivR-TS')
data_dir = "/Volumes/Seagate5TB/data"
output_figure_dir = "/Users/uw20204/Documents/figures/"
#%%
# Read in the training dataset
df = pd.read_csv("/Volumes/Seagate5TB/data/sample_cosmic_gnomad43000.txt", sep = "\t")
#%%
import random
cosmic_gnomad = pd.read_csv(f"{data_dir}/all_features_cosmic_gnomad.bed.gz", compression= "gzip", sep = "\t")

#%%
cosmic_counts = pd.read_csv(f"{data_dir}/final_cosmic_vars_w_count.txt", sep = "\t", names = ["chrom", "pos", "pos1", "ref_allele", "alt_allele", "donor_count"])
#%%

#%%
df = cosmic_gnomad.copy()
#%%
import random
# Define a list of groups and ensure each is used exactly twice
groups = list(range(1, 12)) * 2  # Duplicating group range to cover 22 assignments
random.shuffle(groups)  # Shuffle to randomize distribution

# List of chromosomes
chromosomes = [
    'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9',
    'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
    'chr18', 'chr19', 'chr20', 'chr21', 'chr22'
]

# Assign shuffled groups to chromosomes to ensure even distribution
chromosome_to_group = dict(zip(chromosomes, groups))

# Apply the mapping to DataFrame
df['grouping'] = df['chrom'].map(chromosome_to_group)
#%%
cosmic_gnomad_with_counts = cosmic_gnomad.merge(cosmic_counts, how = "left")
#%%
cosmic_gnomad_with_counts_filtered = pd.concat([cosmic_gnomad_with_counts[cosmic_gnomad_with_counts["donor_count"] > 8], cosmic_gnomad_with_counts[cosmic_gnomad_with_counts["driver_stat"] == 0].sample(394)])
#%%

##### Comparing models
# Run XGB model
actual_predicted_targets, feature_importance, final_model, results = train_classifier(cosmic_gnomad_with_counts_filtered, features = features, classifier = XGBClassifier(random_state=42, **xgb_params), feature_importance=True)
XGB_results_metrics = get_results_table(results[0], model_name = "XGB")
XGB_results_metrics["time (s)"] = round(float(results[1]["time"][0]), 2)
XGB_results_metrics
#%%
ICGC = pd.read_csv(f"{data_dir}/all_features_ICGCv2.bed.gz", sep = "\t", compression='gzip')
df
#%%
# Sort by importance
feature_importance_plotting = feature_importance.sort_values(by='importance', ascending=False)[:30]

plot_feature_importance(feature_importance_plotting, "/Users/uw20204/Documents/figures")
#%%
cons = [x for x in feature_importance_plotting["feature"].tolist() if "phastCons" in x]
#%%
df = df.drop_duplicates(subset = ["chrom", "pos", "ref_allele", "alt_allele"], keep = False)
#%%
plot_violin(df.dropna(), cons[:8], "GnomAD", output_filename = "violin_plot_cosmic_gnomad.png")
#%%
def get_distributions(df, filename)
#%%
cosmic_gnomad 
#%%
df = ICGC_sampled
#%%
custom_palette = sns.color_palette("Set2", 8)
#%%
# Set up the matplotlib figure
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
axes = axes.flatten()

bar_width = 0.1  # Width of each bar
offset = bar_width * 0.7   # Offset between bars

for i, feature in enumerate(cons):
    data_pos = df[df["driver_stat"] == 1][feature].dropna()
    data_neg = df[df["driver_stat"] == 0][feature].dropna()
    
    bins = np.linspace(min(data_pos.min(), data_neg.min()), max(data_pos.max(), data_neg.max()), 20)
    
    # Calculate histogram for driver_stat == 1 (positive)
    counts_pos, edges = np.histogram(data_pos, bins=bins)
    counts_neg, _ = np.histogram(data_neg, bins=bins)

    # Shift the bins for driver_stat == 1 to the left
    bar_positions_pos = edges[:-1] - offset
    # Shift the bins for driver_stat == 0 to the right
    bar_positions_neg = edges[:-1] + offset

    # Plot histogram for driver_stat == 1 (positive)
    axes[i].bar(bar_positions_pos, counts_pos, width=bar_width, color=custom_palette[1], alpha=0.7, label='COSMIC')
    # Plot histogram for driver_stat == 0 (negative)
    axes[i].bar(bar_positions_neg, counts_neg, width=bar_width, color=custom_palette[0], alpha=0.7, label='GnomAD')

    # Set labels and title for each subplot
    axes[i].set_title(f'{feature}', fontsize=16, weight='bold')
    axes[i].set_xlabel('Score', fontsize=14, weight='bold')
    axes[i].set_ylabel('Frequency', fontsize=14, weight='bold')
    axes[i].legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure with a high DPI
plt.savefig(f"/Users/uw20204/Documents/figures/test.png", dpi=300)

# Show the plot
plt.show()
#%%

# Set up the matplotlib figure with 2 columns and 3 rows
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

bar_width = 0.02  # Width of each bar
offset = bar_width * 0.7   # Offset between bars

features = ["hg38phyloP100wayscore", "hg38phastCons30wayscore", "hg38phyloP4wayscore"]
feature = features[1]
for col, driver_stat in enumerate([0, 1]):
    data_pos_ICGC = ICGC_not_sampled[ICGC_not_sampled["driver_stat"] == driver_stat][feature].dropna()
    data_pos_cosmic_gnomad = cosmic_gnomad[cosmic_gnomad["driver_stat"] == driver_stat][feature].dropna().sample(10000)

    bins = np.linspace(min(data_pos_ICGC.min(), data_pos_cosmic_gnomad.min()), max(data_pos_ICGC.max(), data_pos_cosmic_gnomad.max()), 20)

    # Calculate histogram for driver_stat
    counts_ICGC, edges = np.histogram(data_pos_ICGC, bins=bins)
    counts_cosmic_gnomad, _ = np.histogram(data_pos_cosmic_gnomad, bins=bins)

    # Shift the bins
    bar_positions_pos = edges[:-1] - offset
    bar_positions_neg = edges[:-1] + offset

    if driver_stat == 1:
        ICGC_name = "ICGC Positive (r > 2)"
        cosmic_gnomad_name = "COSMIC"
        col_ICGC = custom_palette[1]
        col_cosmic_gnomad = custom_palette[3]
    else:
        ICGC_name = "ICGC Negative (r = 1)"
        cosmic_gnomad_name = "GnomAD"
        col_ICGC = custom_palette[0]
        col_cosmic_gnomad = custom_palette[2]

    # Plot histograms
    axes[col].bar(bar_positions_pos, counts_ICGC, width=bar_width, color=col_ICGC, alpha=0.7, label=ICGC_name)
    axes[col].bar(bar_positions_neg, counts_cosmic_gnomad, width=bar_width, color=col_cosmic_gnomad, alpha=0.7, label=cosmic_gnomad_name)

    # Set labels and title for each subplot
    axes[col].set_title(f"Distribution for Driver Status of '{driver_stat}'", fontsize=14, weight='bold')
    axes[col].set_xlabel(f'{feature} Score', fontsize=12, weight='bold')
    axes[col].set_ylabel('Frequency', fontsize=12, weight='bold')
    axes[col].legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig(f"/Users/uw20204/Documents/figures/ICGC_COSMIC_GnomAD_{feature}_distribution.png", dpi=300)

# Show the plot
plt.show()
#%%
data_neg_ICGC = df[df["driver_stat"] == 0]["hg38phyloP100wayscore"].dropna()
data_neg_cosmic_gnomad = cosmic_gnomad[cosmic_gnomad["driver_stat"] == 0]["hg38phyloP100wayscore"].dropna()
plt.hist(data_neg_ICGC)
#%%
plt.hist(data_neg_cosmic_gnomad)
#%%
ICGC = pd.read_csv("/Volumes/Seagate5TB/data/all_features_ICGC.bed.gz", sep = "\t", compression='gzip')
ICGC
#%%
ICGC = df
#%%
positives = ICGC[ICGC["donor_count"] > 2].sample(10000)
negatives = ICGC[ICGC["donor_count"] == 1].sample(10000)
ICGC_not_sampled = pd.concat([positives, negatives], axis = 0)
#%%
actual_predicted_targets_ICGC, feature_importance_ICGC, final_model_ICGC, results_ICGC = train_classifier(ICGC_sampled, features = features, classifier = XGBClassifier(random_state=42), feature_importance=True)
#%%
scaler = StandardScaler()
X_test = scaler.fit_transform(df[final_model[1]])

# Predict on validation set
y_val_pred = final_model[0].predict(X_test)
y_val_pred
#%%
X_test
#%%
accuracy_score(df["driver_stat"], y_val_pred)
#%%
feature_importance_plotting = feature_importance.sort_values(by='importance', ascending=False)[:30]

plot_feature_importance(feature_importance_plotting, "/Users/uw20204/Documents/figures")
#%%
get_distributions(ICGC_sampled, "distribution_phastcons.png")
#%%
# Set up the matplotlib figure with 2 columns and 1 row
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

bar_width = 0.02  # Width of each bar
offset = bar_width * 0.7  # Offset between bars

features = ["hg38phyloP100wayscore", "hg38phastCons30wayscore", "hg38phyloP4wayscore"]
feature = features[1]

# Plot COSMIC and GnomAD
driver_stat = 1
data_cosmic = cosmic_gnomad[cosmic_gnomad["driver_stat"] == driver_stat][feature].dropna().sample(10000)
data_gnomad = cosmic_gnomad[cosmic_gnomad["driver_stat"] == 0][feature].dropna().sample(10000)

bins = np.linspace(min(data_cosmic.min(), data_gnomad.min()), max(data_cosmic.max(), data_gnomad.max()), 20)

counts_cosmic, edges = np.histogram(data_cosmic, bins=bins)
counts_gnomad, _ = np.histogram(data_gnomad, bins=bins)

bar_positions_cosmic = edges[:-1] - offset
bar_positions_gnomad = edges[:-1] + offset

axes[0].bar(bar_positions_cosmic, counts_cosmic, width=bar_width, color=custom_palette[3], alpha=0.7, label="COSMIC")
axes[0].bar(bar_positions_gnomad, counts_gnomad, width=bar_width, color=custom_palette[2], alpha=0.7, label="GnomAD")

axes[0].set_title(f"Distribution for COSMIC and GnomAD", fontsize=14, weight='bold')
axes[0].set_xlabel(f'{feature} Score', fontsize=12, weight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, weight='bold')
axes[0].legend()

# Plot ICGC (r = 1 and r > 2)
data_icgc_r1 = ICGC_not_sampled[ICGC_not_sampled["driver_stat"] == 0][feature].dropna()
data_icgc_r2 = ICGC_not_sampled[ICGC_not_sampled["driver_stat"] == 1][feature].dropna()

bins = np.linspace(min(data_icgc_r1.min(), data_icgc_r2.min()), max(data_icgc_r1.max(), data_icgc_r2.max()), 20)

counts_icgc_r1, edges = np.histogram(data_icgc_r1, bins=bins)
counts_icgc_r2, _ = np.histogram(data_icgc_r2, bins=bins)

bar_positions_icgc_r1 = edges[:-1] - offset
bar_positions_icgc_r2 = edges[:-1] + offset

axes[1].bar(bar_positions_icgc_r1, counts_icgc_r1, width=bar_width, color=custom_palette[0], alpha=0.7, label="ICGC Negative (r = 1)")
axes[1].bar(bar_positions_icgc_r2, counts_icgc_r2, width=bar_width, color=custom_palette[1], alpha=0.7, label="ICGC Positive (r > 2)")

axes[1].set_title(f"Distribution for ICGC (r = 1 and r > 2)", fontsize=14, weight='bold')
axes[1].set_xlabel(f'{feature} Score', fontsize=12, weight='bold')
axes[1].set_ylabel('Frequency', fontsize=12, weight='bold')
axes[1].legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig(f"/Users/uw20204/Documents/figures/ICGC_COSMIC_GnomAD_{feature}_distribution.png", dpi=300)

# Show the plot
plt.show()
#%%
# Run SVM model
actual_predicted_targets, feature_importance, final_model, results = train_classifier(df, features = features, classifier = SVC(random_state=42, probability=True))
SVM_results_metrics = get_results_table(results[0], model_name = "SVM")
SVM_results_metrics["time (s)"] = round(float(results[1]["time"][0]), 2)
SVM_results_metrics

# Run RF model
actual_predicted_targets, feature_importance, final_model, results = train_classifier(df, features = features, classifier = RandomForestClassifier(random_state=42))
RF_results_metrics = get_results_table(results[0], model_name = "RF")
RF_results_metrics["time (s)"] = round(float(results[1]["time"][0]), 2)

# Run DFFN model
results, total_time, model = DeepFFN_model(df, features = features)
DFFN_results_metrics = get_results_table(results = results, model_name = "DFFN")

DFFN_results_metrics["time (s)"] = round(float(total_time), 2)

# Run ANN model
results, total_time, model = ANN_model(df, features = features)
ANN_results_metrics = get_results_table(results = results, model_name = "ANN")
ANN_results_metrics["time (s)"] = round(float(total_time), 2)
#%%
pd.concat([XGB_results_metrics, SVM_results_metrics, RF_results_metrics, DFFN_results_metrics, ANN_results_metrics])

# %%
# Write crude model comparison to CSV
model_comparison = pd.concat([XGB_results_metrics, SVM_results_metrics, RF_results_metrics, DFFN_results_metrics, ANN_results_metrics])
model_comparison.insert(0, "models", model_comparison["model"])
model_comparison = model_comparison.drop("model", axis = 1)
model_comparison.to_csv('/Volumes/Samsung_T5/data/model_comparison_res.txt', sep = "\t", index = None)
# %%
# Re-run XGB with optimised hyperparams
actual_predicted, feature_importance, final_model, results = train_classifier(df, features = features, classifier = XGBClassifier(random_state=42), feature_importance=True)
XGB_results_metrics = get_results_table(results[0], model_name = "XGB")
XGB_results_metrics["time (s)"] = round(float(results[1]["time"][0]), 2)
#%%
results
#%%
results_sample_size = []
for i in [50, 100, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000, 35000, 43000]:
    # Re-run XGB with optimized hyperparameters
    df2 = df.sample(i*2)
    df2
    actual_predicted, feature_importance, final_model, results = train_classifier(df2, features=features, classifier=XGBClassifier(random_state=42), feature_importance=False)
    XGB_results_metrics = get_results_table(results[0], model_name="XGB")
    XGB_results_metrics["time (s)"] = round(float(results[1]["time"][0]), 2)
    print(len(df2))
    print(XGB_results_metrics)
    results_sample_size.append([actual_predicted, feature_importance, final_model, results, XGB_results_metrics])
#%%

pd.DataFrame[results_sample_size[i][4] for i in range(len(results_sample_size))]
#%%

actual_predicted, feature_importance, final_model, results = train_classifier(df, features = features, classifier = XGBClassifier(random_state=42), feature_importance=True)
XGB_results_metrics = get_results_table(results[0], model_name = "XGB")
XGB_results_metrics["time (s)"] = round(float(results[1]["time"][0]), 2)
#%%
XGB_results_metrics
#%%
sys.path.append('/Users/uw20204/Documents/scripts/CanDrivR/plots')
from confusionMatrix import *
os.chdir("/Users/uw20204/Documents/scripts/CanDrivR/plots")
%run confusionMatrix
plot_confusion_matrix(actual_predicted[0][1], actual_predicted[0][0], "Confusion Matrix for Validation Dataset") # Cross-val predictions
plot_confusion_matrix(actual_predicted[1][1], actual_predicted[1][0], "Confusion Matrix for Test Dataset") # Test predictions
# %%
feature_importance
#%%
# Plot a violin plot of the top features
plot_violin(df, feature_importance[:8], "GnomAD")
#%%
sys.path.append('/Users/uw20204/Documents/scripts/CanDrivR/models')
from test_classifier import test_model
#%%
test_variants = pd.read_csv("/Volumes/Seagate5TB/data/gold_standard_with_features.txt", sep = "\t")
print(test_variants)
metrics, confidence_scores, test_variants, curve_plotting = test_model(df, test_variants, features)
print(metrics)
# %%
sys.path.append('/Users/uw20204/Documents/scripts/CanDrivR/plots')
from plot_test_probabilities import plot_probability_scores
from plot_feature_differences import plot_differences
#%%
os.chdir('/Users/uw20204/Documents/scripts/CanDrivR/plots')
%run plot_test_probabilities
#%%
# Plot the negatives
false_pos = confidence_scores[(confidence_scores["actual_vals"] == 0) & (confidence_scores["predicted_vals"] == 1)]
true_neg = confidence_scores[(confidence_scores["actual_vals"] == 0) & (confidence_scores["predicted_vals"] == 0)]
plot_probability_scores(true_neg, false_pos, title = "Predictions and Confidence Scores for Negative Variants", correct_predictions = "True Negative", incorrect_predictions = "False Positive")

#%%
# Plot the positives
true_pos = confidence_scores[(confidence_scores["actual_vals"] == 1) & (confidence_scores["predicted_vals"] == 1)]
false_neg = confidence_scores[(confidence_scores["actual_vals"] == 1) & (confidence_scores["predicted_vals"] == 0)]
plot_probability_scores(true_pos, false_neg, title = "Predictions and Confidence Scores for Positive Variants", correct_predictions = "True Positive", incorrect_predictions = "False Negative", negatives = False)


# %%
os.chdir('/Users/uw20204/Documents/scripts/CanDrivR/plots')
%run plot_feature_differences
# Plot the mean features for correctly classified positive variant compared to feature value for mis-classified
plot_differences(confidence_scores, features, test_variants)
# %%
from get_test_data import *
#%%
test_variants = get_test_set(thousandG_allele_freq = 0.05, ICGC_count = 1, features = features)
#%%

def get_test_metrics(af, donor_count, features):
    negatives = test_variants[(test_variants["driver_stat"] == 0) & (test_variants["af"] > af)]
    positives = test_variants[(test_variants["driver_stat"] == 1) & (test_variants["donor_count"] > donor_count)].sample(len(negatives))
    sampled = pd.concat([negatives, positives], axis = 0)
    metrics, _ , _ , curve_plotting_res = test_model(train_data = df, test_variants = sampled, features = features)
    metrics["af"] = af
    metrics["donor_count"] = donor_count
    metrics["data_size"] = len(sampled)
    print(metrics)
    return metrics, sampled, curve_plotting_res

# Best results are af of 0.1 and donor count of >2
res, sampled, curve_plotting_res = get_test_metrics(0.1, 2, features)
#%%
res
#%%
fpr, tpr, thresholds, roc_auc = curve_plotting_res
#%%
sampled[["chrom", "pos", "pos", "ref_allele", "alt_allele"]].to_csv("/Volumes/Samsung_T5/data/final_ICGC_1000G_test_data/final_test_variants.txt", sep = "\t", index = None, header = None)

#%%
# Test the classifier on ICGC/1000G data
sys.path.append('/Users/uw20204/Documents/scripts/CanDrivR/data/test_data')
from get_test_data import get_test_set
#%%
test_dataset = pd.read_csv("/Volumes/Samsung_T5/data/final_ICGC_1000G_test_data/final_test_sampled.txt", sep = "\t")

# %%
test_dataset
# %%

##############
import matplotlib.pyplot as plt
import numpy as np

# Set up the matplotlib figure with 2 columns and 1 row
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

bar_width = 0.5  # Width of each bar
offset = bar_width * 0.7   # Offset between bars

features = ["hg38phyloP100wayscore", "hg38phastCons30wayscore", "hg38phyloP470wayscore"]
feature = features[2]

data_pos_ICGC = ICGC_not_sampled[ICGC_not_sampled["donor_count"] == 1][feature].dropna()
data_neg_ICGC = ICGC_not_sampled[ICGC_not_sampled["donor_count"] > 2][feature].dropna()
data_pos_cosmic_gnomad = cosmic_gnomad[cosmic_gnomad["driver_stat"] == 1][feature].dropna().sample(10000)
data_neg_cosmic_gnomad = cosmic_gnomad[cosmic_gnomad["driver_stat"] == 0][feature].dropna().sample(10000)

bins = np.linspace(min(data_pos_ICGC.min(), data_neg_ICGC.min(), data_pos_cosmic_gnomad.min(), data_neg_cosmic_gnomad.min()), 
                   max(data_pos_ICGC.max(), data_neg_ICGC.max(), data_pos_cosmic_gnomad.max(), data_neg_cosmic_gnomad.max()), 20)

# Calculate histogram for ICGC Positive and Negative
counts_pos_ICGC, edges = np.histogram(data_pos_ICGC, bins=bins, density=True)
counts_neg_ICGC, _ = np.histogram(data_neg_ICGC, bins=bins, density=True)

# Calculate histogram for COSMIC and GnomAD
counts_pos_cosmic_gnomad, _ = np.histogram(data_pos_cosmic_gnomad, bins=bins, density=True)
counts_neg_cosmic_gnomad, _ = np.histogram(data_neg_cosmic_gnomad, bins=bins, density=True)

# Shift the bins
bar_positions_pos = edges[:-1] - offset
bar_positions_neg = edges[:-1] + offset

# Plot ICGC Positive vs. Negative
axes[0].bar(bar_positions_pos, counts_pos_ICGC, width=bar_width, color=custom_palette[1], alpha=0.7, label="ICGC Rare (r = 1)")

axes[0].bar(bar_positions_neg, counts_neg_ICGC, width=bar_width, color=custom_palette[0], alpha=0.7, label="ICGC Recurrent (r > 2)")

axes[0].set_title("ICGC Rare vs. Recurrent", fontsize=14, weight='bold')
axes[0].set_xlabel(f'{feature}', fontsize=12, weight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, weight='bold')
axes[0].legend()

# Plot COSMIC vs. GnomAD
axes[1].bar(bar_positions_pos, counts_pos_cosmic_gnomad, width=bar_width, color=custom_palette[1], alpha=0.7, label="COSMIC")

axes[1].bar(bar_positions_neg, counts_neg_cosmic_gnomad, width=bar_width, color=custom_palette[0], alpha=0.7, label="GnomAD")

axes[1].set_title("COSMIC vs. GnomAD", fontsize=14, weight='bold')
axes[1].set_xlabel(f'{feature}', fontsize=12, weight='bold')
axes[1].set_ylabel('Frequency', fontsize=12, weight='bold')
axes[1].legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig(f"/Users/uw20204/Documents/figures/ICGC_COSMIC_GnomAD_{feature}_distribution.png", dpi=300)

# Show the plot
plt.show()

# %%
data_pos_ICGC
# %%
