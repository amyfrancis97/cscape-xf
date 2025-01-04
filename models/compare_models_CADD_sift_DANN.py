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
from test_classifier import test_model
from DeepFFN import *
from ANN import *
import time
sys.path.append('/Users/uw20204/Documents/scripts/CanDrivR/optimisation')
from selected_features import *
from params import xgb_params, ann_params
from typing import *
sys.path.append('/Users/uw20204/Documents/scripts/CanDrivR/data/test_data')
from get_test_data import get_test_set
test_variants = get_test_set(thousandG_allele_freq = 0.05, ICGC_count = 1, features = features)
#%%
from pyliftover import LiftOver
test_vars_positions = test_variants[["chrom", "pos", "ref_allele", "alt_allele"]]
#%%
test_vars_positions
#%%
cscape_coding_train
#%%

cscape_coding_train = pd.read_csv("/Users/uw20204/Downloads/cscape_coding_tests/cscape_coding_TCGA_test.tab", sep = "\t")
cscape_coding_train["Chromosome"] = "chr" + cscape_coding_train["Chromosome"].astype(str)
#%%

cscape_coding_train
#%%
icgc_synonymous = pd.read_csv("/Volumes/Seagate5TB/data/filtered_somatic_mutation.bed", sep ="\t", header = None)
icgc_synonymous
#%%
icgc_synonymous = icgc_synonymous[icgc_synonymous[0] != "MT"]
#%%
icgc_synonymous["study_id"] = icgc_synonymous[7].str.split("OCCURRENCE=", expand = True)[1].str.split("-", expand = True)[0]
#%%
icgc_synonymous["donor_count"] = icgc_synonymous[7].str.split("affected_donors=", expand = True)[1].str.split(";", expand = True)[0]
#%%
icgc_synonymous = icgc_synonymous[[0, 1, 3, 4, "study_id", "donor_count"]]
#%%
icgc_synonymous.columns = ["chrom", "pos", "ref_allele", "alt_allele", "study_id", "donor_count"]
#%%
icgc_synonymous["chrom"] = "chr" + icgc_synonymous["chrom"].astype(str)
#%%
# Initialize the LiftOver object with the appropriate chain file
#lo = LiftOver('/Users/uw20204/Downloads/hg38ToHg19.over.chain.gz')
lo = LiftOver('/Users/uw20204/Downloads/hg19ToHg38.over.chain.gz')

#%%
test_variants.to
#%%
# Function to convert a single coordinate
def convert_coordinate(chrom, pos):
    result = lo.convert_coordinate(chrom, pos)
    if result:
        new_chrom, new_pos = result[0][0], result[0][1]
        return new_chrom, int(new_pos)
    else:
        return None, None

# Apply the conversion to the DataFrame
converted_data = icgc_synonymous.apply(lambda row: convert_coordinate(row['chrom'], row['pos']), axis=1)
#%%
converted_data
#%%
icgc_synonymous[['new_chrom', 'new_pos']] = pd.DataFrame(converted_data.tolist(), index=icgc_synonymous.index)

icgc_synonymous = icgc_synonymous.dropna()
icgc_synonymous["new_pos"] = icgc_synonymous["new_pos"].astype("int64")
icgc_synonymous["new_chrom"] = icgc_synonymous["new_chrom"].str.replace("chr", "")
#%%
icgc_synonymous[["new_chrom", "new_pos", "ref_allele", "alt_allele", "study_id", "donor_count"]].drop_duplicates(keep = "first")
#%%
icgc_synonymous[["new_chrom", "new_pos", "new_pos", "ref_allele", "alt_allele"]].drop_duplicates(keep = "first").to_csv("/Volumes/Seagate5TB/data/ICGC_synonymous_vars.bed", sep = "\t", header = None, index = None)
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
#%%
# Best results are af of 0.1 and donor count of >2
res, sampled, curve_plotting_res = get_test_metrics(0.1, 2, features)
#%%
res
#%%
sampled[["chrom", "pos", "pos", "ref_allele", "alt_allele"]].to_csv("/Volumes/Seagate5TB/data/final_ICGC_1000G_test_data/final_test_variants.txt", sep = "\t", index = None, header = None)
#%%

#%%
# CanDrivR results
fpr1, tpr1, thresholds1, roc_auc1 = curve_plotting_res
#%%
# Loading CScape data and calculating ROC and AUC
cscape = pd.read_csv("/Volumes/Seagate5TB/data/cscape/ICGC_1000G_test_data_hg19_cscape_predictions_head.csv", sep="\t", names = ["new_chrom_hg37", "new_pos_hg37", "ref_allele", "alt_allele", "score_cscape", "_", "driver_status"], skiprows=1)
#%%
cscape["new_chrom_hg37"] = cscape["new_chrom_hg37"].astype("int64")
#%%
test_vars_positions["new_chrom_hg37"] = test_vars_positions["new_chrom_hg37"].astype("int64")
#%%
test_vars_positions["new_pos_hg37"]
#%%
cscape_res
#%%

cscape_res = cscape.merge(test_vars_positions, how = "left", on = ["new_chrom_hg37", "new_pos_hg37"])
cscape_res = cscape_res[["chrom", "pos", "ref_allele_x", "alt_allele_x", "driver_stat", "score_cscape"]]
#%%
cscape_res = cscape_res.dropna()
#%%
fpr6, tpr6, thresholds6 = roc_curve(cscape_res['driver_stat'], cscape_res['score_cscape'], pos_label=1)
roc_auc6 = auc(fpr6, tpr6)
#%%

#%%
# Loading SIFT data and calculating ROC and AUC
sift_scores = pd.read_csv("/Volumes/Seagate5TB/data/external_model_datasets/sift/all_chromosomes_sift.txt", sep="\t").dropna()
sift_scores
#%%
#sift_scores['predicted_label'] = (sift_scores['SIFT_score'] < 0.05).astype(int)
fpr2, tpr2, thresholds2 = roc_curve(sift_scores['driver_stat'], sift_scores['SIFT_score'], pos_label=0)
roc_auc2 = auc(fpr2, tpr2)
#%%
# Loading REVEL data and calculating ROC and AUC
sift_scores = pd.read_csv("/Volumes/Seagate5TB/data/external_model_datasets/REVEL/REVEL_scores.txt", sep="\t").dropna()

fpr3, tpr3, thresholds3 = roc_curve(sift_scores['driver_stat'], sift_scores['score'], pos_label=1)
roc_auc3 = auc(fpr3, tpr3)

# Loading CADD data and calculating ROC and AUC
sift_scores = pd.read_csv("/Volumes/Seagate5TB/data/external_model_datasets/CADD/all_chromosomes_CADD.txt", sep="\t").dropna()
fpr4, tpr4, thresholds4 = roc_curve(sift_scores['driver_stat'], sift_scores['cadd_raw'], pos_label=1)
roc_auc4 = auc(fpr4, tpr4)

# Loading DANN data and calculating ROC and AUC
sift_scores = pd.read_csv("/Volumes/Seagate5TB/data/external_model_datasets/DANN/all_chromosomes_DANN.txt", sep="\t").dropna()
fpr5, tpr5, thresholds5 = roc_curve(sift_scores['driver_stat'], sift_scores['score'], pos_label=1)
roc_auc5 = auc(fpr5, tpr5)
#%%

#%%
# Setting a style
plt.style.use('seaborn-whitegrid')
custom_palette = sns.color_palette("Dark2", 8)

# Plotting CanDrivR model
plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, color=custom_palette[0], linewidth=1, label=f'CanDrivR (AUC = {roc_auc1:.2f})', linestyle='-')

# CADD model
plt.plot(fpr4, tpr4, color=custom_palette[3], linewidth=1, label=f'CADD (AUC = {roc_auc4:.2f})', linestyle='-')

# DANN model
plt.plot(fpr5, tpr5, color=custom_palette[5], linewidth=1, label=f'DANN (AUC = {roc_auc5:.2f})', linestyle='-')


# REVEL model
plt.plot(fpr3, tpr3, color=custom_palette[2], linewidth=1, label=f'REVEL (AUC = {roc_auc3:.2f})', linestyle='-')

# SIFT-based model
plt.plot(fpr2, tpr2, color=custom_palette[1], linewidth=1, label=f'SIFT (AUC = {roc_auc2:.2f})', linestyle='-')

# CScape-based model
plt.plot(fpr6, tpr6, color=custom_palette[7], linewidth=1, label=f'CScape (AUC = {roc_auc6:.2f})', linestyle='-')


# Adding plot decorations
plt.plot([0, 1], [0, 1], color='grey', linestyle='-.', linewidth=1.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Comparison of ROC Curves', fontsize=16)
plt.legend(loc="lower right", frameon=True, fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Improving overall aesthetics
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Save as a high-resolution image
plt.savefig('/Users/uw20204/Documents/figures/ROC_Curves_Comparison.png', format='png', dpi=300, bbox_inches='tight')

plt.show()

# %%
from sklearn.metrics import precision_recall_curve, average_precision_score
sift_scores = pd.read_csv("/Volumes/Samsung_T5/data/external_model_datasets/sift/all_chromosomes_sift.txt", sep="\t").dropna()

precision, recall, thresholds = precision_recall_curve(sift_scores['driver_stat'], sift_scores['SIFT_score'], pos_label=0)
average_precision = average_precision_score(sift_scores['driver_stat'], sift_scores['SIFT_score'])

plt.figure(figsize=(6, 4))
plt.step(recall, precision, where='post', color='b', alpha=0.8, label=f'Average Precision = {average_precision:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve')
plt.legend(loc="upper right")
plt.show()
# %%
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Load SIFT data and calculate metrics
sift_scores = pd.read_csv("/Volumes/Seagate5TB/data/external_model_datasets/sift/all_chromosomes_sift.txt", sep="\t").dropna()
fpr2, tpr2, thresholds2 = roc_curve(sift_scores['driver_stat'], sift_scores['SIFT_score'], pos_label=0)
roc_auc2 = auc(fpr2, tpr2)

optimal_idx2 = np.argmax(tpr2 - fpr2)
optimal_threshold2 = thresholds2[optimal_idx2]
sift_scores['predicted_label'] = (sift_scores['SIFT_score'] < optimal_threshold2).astype(int)
precision2 = precision_score(sift_scores['driver_stat'], sift_scores['predicted_label'])
recall2 = recall_score(sift_scores['driver_stat'], sift_scores['predicted_label'])
f1_2 = f1_score(sift_scores['driver_stat'], sift_scores['predicted_label'])
conf_matrix2 = confusion_matrix(sift_scores['driver_stat'], sift_scores['predicted_label'])
accuracy2 = accuracy_score(sift_scores['driver_stat'], sift_scores['predicted_label'])

# Load REVEL data and calculate metrics
revel_scores = pd.read_csv("/Volumes/Seagate5TB/data/external_model_datasets/REVEL/REVEL_scores.txt", sep="\t").dropna()
fpr3, tpr3, thresholds3 = roc_curve(revel_scores['driver_stat'], revel_scores['score'], pos_label=1)
roc_auc3 = auc(fpr3, tpr3)

optimal_idx3 = np.argmax(tpr3 - fpr3)
optimal_threshold3 = thresholds3[optimal_idx3]
revel_scores['predicted_label'] = (revel_scores['score'] >= optimal_threshold3).astype(int)
precision3 = precision_score(revel_scores['driver_stat'], revel_scores['predicted_label'])
recall3 = recall_score(revel_scores['driver_stat'], revel_scores['predicted_label'])
f1_3 = f1_score(revel_scores['driver_stat'], revel_scores['predicted_label'])
conf_matrix3 = confusion_matrix(revel_scores['driver_stat'], revel_scores['predicted_label'])
accuracy3 = accuracy_score(revel_scores['driver_stat'], revel_scores['predicted_label'])

# Load CADD data and calculate metrics
cadd_scores = pd.read_csv("/Volumes/Seagate5TB/data/external_model_datasets/CADD/all_chromosomes_CADD.txt", sep="\t").dropna()
fpr4, tpr4, thresholds4 = roc_curve(cadd_scores['driver_stat'], cadd_scores['cadd_raw'], pos_label=1)
roc_auc4 = auc(fpr4, tpr4)

optimal_idx4 = np.argmax(tpr4 - fpr4)
optimal_threshold4 = thresholds4[optimal_idx4]
cadd_scores['predicted_label'] = (cadd_scores['cadd_raw'] >= optimal_threshold4).astype(int)
precision4 = precision_score(cadd_scores['driver_stat'], cadd_scores['predicted_label'])
recall4 = recall_score(cadd_scores['driver_stat'], cadd_scores['predicted_label'])
f1_4 = f1_score(cadd_scores['driver_stat'], cadd_scores['predicted_label'])
conf_matrix4 = confusion_matrix(cadd_scores['driver_stat'], cadd_scores['predicted_label'])
accuracy4 = accuracy_score(cadd_scores['driver_stat'], cadd_scores['predicted_label'])

# Load DANN data and calculate metrics
dann_scores = pd.read_csv("/Volumes/Seagate5TB/data/external_model_datasets/DANN/all_chromosomes_DANN.txt", sep="\t").dropna()
fpr5, tpr5, thresholds5 = roc_curve(dann_scores['driver_stat'], dann_scores['score'], pos_label=1)
roc_auc5 = auc(fpr5, tpr5)

optimal_idx5 = np.argmax(tpr5 - fpr5)
optimal_threshold5 = thresholds5[optimal_idx5]
dann_scores['predicted_label'] = (dann_scores['score'] >= optimal_threshold5).astype(int)
precision5 = precision_score(dann_scores['driver_stat'], dann_scores['predicted_label'])
recall5 = recall_score(dann_scores['driver_stat'], dann_scores['predicted_label'])
f1_5 = f1_score(dann_scores['driver_stat'], dann_scores['predicted_label'])
conf_matrix5 = confusion_matrix(dann_scores['driver_stat'], dann_scores['predicted_label'])
accuracy5 = accuracy_score(dann_scores['driver_stat'], dann_scores['predicted_label'])

cscape_res = cscape_res.dropna()
#%%
cscape_res = cscape_res.dropna()

#%%
fpr6, tpr6, thresholds6 = roc_curve(cscape_res['driver_stat'], cscape_res['score_cscape'], pos_label=1)
roc_auc6 = auc(fpr6, tpr6)
optimal_idx6 = np.argmax(tpr6 - fpr6)
optimal_threshold6 = thresholds5[optimal_idx6]
cscape_res['predicted_label'] = (dann_scores['score'] >= optimal_threshold6).astype(int)
precision6 = precision_score(cscape_res['driver_stat'], cscape_res['predicted_label'])
recall6 = recall_score(cscape_res['driver_stat'], cscape_res['predicted_label'])
f1_6 = f1_score(cscape_res['driver_stat'], cscape_res['predicted_label'])
conf_matrix6 = confusion_matrix(cscape_res['driver_stat'], cscape_res['predicted_label'])
accuracy6 = accuracy_score(cscape_res['driver_stat'], cscape_res['predicted_label'])


# Plotting ROC curves
plt.figure()
#plt.plot(fpr1, tpr1, label=f'CanDrivR (area = {roc_auc1:.2f})')
plt.plot(fpr2, tpr2, label=f'SIFT (area = {roc_auc2:.2f})')
plt.plot(fpr3, tpr3, label=f'REVEL (area = {roc_auc3:.2f})')
plt.plot(fpr4, tpr4, label=f'CADD (area = {roc_auc4:.2f})')
plt.plot(fpr5, tpr5, label=f'DANN (area = {roc_auc5:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Displaying the metrics
metrics_df = pd.DataFrame({
    'Model': ['CanDrivR', 'SIFT', 'REVEL', 'CADD', 'DANN', 'CScape'],
    'AUC': [roc_auc1, roc_auc2, roc_auc3, roc_auc4, roc_auc5, roc_auc6],
    'Precision': [res["precision"][0], precision2, precision3, precision4, precision5,precision6],
    'Recall': [res["recall"][0], recall2, recall3, recall4, recall5, recall6],
    'F1 Score': [res["f1"][0], f1_2, f1_3, f1_4, f1_5, f1_6],
    'Accuracy': [res["accuracy"][0], accuracy2, accuracy3, accuracy4, accuracy5,accuracy6]
})


# %%
round(metrics_df, 3).sort_values("F1 Score", ascending=False)
# %%
