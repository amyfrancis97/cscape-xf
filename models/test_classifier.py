#%%
import sys

sys.path.append('/user/home/uw20204/CanDrivR_data/cscape-xf/test_data')
from get_test_data import *
import pandas as pd
sys.path.append('/user/home/uw20204/CanDrivR_data/cscape-xf/optimisation')
from sequencial_feature_selection import *
from selected_features import *
from params import xgb_params, ann_params
sys.path.append('/user/home/uw20204/CanDrivR_data/cscape-xf/models')
from prepare_training_data import prepare_data
from train_evaluate import train_and_evaluate_model
from metric_results_table import get_results_table
from train_classifier import train_classifier
from DeepFFN import *
from ANN import *
#%%
from sklearn.metrics import balanced_accuracy_score
#%%
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
#%%
def test_model(train_data, test_variants, features):
    """
    Test ICGC/ 1000G test dataset.

    Parameters:
        df (pd.DataFrame): The input dataframe containing the dataset to be tested with features.
        features (list, optional): List of feature columns. If None, default columns are used. Default is None.
    Returns:
        dict: A dictionary containing the performance metrics for the test dataset on the model.
        pd.DataFrame: A dataframe containing the confidence scores for each of the predicted variants.
        pd.DataFrame: The final test variants (excluding those removed due to being present in training dataset).
    """

    # Remove any variants from test data that are present in training data
    train_data["id"] = train_data["chrom"] + "_" + train_data["pos"].astype(str) + "_" + train_data["ref_allele"] + "_" + train_data["alt_allele"]
    test_variants["id"] = test_variants["chrom"] + "_" + test_variants["pos"].astype(str) + "_" + test_variants["ref_allele"] + "_" + test_variants["alt_allele"]
    test_variants = test_variants[~test_variants['id'].isin(list(set(train_data["id"]) & set(test_variants["id"])))].drop("id", axis = 1).reset_index(drop = True)
    print(len(test_variants))
    features_updated = list(set(test_variants.columns.tolist()) & set(features))
    print(len(features_updated))
    actual_predicted_targets, feature_importance, final_model, results = train_classifier(train_data, features = features_updated, classifier = XGBClassifier(random_state=42))

    gold_standards_features = test_variants[final_model[1]]

    # Predict probabilities
    y_prob = final_model[0].predict_proba(gold_standards_features)

    y_val_pred = final_model[0].predict(gold_standards_features)

    test_variants["id"] = test_variants["chrom"] + "_" + test_variants["pos"].astype(str) + "_" + test_variants["ref_allele"] +  "/" + test_variants["alt_allele"]

    test_variants["positive_confidence"] = y_prob[:, 1]
    test_variants["negative_confidence"] = y_prob[:, 0]

    test_variants[["id", "positive_confidence", "negative_confidence"]]

    # just removing this variant as it is and error in the feature script (ref is T, not G)

    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
        'balanced_acc': []
    }
    fpr, tpr, thresholds = roc_curve(test_variants["driver_stat"], y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    curve_plotting = fpr, tpr, thresholds, roc_auc

    metrics['accuracy'].append(accuracy_score(test_variants["driver_stat"], y_val_pred))
    metrics['precision'].append(precision_score(test_variants["driver_stat"], y_val_pred))
    metrics['recall'].append(recall_score(test_variants["driver_stat"], y_val_pred))
    metrics['f1'].append(f1_score(test_variants["driver_stat"], y_val_pred))
    metrics['roc_auc'].append(roc_auc_score(test_variants["driver_stat"], y_prob[:, 1]))
    #metrics['balanced_acc'].append(balanced_accuracy_score(test_variants["driver_stat"], y_val_pred))

    test_variants["confidence_scores"] = y_prob[np.arange(len(y_val_pred)), y_val_pred]
    test_variants["actual_vals"] = test_variants["driver_stat"]
    test_variants["predicted_vals"] = y_val_pred
    confidence_scores = test_variants[["id", "actual_vals", "predicted_vals", "confidence_scores"]]

    return metrics, confidence_scores, test_variants, curve_plotting

#%%

if __name__ == "__main__":
    # This gets the larger ICGC/1000G dataset for testing
    def get_res(allele_freq_1000G_lower_threshold, ICGC_donor_count_threshold):
        test_variants, test_data = get_ICGC_1000G_dataset(allele_freq_1000G_lower_threshold, ICGC_donor_count_threshold)
        test_variants = pd.read_csv("/Volumes/Samsung_T5/data/test_data_ICGC_1000G_variants.txt", sep = "\t")
        test_data = pd.read_csv("/Volumes/Samsung_T5/data/test_data_ICGC_1000G_features_only.txt", sep = "\t")
        metrics = test_model(test_variants, test_data, features)
        print(allele_freq_1000G_lower_threshold, ICGC_donor_count_threshold, "")
        print(metrics)
    
    get_res(0.05, 1)

#%%



# %%
if __name__ == "__main__":
    test_variants = pd.read_csv("/Volumes/Samsung_T5/data/gold_standard_variants_all.txt", sep = "\t")
    test_data = pd.read_csv("/Volumes/Samsung_T5/data/gold_standard_with_features.txt", sep = "\t")
    metrics = test_model(test_variants, test_data, features)
    print(metrics)

    test_variants = pd.read_csv("/Volumes/Samsung_T5/data/test_data_ICGC_1000G_variants.txt", sep = "\t")
    test_data = pd.read_csv("/Volumes/Samsung_T5/data/test_data_ICGC_1000G_features_only.txt", sep = "\t")

    df = pd.read_csv("/Volumes/Samsung_T5/data/sample_cosmic_gnomad43000.txt", sep = "\t")
    #nucleotide_encoding = pd.read_csv("/Volumes/Samsung_T5/data/gnomad_cosmic/14000_sample_nucleotide_21bp_encoding.txt", sep = "\t")
    #df2 = pd.concat([df2, nucleotide_encoding],axis = 1)
    df = df.drop(df.columns[df.isna().any()].tolist(), axis = 1)

    # Remove any variants from test data that are present in training data
    df["id"] = df["chrom"] + "_" + df["pos"].astype(str) + "_" + df["ref_allele"] + "_" + df["alt_allele"]
    test_variants["id"] = test_variants["chrom"] + "_" + test_variants["pos"].astype(str) + "_" + test_variants["ref_allele"] + "_" + test_variants["alt_allele"]
    test_data = test_data[~test_variants['id'].isin(list(set(df["id"]) & set(test_variants["id"])))].reset_index(drop = True)
    test_variants = test_variants[~test_variants['id'].isin(list(set(df["id"]) & set(test_variants["id"])))].drop("id", axis = 1).reset_index(drop = True)
    #%%
