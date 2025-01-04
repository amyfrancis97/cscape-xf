import pandas as pd
import sys
import os
from xgboost import XGBClassifier

from prepare_training_data import prepare_data
from train_evaluate import train_and_evaluate_model
from metric_results_table import get_results_table
from train_classifier import train_classifier


def run_sample(dataset):
    """
    Run XGBoost with different dataset sizes to optimise performance.

    Parameters:
        dataset (pd.DataFrame): A dataframe containing the sampled dataset.

    Returns:
        pd.DataFrame: Dataframe containing mean and standard deviations of metrics, and the run time for a given sample.
    """
    # Load the dataset from a file
    sample = pd.read_csv(dataset, sep="\t")

    # Drop columns with any missing values
    sample = sample.drop(sample.columns[sample.isna().any()].tolist(), axis=1)

    # Train the classifier on the sample data
    actual_predicted_targets, feature_importance, final_model, results = train_classifier(sample, classifier=XGBClassifier(random_state=42))
    
    # Process the results to get a summary table
    res = get_results_table(results[0], model_name="XGB")
    
    # Add the elapsed time and sample size to the results
    res["elapsed_time (s)"] = results[1]["time"]
    res["number_samples"] = len(sample)
    return res

if __name__ == "__main__":
    # Extract the path to the sampled data and output directory from command-line arguments
    sample_data_path = sys.argv[1]
    datasets = os.listdir(f'{sample_data_path}sample_cosmic_gnomad')
    os.chdir(f'{sample_data_path}sample_cosmic_gnomad')
    outputDir = sys.argv[2]

    # Process each sampled dataset and collect the results
    sample_res = [run_sample(dataset) for dataset in datasets]

    # Concatenate all result DataFrames
    sample_res = pd.concat(sample_res)

    # Save the results to a file
    sample_res.to_csv(f'{outputDir}sample_vs_accuracy.txt', sep="\t", index=None)
