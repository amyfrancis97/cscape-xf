import os
import pandas as pd
from xgboost import XGBClassifier
from models.train_classifier import train_classifier
from models.metric_results_table import get_results_table
import random 

# Function to run feature selection
def run_feature_selection(df, features):
    """
    Trains a model using a subset of features and collects performance metrics.

    Parameters:
        df (pd.DataFrame): The dataset for training.
        features (list): List of features to include in the model.

    Returns:
        pd.DataFrame: Results table with performance metrics and features.
    """
    try:
        _, _, _, results = train_classifier(df, features=features, classifier=XGBClassifier(random_state=42))
        res = get_results_table(results[0], model_name="XGB")
        # Add elapsed time, number of features, and feature list to the results
        res["elapsed_time (s)"] = results[1]["time"]
        res["number_features"] = len(features)
        res["feature_list"] = [",".join(features)]  # Save features as a single string
        print(res.head())
        return res
    except Exception as e:
        print(f"Error with feature selection for {len(features)} features: {e}")
        return pd.DataFrame()

# Function to ensure the output directory exists
def ensure_output_directory(output_dir):
    """
    Ensures that the specified output directory exists.

    Parameters:
        output_dir (str): Path to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory verified: {output_dir}")

# Function to save DataFrame to a file
def save_results(df, file_path):
    """
    Saves a DataFrame to a specified file.

    Parameters:
        df (pd.DataFrame): DataFrame to save.
        file_path (str): Path to save the file.
    """
    try:
        df.to_csv(file_path, sep="\t", index=False)
        print(f"Results saved to {file_path}")
    except Exception as e:
        print(f"Error saving results to {file_path}: {e}")

# Define file paths
data_path = "/user/home/uw20204/CanDrivR_data/cscape-xf/data/all_features_cosmic_gnomad.bed.gz"
output_dir = "/user/home/uw20204/CanDrivR_data/cscape-xf/outputs"

# Ensure output directory exists
ensure_output_directory(output_dir)

# Specify the chunk size for reading the data
chunk_size = 10000

# Load data in chunks
chunks = []
try:
    for chunk in pd.read_csv(data_path, sep="\t", chunksize=chunk_size):
        chunks.append(chunk)
except Exception as e:
    print(f"Error reading the file: {e}")

# Combine chunks into a single DataFrame
df = pd.concat(chunks, ignore_index=True)

# Identify and process string columns
string_columns = df.select_dtypes(include=['object']).columns.tolist()
print("Columns that contain strings:", string_columns)
df = df.fillna(0)

# Exclude unwanted columns
df = df.drop(columns=string_columns[3:], errors='ignore')

random.seed(42)

# Sample data for feature importance analysis
sampled = df.sample(1000).reset_index(drop=True)

# Train an initial model to extract feature importances
print("Training initial model to extract feature importances...")
try:
    _, feature_importance, _, _ = train_classifier(sampled, classifier=XGBClassifier(random_state=42), feature_importance=True)
except Exception as e:
    print(f"Error during initial model training: {e}")
    exit()

# Save detailed feature lists to a separate file
feature_importance_file = os.path.join(output_dir, "feature_importances.txt")
try:
    feature_importance.to_csv(
        feature_importance_file, sep="\t", index=False
    )
    print(f"Feature importances saved to {feature_importance_file}")
except Exception as e:
    print(f"Error saving detailed feature importances: {e}")

# Generate feature subsets
importances_to_run = [
    feature_importance["feature"][0:i].tolist() for i in range(1, len(feature_importance))
]

# Process each subset of features and collect results
print("Starting feature selection optimisation...")
feature_results = [
    run_feature_selection(sampled, feature_list) for feature_list in importances_to_run
]

# Combine all result DataFrames
try:
    feature_res_df = pd.concat(feature_results, ignore_index=True)
except Exception as e:
    print(f"Error combining feature results: {e}")
    exit()

# Save results to a file
feature_results_file = os.path.join(output_dir, "number_features_vs_accuracy.txt")
save_results(feature_res_df, feature_results_file)

# Save detailed feature lists to a separate file
detailed_features_file = os.path.join(output_dir, "detailed_feature_list.txt")
try:
    feature_res_df[["number_features", "accuracy", "feature_list"]].to_csv(
        detailed_features_file, sep="\t", index=False
    )
    print(f"Detailed feature list saved to {detailed_features_file}")
except Exception as e:
    print(f"Error saving detailed feature list: {e}")

