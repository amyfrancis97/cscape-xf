import os
import pandas as pd
import numpy as np
import time
from xgboost import XGBClassifier
from models.train_classifier import train_classifier
from models.metric_results_table import get_results_table
from plots.plot_sample_features_vs_metrics import plot_sample_metrics_smooth_lowess

# Define file paths
data_path = "/user/home/uw20204/CanDrivR_data/cscape-xf/data/all_features_cosmic_gnomad.bed.gz"
output_dir = "/user/home/uw20204/CanDrivR_data/cscape-xf/outputs"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

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

# Define the label column
labels = df['driver_stat']

# Define sample sizes to test
sample_sizes = [500, 1000, 5000, 10000, 20000, 40000, 80000]
results_list = []

# Optimise for sample sizes
for sample_size in sample_sizes:
    print(f"Processing sample size: {sample_size}")
    try:
        # Subsample the data
        df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

        # Train the classifier
        start_time = time.time()
        _, _, _, results = train_classifier(df_sample, classifier=XGBClassifier(random_state=42))
        elapsed_time = time.time() - start_time

        # Get results table
        res = get_results_table(results[0], model_name="XGB")
        res["elapsed_time (s)"] = elapsed_time
        res["number_samples"] = sample_size
        print(res)

        # Append results
        results_list.append(res)
    except Exception as e:
        print(f"Error with sample size {sample_size}: {e}")

# Combine results into a DataFrame
results_df = pd.concat(results_list, ignore_index=True)

# Save results to a file
results_file = os.path.join(output_dir, "sample_vs_accuracy_processed.txt")
results_df.to_csv(results_file, sep="\t", index=False)
print(f"Results saved to {results_file}")

# Plot results
plot_sample_metrics_smooth_lowess(results_df, output_dir, "number_samples", "Sample Size")
print(f"Plots saved to {output_dir}")

