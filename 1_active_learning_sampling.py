import os
import pandas as pd
import numpy as np
import time
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from models.train_classifier import train_classifier
from models.metric_results_table import get_results_table
#from plots.plot_sample_features_vs_metrics import plot_sample_metrics_smooth_lowess
import random

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

# Define label column and initial training and pool sets
labels = df['driver_stat']
initial_size = 500
query_size = 500
max_iterations = 50

# Split the data into initial training, validation, and query pool
# Set a random seed for reproducibility
random.seed(42)

# Get unique chromosome identifiers
unique_chroms = df['chrom'].unique()

# Randomly select three chromosomes for validation
validation_chroms = random.sample(list(unique_chroms), 3)
print(f"Selected chromosomes for validation: {validation_chroms}")

# Split the dataset
validation_set = df[df['chrom'].isin(validation_chroms)].reset_index(drop=True)
training_pool = df[~df['chrom'].isin(validation_chroms)].reset_index(drop=True)

train_set = training_pool.sample(n=initial_size, random_state=42).reset_index(drop=True)
query_pool = remaining_data.drop(train_set.index).reset_index(drop=True)
results_list = []

# Active learning loop
for iteration in range(max_iterations):
    print(f"Active learning iteration {iteration + 1}/{max_iterations}")
    
    # Train the classifier on the current training set
    start_time = time.time()
    _, _, model, results = train_classifier(train_set, classifier=XGBClassifier(random_state=42))
    elapsed_time = time.time() - start_time

    # Evaluate performance on the validation set
    y_val_true = validation_set['driver_stat']
    y_val_pred = model[0].predict(validation_set[model[1]])
    accuracy = accuracy_score(y_val_true, y_val_pred)
    f1 = f1_score(y_val_true, y_val_pred, zero_division=1)

    print(f"Iteration {iteration + 1}: Validation Accuracy = {accuracy:.4f}, F1 Score = {f1:.4f}")

    # Save performance results
    results_table = get_results_table(results[0], model_name="XGB")
    results_table["elapsed_time (s)"] = elapsed_time
    results_table["training_size"] = len(train_set)
    results_table["validation_accuracy"] = accuracy
    results_table["validation_f1"] = f1
    results_list.append(results_table)

    # Stop if the query pool is empty
    if query_pool.empty:
        print("Query pool is empty. Stopping active learning.")
        break

    # Query the most uncertain samples from the pool
    try:
        # Ensure query pool has the same features as the training set
        query_features = model[1]  # Feature list used during training
        query_data = query_pool[query_features].copy()

        # Handle missing columns in the query pool
        missing_columns = [col for col in query_features if col not in query_data.columns]
        for col in missing_columns:
            query_data[col] = 0  # Fill missing columns with default values

        # Predict probabilities and calculate uncertainty
        probas = model[0].predict_proba(query_data)[:, 1]
        uncertainty = np.abs(probas - 0.5)  # Measure uncertainty
        query_indices = np.argsort(uncertainty)[:query_size]  # Select most uncertain samples

        # Select queried samples
        queried_samples = query_pool.iloc[query_indices]

        # Add queried samples to the training set
        train_set = pd.concat([train_set, queried_samples], ignore_index=True)

        # Remove queried samples from the pool
        query_pool = query_pool.drop(query_indices).reset_index(drop=True)

    except ValueError as e:
        print(f"Error during querying: {e}")
        break

# Combine results into a DataFrame
results_df = pd.concat(results_list, ignore_index=True)

# Save results to a file
results_file = os.path.join(output_dir, "active_learning_results.txt")
results_df.to_csv(results_file, sep="\t", index=False)
print(f"Active learning results saved to {results_file}")

# Plot results
#plot_sample_metrics_smooth_lowess(results_df, output_dir, "training_size", "Training Size")
#print(f"Plots saved to {output_dir}")

