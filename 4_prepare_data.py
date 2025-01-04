import os
import pandas as pd
import random

def process_data(res_dir, file_path, processed_data_path, chunk_size=10000):
    """
    Reads and processes the dataset by filtering, sampling, and handling missing values.

    Parameters:
        res_dir (str): Directory containing the feature vs accuracy file.
        file_path (str): Path to the dataset file.
        processed_data_path (str): Path to save the processed data.
        chunk_size (int): Number of rows to read per chunk for large datasets.
    """
    # Read feature vs accuracy file
    features_vs_acc_path = os.path.join(res_dir, "number_features_vs_accuracy.txt")
    features_vs_acc = pd.read_csv(features_vs_acc_path, sep="\t")
    print(features_vs_acc.head())

    # Select the best number of features and corresponding feature list
    best_num_features = features_vs_acc.sort_values('precision', ascending=False).reset_index(drop=True)['number_features'][0]
    features_to_keep = features_vs_acc.sort_values('precision', ascending=False).reset_index(drop=True)['feature_list'][0].split(',')

    # Read and process data in chunks
    chunks = []
    try:
        for chunk in pd.read_csv(file_path, sep="\t", chunksize=chunk_size):
            chunks.append(chunk)
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    # Combine all chunks into a single DataFrame
    df = pd.concat(chunks, ignore_index=True)
    print(df.head())

    # Filter and sample data
    random.seed(42)
    df = pd.concat([
        df[df['driver_stat'] == 0].sample(20000, random_state=42),
        df[df['driver_stat'] == 1].sample(20000, random_state=42)
    ]).reset_index(drop=True)
    print(df.head())
    print(features_to_keep)

    # Keep only selected features and grouping
    df = df[df.columns[:5].tolist() + features_to_keep + ['grouping']]

    # Replace discrete/binary columns with mode and continuous columns with mean
    binary_columns = [col for col in df.columns if df[col].nunique() == 2]  # Binary columns
    continuous_columns = [col for col in df.columns if df[col].dtype != 'object' and df[col].nunique() > 2]  # Continuous columns

    # Replace binary columns with mode
    for col in binary_columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Replace continuous columns with mean
    df[continuous_columns] = df[continuous_columns].fillna(df[continuous_columns].mean())

    # Save the processed DataFrame
    df.to_csv(processed_data_path, index=False)
    print(f"Processed DataFrame saved to {processed_data_path}")

# Define file paths
RES_DIR = "/user/home/uw20204/CanDrivR_data/cscape-xf/outputs"
FILE_PATH = "/user/home/uw20204/CanDrivR_data/cscape-xf/data/all_features_cosmic_gnomad.bed.gz"
PROCESSED_DATA_PATH = "/user/home/uw20204/CanDrivR_data/cscape-xf/data/processed_data.csv"

# Process the data
process_data(RES_DIR, FILE_PATH, PROCESSED_DATA_PATH)

