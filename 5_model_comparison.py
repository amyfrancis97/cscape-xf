import pandas as pd
from models.DeepFFN import *
from models.ANN import *
from models.train_classifier import train_classifier
from models.metric_results_table import get_results_table
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Define paths
processed_data_path = "/user/home/uw20204/CanDrivR_data/cscape-xf/outputs/processed_data.csv"
output_comparison_path = "/user/home/uw20204/CanDrivR_data/cscape-xf/outputs/model_comparison_res.txt"

# Load the processed DataFrame
df = pd.read_csv(processed_data_path)
print("Loaded processed DataFrame:")
print(df.head())

# Initialise results list
results_list = []

# Run XGBoost model
actual_predicted_targets, feature_importance, final_model, results = train_classifier(df, classifier=XGBClassifier(random_state=42))
XGB_results_metrics = get_results_table(results[0], model_name="XGB")
XGB_results_metrics["time (s)"] = round(float(results[1]["time"][0]), 2)
results_list.append(XGB_results_metrics)

# Run SVM model
actual_predicted_targets, feature_importance, final_model, results = train_classifier(df, classifier=SVC(random_state=42, probability=True))
SVM_results_metrics = get_results_table(results[0], model_name="SVM")
SVM_results_metrics["time (s)"] = round(float(results[1]["time"][0]), 2)
results_list.append(SVM_results_metrics)

# Run Random Forest model
actual_predicted_targets, feature_importance, final_model, results = train_classifier(df, classifier=RandomForestClassifier(random_state=42))
RF_results_metrics = get_results_table(results[0], model_name="RF")
RF_results_metrics["time (s)"] = round(float(results[1]["time"][0]), 2)
results_list.append(RF_results_metrics)

# Run DeepFFN model
results, total_time, model = DeepFFN_model(df)
DFFN_results_metrics = get_results_table(results=results, model_name="DFFN")
DFFN_results_metrics["time (s)"] = round(float(total_time), 2)
results_list.append(DFFN_results_metrics)

# Run ANN model
results, total_time, model = ANN_model(df)
ANN_results_metrics = get_results_table(results=results, model_name="ANN")
ANN_results_metrics["time (s)"] = round(float(total_time), 2)
results_list.append(ANN_results_metrics)

# Combine results and save to file
model_comparison = pd.concat(results_list)
model_comparison.insert(0, "models", model_comparison["model"])
model_comparison = model_comparison.drop("model", axis=1)
model_comparison.to_csv(output_comparison_path, sep="\t", index=False)

print(f"Model comparison results saved to {output_comparison_path}")

