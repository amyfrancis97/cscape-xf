#%%

import sys
sys.path.append('/Users/uw20204/Documents/scripts/CanDrivR/plots')
from violinPlot import *
from confusionMatrix import *

sys.path.append('/Users/uw20204/Documents/scripts/CanDrivR/data/train_data')
from cosmicGnomadDataLoader import *

sys.path.append('/Users/uw20204/Documents/scripts/CanDrivR/optimisation')
from sequencial_feature_selection import *
from selected_features import *

sys.path.append('/Users/uw20204/Documents/scripts/CanDrivR/models')
from prepare_training_data import prepare_data
from train_evaluate import train_and_evaluate_model
from metric_results_table import get_results_table
from train_classifier import train_classifier
from DeepFFN import *
from ANN import *

import pandas as pd
import importlib

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


metrics = test_model(test_variants, test_data, features)
metrics

#%%

acc_res = []
std_res = []
for i in importance_df[1].unique().tolist():
    features_to_filter = importance_df[importance_df[1] >= i][0].tolist()
    print(features_to_filter)
    actual_targets, predicted_targets, df, top_100_features, feature_importance, mean_accuracy, std, total_time, final_model, X_test_cols, results_df, metrics_test = xgb(df2, features_to_filter)
    acc_res.append(mean_accuracy)
    std_res.append(std)
#%%
importance_curve_2 = pd.DataFrame({"accuracy": acc_res, "std": std_res, "importance_threshold": importance_df[1].unique().tolist()})
#%%
data = importance_curve_2.sort_values("importance_threshold", ascending=False)
data
#%%
df = data
from scipy.stats import linregress
custom_palette = sns.color_palette("Set2", 8)
# Polynomial regression
coefficients = np.polyfit(df['importance_threshold'], df['accuracy'], 3)  # Adjust degree as needed
p = np.poly1d(coefficients)

# Plot
plt.errorbar(df['importance_threshold'], df['accuracy'], yerr=df['std'], fmt='o', label='Data')
plt.plot(df['importance_threshold'].values, p(df['importance_threshold']), color=custom_palette[1], label='Fitted polynomial')
plt.xlabel('Importance Threshold')
plt.xscale("log")
plt.ylabel('Accuracy')
plt.title('Accuracy vs Importance Threshold')
plt.legend()
plt.grid(True)
plt.savefig("/Users/uw20204/Documents/figures/" + "optimised_model_feature_selection.png", dpi=300)
plt.show()
#%%
data

#%%


#%%
def plot_confidence_scores(positive_df_62, negative_df_62, positive_df_all, negative_df_all, plot_both=True):
    custom_palette = sns.color_palette("Set2", 8)
    fig, ax = plt.subplots(2 if plot_both else 1, 1, figsize=(10, 8))  
    
    if plot_both:
        ax1, ax2 = ax
        
        ax1.scatter(positive_df_62['id'], positive_df_62['positive_confidence'], color=custom_palette[0], label='Positive Predictions')
        ax1.scatter(negative_df_62['id'], negative_df_62['negative_confidence'], color=custom_palette[1], label='Negative Predictions')
        ax1.set_title('Model Trained on Top 62 Features', fontsize=18)
        ax1.legend()

        ax2.scatter(positive_df_all['id'], positive_df_all['positive_confidence'], color=custom_palette[0], label='Positive Predictions')
        ax2.scatter(negative_df_all['id'], negative_df_all['negative_confidence'], color=custom_palette[1], label='Negative Predictions')
        ax2.set_title('Model Trained on all Features', fontsize=18)
        ax2.legend()
        
        for axi in ax:
            axi.tick_params(axis='x', rotation=90)
            axi.set_xlabel('Genomic Variants', fontsize=14)
            axi.set_ylabel('Confidence Scores', fontsize=14)
            axi.grid(True)
    else:
        ax.scatter(positive_df_all['id'], positive_df_all['positive_confidence'], color=custom_palette[0], label='Positive Predictions')
        ax.scatter(negative_df_all['id'], negative_df_all['negative_confidence'], color=custom_palette[1], label='Negative Predictions')
        ax.set_title('Model Trained on all Features', fontsize=18)
        ax.legend()
        ax.tick_params(axis='x', rotation=90)
        ax.set_xlabel('Genomic Variants', fontsize=14)
        ax.set_ylim(0.4, 1)
        ax.set_ylabel('Confidence Scores', fontsize=14)
        ax.grid(True)
        
    plt.tight_layout()
    plt.savefig("/Users/uw20204/Documents/figures/" + "gold_standards_confidence_scores_negatives.png", dpi=300)
    plt.show()

# Example usage
plot_confidence_scores(positive_df_62, negative_df_62, positive_df_all, negative_df_all, plot_both=False)
#%%
known_drivers
#%%
def get_mean_differences(known_drivers, feature):
    T_C = np.mean(known_drivers[["chr3_179234298_T/C" in i for i in known_drivers["id"]]][feature])
    #T_G = np.mean(known_drivers[["chr3_179234298_T/G" in i for i in known_drivers["id"]]][feature])
    #T_A = np.mean(known_drivers[["chr3_179234298_T/A" in i for i in known_drivers["id"]]][feature])
    close_proxim_well_predicted = np.mean(known_drivers[["chr3_179234298_T/C" not in i for i in known_drivers["id"]]][feature])
    return T_C, close_proxim_well_predicted

features = pd.DataFrame(feature_importance.items()).sort_values(1, ascending = False)[0].tolist()
res = [get_mean_differences(known_drivers, feature) for feature in features]
df_drivers = pd.DataFrame(res, columns = ["chr3:179234298_T/C","Mean Values of Other Gold Standard Variants"], index = features)
#%%

#%%
# Define a threshold for standard deviation.
threshold = 10  # Adjust value

# Filter rows where the standard deviation of the selected columns is below the threshold.
filtered_df = df_drivers[df_drivers.apply(lambda row: row[['chr3:179234298_T/C', "Mean Values of Other Gold Standard Variants"]].std(), axis=1) > threshold]
filtered_df

#%%
from sklearn.preprocessing import MinMaxScaler

# Initialise a MinMaxScaler
#scaler = MinMaxScaler()

# Select columns to normalise
#columns_to_normalise = ['chr3:179234298_T/C','Mean Values of Other Gold Standard Variants']

# Apply Min-Max normalisation on the selected columns
#filtered_df[columns_to_normalise] = scaler.fit_transform(filtered_df[columns_to_normalise])

# Plotting the normalised data with custom colors
#ax = filtered_df[columns_to_normalise].plot(kind='bar', figsize=(19, 19), width=1, color=custom_palette, edgecolor='black', linewidth=0.5)
ax = filtered_df.plot(kind='bar', figsize=(19, 19), width=1, color=custom_palette, edgecolor='black', linewidth=0.5)

ax.set_ylabel('Values')
ax.set_title('Comparison of Features')

# Place the legend above the plot on the left side (outside the plot area)
ax.legend(title='Variants', loc='upper left', ncol=1, frameon=False)

plt.tight_layout()

# Save figure
plt.savefig("/Users/uw20204/Documents/figures/" + "gold_standards_top_standard_deviations.png", dpi=300)

# Show the plot
plt.show()

# %%
