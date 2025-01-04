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

def plot_probability_scores(true_df, false_df, title, correct_predictions, incorrect_predictions, negatives = True):
    # Professional color palette
    custom_palette = sns.color_palette("Dark2", 4)  
    
    # Set plot style for better readability
    sns.set(style="whitegrid")
    
    fig, ax = plt.subplots(1, figsize=(10, 6), sharex=True)  
    
    if negatives:
        ax.scatter(true_df['id'], true_df['confidence_scores'], color=custom_palette[1], label=correct_predictions)

        ax.scatter(false_df['id'], false_df['confidence_scores'], color=custom_palette[0], label=incorrect_predictions)

    else:
        ax.scatter(true_df['id'], true_df['confidence_scores'], color=custom_palette[0], label=correct_predictions)

        ax.scatter(false_df['id'], false_df['confidence_scores'], color=custom_palette[1], label=incorrect_predictions)


    ax.set_title(title, fontsize=18)
    ax.legend(fontsize=12, loc='best')
    ax.tick_params(axis='x', rotation=90, labelsize=12)
    ax.set_xlabel('Genomic Variants', fontsize=14)
    ax.set_ylabel('Probability Scores', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig("/Users/uw20204/Documents/figures/" + title.replace(" ", "_") + ".png", dpi=300, bbox_inches='tight')
    plt.show()
# %%
