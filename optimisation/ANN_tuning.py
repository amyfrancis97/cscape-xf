#%%
import sys
import optuna
sys.path.append('/Users/uw20204/Documents/scripts/CanDrivR/models')
from prepare_training_data import prepare_data
from train_evaluate import train_and_evaluate_model
from metric_results_table import get_results_table
from train_classifier import train_classifier
from ANN import *
sys.path.append('/Users/uw20204/Documents/scripts/CanDrivR/optimisation')
from selected_features import *
from params import *
from typing import *
import pandas as pd
import numpy as np


def objective(trial):
    df = pd.read_csv("/Volumes/Samsung_T5/data/sample_cosmic_gnomad43000.txt", sep = "\t")
    df = df.drop(df.columns[df.isna().any()].tolist(), axis = 1) # Remove any columns that are na
    df = df.sample(5000)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    n_layers = trial.suggest_int('n_layers', 1, 5)
    hidden_sizes = [trial.suggest_int('hidden_size_l{}'.format(i), 32, 1024) for i in range(n_layers)]
    hyperparameters = {'lr': lr, 'hidden_sizes': hidden_sizes}
    results, total_time, model = ANN_model(df, hyperparameters=hyperparameters, features = features)
    mean_accuracy = results['accuracy'].mean()
    return mean_accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100) 
    best_params = study.best_trial.params
    print("Best hyperparameters:", best_params)
    import json

    # Convert the best_params dict into a JSON string and save it to a file
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f)

    print("Best hyperparameters have been saved to 'best_hyperparameters.json'.")

# %%
