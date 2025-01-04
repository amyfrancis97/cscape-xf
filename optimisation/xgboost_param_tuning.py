#%%
from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.svm import OneClassSVM
from collections import Counter
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from numpy import asarray
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import sys 
import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

# plot feature importance using built-in function
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot

import scikitplot as skplt
import seaborn as sns

import importlib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import LeaveOneGroupOut
import numpy as np
from matplotlib import pyplot

from itertools import product

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.append('/Users/uw20204/Documents/scripts/CanDrivR/models')
from prepare_training_data import prepare_data
from train_evaluate import train_and_evaluate_model
import time
sys.path.append('/Users/uw20204/Documents/scripts/CanDrivR/optimisation')
from params import *
from typing import *
from sequencial_feature_selection import *
from selected_features import *

import time

def xgb_grid_search_cv(df, features=None, classifier=XGBClassifier(random_state=42)):
    if features is None:
        X = df.drop(["chrom", "pos", "ref_allele", "alt_allele", "driver_stat", "grouping"], axis=1)
    else:
        X = df[features]
    
    y = df["driver_stat"]
    groups = df["grouping"]

    # Split data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the parameter grid
    param_grid = {
        'max_depth': [5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2],
        'scale_pos_weight': [1, 2, 3],
        'reg_lambda': [0.1, 0.5, 1.0],  # Note: 'lambda' is a reserved keyword in Python, use 'reg_lambda' instead
        'reg_alpha': [0, 0.1, 0.5],
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, 
                               scoring='accuracy', n_jobs=-1, cv=5, verbose=2)
    
    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Best parameter set
    print(f"Best parameters found: {grid_search.best_params_}")

    # Best model
    best_model = grid_search.best_estimator_

    # Predict and evaluate on the test set
    y_test_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test set accuracy: {accuracy}")

    best_params = grid_search.best_params_

    # Convert best parameters to a DataFrame for easy CSV export
    best_params_df = pd.DataFrame([best_params])

    # Save to CSV
    best_params_df.to_csv('best_model_params.csv', index=False)

    # You can also return the GridSearchCV object, best_model, and any other metrics or models you need
    return grid_search, best_model, accuracy

if __name__ == "__main__":
    df2 = pd.read_csv("/user/home/uw20204/scratch/sample_cosmic_gnomad14000.txt", sep = "\t")
    df2 = df2.drop(df2.columns[df2.isna().any()].tolist(), axis = 1)
    xgb_grid_search_cv(df2.sample(2000), features = features)
