import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import time

# Define the neural network architecture
class DeepFFN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DeepFFN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.5))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def DeepFFN_model(df, features=None):
    if features is None:
        X = df.drop(["chrom", "pos", "ref_allele", "alt_allele", "driver_stat", "grouping"], axis=1)
    elif isinstance(features, list):
        X = df[features]
    else:
        X = df[features]
        X = np.array(X)  # Ensure X is a numpy array
        X = X.reshape(-1, 1)

    y = df["driver_stat"]
    groups = df["grouping"]
    
    # Convert X, y, and groups to DataFrames
    X = pd.DataFrame(X, columns=X.columns)
    y = pd.Series(y, name="driver_stat")
    groups = pd.Series(groups, name="grouping")
    
    # Split data into train/validation set and test set
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=42)

    logo = LeaveOneGroupOut()
    
    predicted_targets = np.array([])
    actual_targets = np.array([])
    accuracy_mean = []

    # GridSearch for hyperparameter tuning
    input_size = X_train.shape[1]
    hidden_sizes = [512, 256, 128]  # List of hidden layer sizes
    output_size = 1

    start = time.time()
    
    validation_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': []
    }
    
    # LOGO cross-validation for model training
    for train_index, val_index in logo.split(X_train, y_train, groups_train):
        X_train_split, X_val_split = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_split, y_val_split = y_train.iloc[train_index], y_train.iloc[val_index]

        # Random undersampling to balance the training data
        sampler = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_split, y_train_split)

        # Standardize features
        scaler = StandardScaler()
        X_train_resampled = scaler.fit_transform(X_train_resampled)
        X_val_split = scaler.transform(X_val_split)
        X_test_scaled = scaler.transform(X_test)

        model = DeepFFN(input_size, hidden_sizes, output_size)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Convert to PyTorch tensors
        X_train_resampled = torch.FloatTensor(X_train_resampled)
        y_train_resampled = torch.FloatTensor(y_train_resampled.values).view(-1, 1)
        X_val_split = torch.FloatTensor(X_val_split)
        y_val_split = torch.FloatTensor(y_val_split.values).view(-1, 1)

        # Training the model
        epochs = 50
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_resampled)
            loss = criterion(outputs, y_train_resampled)
            loss.backward()
            optimizer.step()

        # Evaluate the model on validation set
        with torch.no_grad():
            model.eval()
            y_val_pred = model(X_val_split)
            y_val_pred = (y_val_pred > 0.5).float()

        # Convert predictions to numpy arrays
        y_val_pred = y_val_pred.numpy()
        y_val_true = y_val_split.numpy()

        # Calculate evaluation metrics for validation set
        accuracy = accuracy_score(y_val_true, y_val_pred)
        accuracy_mean.append(accuracy)
        
        validation_metrics['accuracy'].append(accuracy)
        validation_metrics['precision'].append(precision_score(y_val_true, y_val_pred))
        validation_metrics['recall'].append(recall_score(y_val_true, y_val_pred))
        validation_metrics['f1'].append(f1_score(y_val_true, y_val_pred))
        validation_metrics['roc_auc'].append(roc_auc_score(y_val_true, y_val_pred))

    end = time.time()
    total_time = end - start
    results = pd.DataFrame(validation_metrics)

    return results, total_time, model

