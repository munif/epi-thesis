import os

import pandas as pd
import polars as pl
import itertools
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc, precision_recall_fscore_support, confusion_matrix, classification_report

WORKING_DIR = '/group/pmc021/amunif/epi-thesis/workflow/09_Learning_to_Rank'
DATASET_DIR = '/group/pmc021/amunif/epi-thesis/workflow/08_HepG2'

sample_size = 1_000_000

def split_data(X, y):
    # Split the dataset into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.666, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    return X_train, X_val, X_test, y_train, y_val, y_test


# Class Dataset for HepG2
class HepG2Dataset(Dataset):
    def __init__(self, X, y, features):
        self.X = X
        self.y = y
        self.features = features

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        feature_1 = self.features[self.X[idx, 0], 2]
        feature_2 = self.features[self.X[idx, 1], 2]
        feature = np.concatenate((feature_1, feature_2), axis=0)
        return feature, self.y[idx]
    
class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden1_size=64, hidden2_size=32, output_size=1):
        super(BinaryClassifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden1_size),
            nn.ReLU(),
            nn.Linear(hidden1_size, hidden2_size),
            nn.ReLU(),
            nn.Linear(hidden2_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.squeeze(1)
        return self.network(x)

    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
            return (outputs >= 0.5).long()

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    return device

device = get_device()

# Load dataset
gene_pl = pl.read_parquet(os.path.join(DATASET_DIR, 'dataset', 'gene_w_label_value_1.parquet'))

# Combine the features into single column
marker_list = ['H3K4me3', 'H3K9ac', 'H3K9me3', 'H3K27ac', 'H3K27me3']
gene_features = gene_pl.with_columns(
    pl.concat_list(marker_list).alias("combined_features")
)

# Create data index
gene_features = gene_features.with_row_index("index")

# Select only the index, gene_id and the value_1
gene_id_val = gene_features.select(["index", "gene_id", "value_1"])

# Doing the permutation using cross join
permuted_df = gene_id_val.join(gene_id_val, how="cross")

# Create the label
permuted_w_label_df = permuted_df.with_columns(
    pl.when(pl.col("value_1") > pl.col("value_1_right")).then(1)
      .otherwise(0)
      .alias("label")
)

# Prepare the X and y
X_df = permuted_w_label_df.select(['index', 'index_right'])
y_df = permuted_w_label_df.select(['label'])

# Select gene_id and its features
gene_id_features_df = gene_features.select(["index", "gene_id", "combined_features"])

# Convert data into Numpy Array
X_np = X_df.to_numpy()
y_np = y_df.to_numpy()
features_np = gene_id_features_df.to_numpy()

# Prepare the data sample
X_sample = X_np[:sample_size]
y_sample = y_np[:sample_size]

# Split the dataset
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_sample, y_sample)

# Prepare the data loader
dataset = HepG2Dataset(X_train, y_train, features_np)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True, pin_memory=True)

# Data loader for validation
val_dataset = HepG2Dataset(X_val, y_val, features_np)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True, pin_memory=True)

# Initialize model
model = BinaryClassifier(input_size=40000).to(device)

# Define loss function for binary classification
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    # Manual batch generation
    for X_batch, y_batch in dataloader:        
        # Forward pass
        outputs = model(X_batch.to(dtype=torch.float32).to(device))
        loss = criterion(outputs, y_batch.to(dtype=torch.float32).to(device))
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        predictions = (outputs >= 0.5).long()
        correct += (predictions == y_batch.to(dtype=torch.float32).to(device)).sum().item()
        total += y_batch.size(0)
        total_loss += loss.item()
    
    # Print epoch statistics
    avg_loss = total_loss / len(X_train)
    accuracy = correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

    # Validation step
    model.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Forward pass
            outputs = model(inputs.to(dtype=torch.float32).to(device))

            # Compute validation loss
            val_loss = criterion(outputs, labels.to(dtype=torch.float32).to(device))
            total_val_loss += val_loss.item()

            