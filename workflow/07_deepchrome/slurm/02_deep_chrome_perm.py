# DeepChrome with histone permutation

# Import required library
import os
import sys

import polars as pl
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc

DATASET_PATH = '/group/pmc021/amunif/epi-thesis/workflow/07_deepchrome/dataset/E066/'

start = int(sys.argv[1])
end = int(sys.argv[2])

# Check whether CUDA is available
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# DeepClassifier class
class DeepClassifier(nn.Module):
    def __init__(self, input_dim = [5, 100], out_channels = 50, conv_kernel_size = 10, max_pooling_kernel_size = 5):
        super(DeepClassifier, self).__init__()
        
        flatten_dim = out_channels * ((input_dim[1] - conv_kernel_size + 1) // max_pooling_kernel_size)

        self.conv = nn.Conv2d(1, out_channels, kernel_size=(input_dim[0], conv_kernel_size))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((1, max_pooling_kernel_size))
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(flatten_dim, 625)
        self.linear2 = nn.Linear(625, 125)
        self.linear3 = nn.Linear(125, 2)
        self.softmax = nn.LogSoftmax(dim=1)
        

    def forward(self, x):
        # Stage 1: filter bank -> squashing -> max pooling
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)

        # Stage 2: standar 2-layer neural network        
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x

# Function to return min, avg, and max
def min_avg_max(lst):
    return round(min(lst), 2), round(sum(lst)/len(lst), 2), round(max(lst), 2)

# Load dataset
print("Loading the data ...")
E066_pl = pl.read_parquet(os.path.join(DATASET_PATH, 'E066_w_histone_pl.parquet'))

# Load permutation list
perm_pl = pl.read_parquet(os.path.join(DATASET_PATH, 'permutation.parquet'))
perm_lst = perm_pl['markers_perm'].to_list()

output_rows = []

for item in perm_lst[start:end]:
    # Generate histone item name
    item_name = '-'.join(item)

    print(f"Item: {item_name}")

    # Create dataframe based on histone marker permutation item
    perm_histone = E066_pl.with_columns(
        pl.struct(item).map_elements(
            lambda x: [x[col_name] for col_name in item],
            return_dtype = pl.List(pl.List(pl.Float64))
        )
        .alias('histone')
    )

    # Select X and y column
    X = perm_histone.select(pl.col('histone')).to_series().to_list()
    y = perm_histone.select(pl.col('label')).to_series().to_list()

    # Convert to Numpy array
    X = np.array(X)
    y = np.array(y)

    # Split the dataset into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.666, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Convert numpy arrays to PyTorch tensors
    X_train = torch.from_numpy(X_train).float().unsqueeze(1).to(device)
    X_val = torch.from_numpy(X_val).float().unsqueeze(1).to(device)
    X_test = torch.from_numpy(X_test).float().unsqueeze(1).to(device)

    y_train = torch.from_numpy(y_train).float().to(device)
    y_val = torch.from_numpy(y_val).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    # Create DataLoaders
    batch_size = 32

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)


    # Define Deep Learning model instannce
    model = DeepClassifier(input_dim = [len(item), 100]).to(device)
    criterion = nn.NLLLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    num_epochs = 100

    # For tracking train and validation
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_aucs, val_aucs = [], []

    # For saving the best model
    best_val_metric = float('-inf')
    best_model_path = os.path.join(DATASET_PATH, "experiments", "model", f'{item_name}.pth')

    for epoch in range(num_epochs):
        # Training phase
        model.train()  # Set the model to training mode
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_true_labels = []
        train_predicted_probs = []
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.cuda(), labels.type(torch.LongTensor).cuda())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_true_labels.extend(labels.cpu().numpy())
            train_predicted_probs.extend(predicted.cpu().numpy())
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_auc_score = roc_auc_score(train_true_labels, train_predicted_probs)
        
        train_losses.append(round(avg_train_loss, 4))
        train_accuracies.append(round(train_accuracy, 2))
        train_aucs.append(round(train_auc_score, 2))
        
        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_metric = 0
        val_true_labels = []
        val_predicted_probs = []
        
        # Training and Validation
        with torch.no_grad():  # Disable gradient computation
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs.cuda(), labels.type(torch.LongTensor).cuda())
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_true_labels.extend(labels.cpu().numpy())
                val_predicted_probs.extend(predicted.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            val_metric = val_correct / len(val_loader)

            # Save the best model
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                torch.save(model.state_dict(), best_model_path)
                # print(f"New best model saved with validation metric: {best_val_metric:.4f}")
            
            val_true_labels = np.array(val_true_labels)
            val_predicted_probs = np.array(val_predicted_probs)
            val_auc_score = roc_auc_score(val_true_labels, val_predicted_probs)

            val_losses.append(round(avg_val_loss, 4))
            val_accuracies.append(round(val_accuracy, 2))
            val_aucs.append(round(val_auc_score, 2))
            
            if ((epoch + 1) % 10 == 0):    
                print(f'Epoch [{epoch+1}/{num_epochs}] - '
                    f'[TRAIN] Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%, AUC: {train_auc_score:.2f} - '
                    f'[VAL] Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%, AUC: {val_auc_score:.2f}')
    
    print("Training finished!")

    # Load the best model
    best_model = DeepClassifier(input_dim = [len(item), 100]).to(device)
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.eval()

    # Predict on test set
    test_predictions = []
    test_true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = best_model(inputs)
            # probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())

    # Calculate evaluation metric
    test_acc_score = round(accuracy_score(test_true_labels, test_predictions) * 100, 2)
    print(f"Accuracy Score on test set: {test_acc_score:.2f} %")
    test_auc_score = round(roc_auc_score(test_true_labels, test_predictions), 2)
    print(f"AUC Score on test set: {test_auc_score:.2f}\n")

    # Create dataframe from array
    experiment_results = {
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'train_auc': train_aucs,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies,
        'val_auc': val_aucs
    }

    # Saving the experiments into CSV files
    experiment_df = pl.DataFrame(experiment_results)
    experiment_df.write_csv(os.path.join(DATASET_PATH, 'experiments', f'{item_name}.csv'))

    # Find the min, avg, max for training and validation step
    train_loss_min, train_loss_avg, train_loss_max = min_avg_max(train_losses)
    train_acc_min, train_acc_avg, train_acc_max = min_avg_max(train_accuracies)
    train_auc_min, train_auc_avg, train_auc_max = min_avg_max(train_aucs)

    val_loss_min, val_loss_avg, val_loss_max = min_avg_max(val_losses)
    val_acc_min, val_acc_avg, val_acc_max = min_avg_max(val_accuracies)
    val_auc_min, val_auc_avg, val_auc_max = min_avg_max(val_aucs)

    output_rows.append([
        item_name, 
        train_loss_min, train_loss_avg, train_loss_max,
        train_acc_min, train_acc_avg, train_acc_max,
        train_auc_min, train_auc_avg, train_auc_max,
        val_loss_min, val_loss_avg, val_loss_max,
        val_acc_min, val_acc_avg, val_acc_max,
        val_auc_min, val_auc_avg, val_auc_max,
    ])


# Saving current batch results
output_rows_df = pd.DataFrame(output_rows, 
    columns=[
        "item_name",
        "train_loss_min", "train_loss_avg", "train_loss_max",
        "train_acc_min", "train_acc_avg", "train_acc_max",
        "train_auc_min", "train_auc_avg", "train_auc_max",
        "val_loss_min", "val_loss_avg", "val_loss_max",
        "val_acc_min", "val_acc_avg", "val_acc_max",
        "val_auc_min", "val_auc_avg", "val_auc_max",
    ]
)

print("Saving the final results")
output_rows_df.to_csv(os.path.join(DATASET_PATH, "experiments/min-avg-max", 
                              f"{str(start + 1).rjust(3, '0')}-{str(end).rjust(3, '0')}.csv"), 
                 header=True, index=False)

print("Batch FINISHED!!!")