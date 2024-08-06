# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import pandas as pd
import numpy as np
import polars as pl

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from datetime import datetime
import time

from torch.utils.tensorboard import SummaryWriter
import os


dataset_path = "/group/pmc021/amunif/epi-thesis/dataset/"
working_dir = "/group/pmc021/amunif/epi-thesis/workflow/06_neural_network_gpu/"

current_time = datetime.now().strftime("%Y%m%d%H%M%S")
progress_file = f'{working_dir}progress/progress_nn_train_{current_time}.txt'
best_model_file = 'nn_best_model.pth'

def save_progress(file_name, message):
    with open(file_name, 'a+') as file:
        file.write(message + "\n")

# Loading file using polars
print('Loading data ...')
X = pl.read_csv(f"{dataset_path}histone_features.csv")
y = pl.read_csv(f"{dataset_path}value_1_df.csv")
print('Finished loading data!')

# Convert to numpy
X_np = X.to_numpy()
y_np = y.to_numpy()

# Split the data into training and testing
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

# Check the GPU support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_progress(progress_file, f'Using device: {device}')

# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32).to(device)

# Create TensorDataset for training and testing sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create a DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 1024)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, 512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(512, 256)
        self.dropout4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(256, 1)  # Single output for regression
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.leaky_relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.leaky_relu(self.fc4(x))
        x = self.dropout4(x)
        x = self.fc5(x)  #
        return x
    
input_dim = 20000
model = NeuralNetwork(input_dim).to(device)
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Load the best model
model.load_state_dict(torch.load(best_model_file))
print(f"Loaded the best model from {best_model_file}")

# Making predictions with the test data
with torch.no_grad():
    predictions = model(X_train_tensor)

# Convert predictions to NumPy array
predictions_np = predictions.cpu().numpy()
y_train_np = y_train_tensor.cpu().numpy()

# Calculate evaluation metrics
mse = mean_squared_error(y_train_np, predictions_np)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_train_np, predictions_np)
r2 = r2_score(y_train_np, predictions_np)

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R2 Score: {r2}')

save_progress(progress_file, f'MSE: {mse}')
save_progress(progress_file, f'RMSE: {rmse}')
save_progress(progress_file, f'MAE: {mae}')
save_progress(progress_file, f'R2 Score: {r2}')

# Create a DataFrame with true values and predictions
results_df = pd.DataFrame({
    'y_train': y_train_np.flatten(),
    'y_pred': predictions_np.flatten()
})

# Save the DataFrame to a CSV file
results_df.to_csv(f"{working_dir}predictions/nn_train_predictions_{current_time}.csv", index=False)
print(f"Predictions and true values saved to 'nn_train_predictions_{current_time}.csv'.")