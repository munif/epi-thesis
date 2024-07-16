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

# Prepare the progress file
current_time = datetime.now().strftime("%Y%m%d%H%M%S")
progress_file = f'progress_conv1d_tensorboard_{current_time}.txt'

'''
User defined functions
'''
def save_progress(file_name, message):
    with open(file_name, 'a+') as file:
        file.write(message + "\n")


print("Loading the data")
save_progress(progress_file, "Loading the data")
# Load the data
# X = pl.read_csv("histone_features.csv", n_rows=100)
# y = pl.read_csv("value_1_df.csv", n_rows=100)
X = pl.read_csv("histone_features.csv")
y = pl.read_csv("value_1_df.csv")
save_progress(progress_file, "Finished load the data")
print("Finished load the data")

# Convert to numpy
X_np = X.to_numpy()
y_np = y.to_numpy()

# Split the data into training and testing
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

# Setup the CUDA devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_progress(progress_file, f'Using device: {device}')

# Convert NumPy arrays to PyTorch tensors and then send it to CUDA
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).unsqueeze(1).to(device)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).unsqueeze(1).to(device)
y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32).to(device)

# Create TensorDataset for training and testing sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create a DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create CNN1D class
class HighDimCNN1D(nn.Module):
    def __init__(self, input_size):
        super(HighDimCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after the convolution and pooling layers
        conv_output_size = input_size // 8  # Adjust this based on the number of pooling layers
        
        self.fc1 = nn.Linear(256 * conv_output_size, 512)
        self.fc2 = nn.Linear(512, 1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.conv1(x)))
        x = self.pool2(self.leaky_relu(self.conv2(x)))
        x = self.pool3(self.leaky_relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

'''
Training
'''
input_size = 20000  # Number of features in the input
model = HighDimCNN1D(input_size=input_size).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Setup TensorBoard writer
LOG_DIR = "runs/cnn1d_model"
writer = SummaryWriter(LOG_DIR)

# Track the best test loss
best_test_loss = float('inf')

# Training loop
num_epochs = 200
for epoch in range(num_epochs): 
    print(f"Epoch {epoch + 1} ...")
    start_time = time.time()
    
    model.train()
    train_loss = 0.0
    train_mae = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        mae = mean_absolute_error(targets.cpu().numpy(), outputs.cpu().detach().numpy())
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        train_mae += mae * inputs.size(0)
    
    train_loss /= len(train_loader.dataset)
    train_mae /= len(train_loader.dataset)

    # Test step
    model.eval()
    test_loss = 0.0
    test_mae = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            mae = mean_absolute_error(targets.cpu().numpy(), outputs.cpu().numpy())
            test_loss += loss.item() * inputs.size(0)
            test_mae += mae * inputs.size(0)

    test_loss /= len(test_loader.dataset)
    test_mae /= len(test_loader.dataset)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    message = f'Epoch {epoch+1}, Time:{elapsed_time},  Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}'
    print(message)
    save_progress(progress_file, message)

    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Test', test_loss, epoch)
    writer.add_scalar('MAE/Train', train_mae, epoch)
    writer.add_scalar('MAE/Test', test_mae, epoch)

    # Save the best model
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), 'best_conv1d_model.pth')
        print(f'Saved best model at epoch {epoch+1} with test loss: {test_loss:.4f}')

writer.close()

# Load the best model's state dictionary
model.load_state_dict(torch.load('best_conv1d_model.pth'))

'''
Evaluation
'''
# Evaluation
model.eval()
total_loss = 0.0
all_targets = []
all_predictions = []

# Evaluate the model
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(outputs.cpu().numpy())

all_targets = np.array(all_targets)
all_predictions = np.array(all_predictions)

mse = mean_squared_error(all_targets, all_predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(all_targets, all_predictions)
r2 = r2_score(all_targets, all_predictions)

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
    'y_test': all_targets.flatten(),
    'y_pred': all_predictions.flatten()
})

# Save the DataFrame to a CSV file
results_df.to_csv(f'predictions_conv1d_tensorboard_{current_time}.csv', index=False)

print(f"Predictions and true values saved to predictions_conv1d_tensorboard_{current_time}.csv.")