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


# dataset_path = "/group/pmc021/amunif/epi-thesis/dataset/"

current_time = datetime.now().strftime("%Y%m%d%H%M%S")
# progress_file = f"/group/pmc021/amunif/epi-thesis/scripts/shell/slurm/workflow/output/09/v2/progress_{current_time}.txt"
progress_file = f'progress_nn_{current_time}.txt'
best_model_file = 'nn_best_model.pth'

def save_progress(file_name, message):
    with open(file_name, 'a+') as file:
        file.write(message + "\n")

# Loading file using polars
# X = pl.read_csv(f"{dataset_path}histone_features.csv")
# y = pl.read_csv(f"{dataset_path}value_1_df.csv")

print('Loading data ...')
X = pl.read_csv("histone_features.csv")
y = pl.read_csv("value_1_df.csv")
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
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 1)  # Single output for regression
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.leaky_relu(self.fc4(x))
        x = self.fc5(x)  #
        return x
    
input_dim = 20000
model = NeuralNetwork(input_dim).to(device)
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Setup TensorBoard writer
LOG_DIR = "runs/nn_model"
# train_writer = SummaryWriter(os.path.join(LOG_DIR, "train"))
# test_writer = SummaryWriter(os.path.join(LOG_DIR, "test"))
writer = SummaryWriter(LOG_DIR)

# Function to save the best model
best_val_loss = float('inf')
def save_best_model(model, val_loss, epoch):
    global best_val_loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_file)
        print(f"Saved best model with validation loss: {val_loss:.4f} at epoch {epoch + 1}")


# Implement Early Stopping
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

early_stopping = EarlyStopping(patience=10, min_delta=0.001)

# Train the model
num_epochs = 200
for epoch in range(num_epochs):
    start_time = time.time()

    model.train()

    running_loss = 0.0
    # running_mse = 0.0
    # running_rmse = 0.0
    running_mae = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device) # Move the data to GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # mse = loss.item()
        # rmse = np.sqrt(mse)
        mae = mean_absolute_error(labels.cpu().numpy(), outputs.cpu().detach().numpy())        
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # running_mse += mse
        # running_rmse += rmse
        running_mae += mae
        
    # Log training loss to TensorBoard
    # writer.add_scalar('loss', {'train': running_loss / len(train_loader)}, epoch)
    # writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)
    avg_train_loss = running_loss / len(train_loader)
    # avg_train_mse = running_mse / len(train_loader)
    # avg_train_rmse = running_rmse / len(train_loader)
    avg_train_mae = running_mae / len(train_loader)

    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    # writer.add_scalar('MSE/Train', avg_train_mse, epoch)
    # writer.add_scalar('RMSE/Train', avg_train_rmse, epoch)
    writer.add_scalar('MAE/Train', avg_train_mae, epoch)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # message = f'Epoch [{epoch + 1}/{num_epochs}], Time: {elapsed_time}, Loss: {avg_train_loss:.4f}, MSE: {avg_train_mse:.4f}, RMSE: {avg_train_rmse:.4f}, MAE: {avg_train_mae:.4f}'

    message = f'Epoch [{epoch + 1}/{num_epochs}], Time: {elapsed_time:.4f}, Loss: {avg_train_loss:.4f}, MAE: {avg_train_mae:.4f}'
    print(message)
    save_progress(progress_file, message)

    # Evaluate the model on the test dataset
    model.eval()
    total_loss = 0.0
    # total_mse = 0.0
    # total_rmse = 0.0
    total_mae = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion (outputs, labels)
            
            mse = loss.item()
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(labels.cpu().numpy(), outputs.cpu().numpy())

            total_loss += loss.item()
            # total_mse += mse
            # total_rmse += rmse
            total_mae += mae

    avg_test_loss = total_loss / len(test_loader)
    # avg_test_mse = total_mse / len(test_loader)
    # avg_test_rmse = total_rmse / len(test_loader)
    avg_test_mae = total_mae / len(test_loader)
    
    writer.add_scalar('Loss/Test', avg_test_loss, epoch)
    # writer.add_scalar('MSE/Test', avg_test_mse, epoch)
    # writer.add_scalar('RMSE/Test', avg_test_rmse, epoch)
    writer.add_scalar('MAE/Test', avg_test_mae, epoch)
    
    # message = f'Average loss on the test data: {avg_test_loss:.4f}, MSE: {avg_test_mse:.4f}, RMSE: {avg_test_rmse:.4f}, MAE: {avg_test_mae:.4f}'
    message = f'Average loss on the test data: {avg_test_loss:.4f}, MAE: {avg_test_mae:.4f}'

    print(message)
    save_progress(progress_file, message)

    # # Check early stopping condition
    # early_stopping(avg_test_loss)
    # if early_stopping.early_stop:
    #     print("Early stopping triggered. Stopping training.")
    #     break

    # Save the best model
    save_best_model(model, avg_test_loss, epoch)

# Close the TensorBoard writer
# train_writer.close()
# test_writer.close()
writer.close()

# Load the best model
model.load_state_dict(torch.load(best_model_file))
print(f"Loaded the best model from {best_model_file}")

# Making predictions with the test data
with torch.no_grad():
    predictions = model(X_test_tensor)

# Convert predictions to NumPy array
predictions_np = predictions.cpu().numpy()
y_test_np = y_test_tensor.cpu().numpy()

# Calculate evaluation metrics
mse = mean_squared_error(y_test_np, predictions_np)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_np, predictions_np)
r2 = r2_score(y_test_np, predictions_np)

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
    'y_test': y_test_np.flatten(),
    'y_pred': predictions_np.flatten()
})

# Save the DataFrame to a CSV file
results_df.to_csv(f"nn_predictions_{current_time}.csv", index=False)
print(f"Predictions and true values saved to 'nn_predictions_{current_time}.csv'.")