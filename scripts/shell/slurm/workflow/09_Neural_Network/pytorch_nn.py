# Load the required library
import pandas as pd
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

dataset_path = "/group/pmc021/amunif/epi-thesis/dataset/"

current_time = datetime.now().strftime("%Y%m%d%H%M%S")
progress_file = f"/group/pmc021/amunif/epi-thesis/scripts/shell/slurm/workflow/output/09/progress_{current_time}.txt"


def load_large_csv(file_name, chunksize=20000):
    # Read the CSV file
    mylist = []

    for chunk in pd.read_csv(file_name, chunksize = chunksize):
        mylist.append(chunk)

    df = pd.concat(mylist, axis = 0)
    
    del mylist
    return df

def save_progress(file_name, message):
    with open(file_name, 'a+') as file:
        file.write(message + "\n")

# Load the data
X = load_large_csv(f"{dataset_path}histone_features.csv")
y = load_large_csv(f"{dataset_path}value_1_df.csv")

# Convert to numpy
X_np = X.to_numpy()
y_np = y.to_numpy()

# Split the data into training and testing
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)

# Create TensorDataset for training and testing sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create a DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)  # Single output for regression

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)  # No activation function for the output layer in regression
        return x
    
input_dim = 20000
model = NeuralNetwork(input_dim)
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    message = f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}'
    print(message)
    save_progress(progress_file, message)

# Evaluate the model
model.eval()
total_loss = 0.0
with torch.no_grad():
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

print(f'Average loss on the training data: {total_loss/len(train_loader):.4f}')


# Making predictions with the test data
with torch.no_grad():
    predictions = model(X_test_tensor)

# Convert predictions to NumPy array
predictions_np = predictions.numpy()
y_test_np = y_test_tensor.numpy()

# Calculate evaluation metrics
mse = mean_squared_error(y_test_np, predictions_np)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_np, predictions_np)
r2 = r2_score(y_test_np, predictions_np)

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R2 Score: {r2}')

# Create a DataFrame with true values and predictions
results_df = pd.DataFrame({
    'True Values': y_test_np.flatten(),
    'Predictions': predictions_np.flatten()
})

# Save the DataFrame to a CSV file
results_df.to_csv(f"predictions_{current_time}.csv", index=False)
print(f"Predictions and true values saved to 'predictions_{current_time}.csv'.")