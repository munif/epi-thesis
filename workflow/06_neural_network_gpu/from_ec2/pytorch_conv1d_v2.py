# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import numpy as np

from datetime import datetime
import time

# Prepare the progress file
current_time = datetime.now().strftime("%Y%m%d%H%M%S")
progress_file = f'progress_conv1d_{current_time}.txt'

'''
User defined functions
'''
def load_large_csv(file_name, chunksize=1000):
    # Read the CSV file
    mylist = []

    print('Loading the data ...')
    for chunk in pd.read_csv(file_name, chunksize = chunksize):
        print('Loading the chunk ...')
        mylist.append(chunk)

    df = pd.concat(mylist, axis = 0)
    
    del mylist
    return df

def save_progress(file_name, message):
    with open(file_name, 'a+') as file:
        file.write(message + "\n")


# Load the data
X = load_large_csv('histone_features.csv')
y = load_large_csv('value_1_df.csv')

save_progress(progress_file, "Finished load the data")

# Convert to numpy
X_np = X.to_numpy()
y_np = y.to_numpy()

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_np, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y_np, dtype=torch.float32)

# Create a TensorDataset
dataset = TensorDataset(X_tensor, y_tensor)

# Split dataset into training, validation, and testing sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create CNN1D class
class HighDimCNN1D(nn.Module):
    def __init__(self, input_size):
        super(HighDimCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after the convolution and pooling layers
        conv_output_size = input_size // 8 
        
        self.fc1 = nn.Linear(128 * conv_output_size, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

'''
Training
'''
# Create the model object
input_size = 20000  # Number of features in the input
model = HighDimCNN1D(input_size=input_size).cuda()  # Move model to GPU

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Gradient accumulation steps
accumulation_steps = 4

# Using mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Training loop with validation, gradient accumulation, and mixed precision
num_epochs = 100

for epoch in range(num_epochs):
    start_time = time.time()

    model.train()
    train_loss = 0.0
    optimizer.zero_grad()  # Zero the parameter gradients
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()  # Move data to GPU

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = loss / accumulation_steps  # Normalize loss to account for gradient accumulation

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += loss.item() * inputs.size(0) * accumulation_steps  # Accumulate total loss

    train_loss /= len(train_loader.dataset)

    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.cuda(), targets.cuda()  # Move data to GPU
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_loader.dataset)

    end_time = time.time()
    elapsed_time = end_time - start_time
    message = f'Epoch {epoch+1}, Time: {elapsed_time}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}'
    print(message)
    save_progress(progress_file, message)

# Save the model
torch.save(model, 'model_v2.pt')

'''
Evaluation
'''
# Evaluation on the test set
model.eval()
all_targets = []
all_predictions = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.cuda(), targets.cuda()  # Move data to GPU
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(outputs.cpu().numpy())

all_targets = np.array(all_targets)
all_predictions = np.array(all_predictions)

# Print the evaluation metrics
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

# Save the test targets and predictions to a CSV file
results_df = pd.DataFrame({'y_test': all_targets.flatten(), 'y_pred': all_predictions.flatten()})
results_df.to_csv(f'predictions_{current_time}.csv', index=False)

print(f'Predictions saved to predictions_{current_time}.csv')