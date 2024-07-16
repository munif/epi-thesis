# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import pandas as pd
import polars as pl

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import numpy as np

from datetime import datetime
import time

print("Hello World!")

# Setup the CUDA devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"I am using: {device}")