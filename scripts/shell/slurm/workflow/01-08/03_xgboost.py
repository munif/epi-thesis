# Load required libraries
import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

'''
User defined functions & variables
'''

dataset_path = "/group/pmc021/amunif/epi-thesis/dataset/"

def print_df_info(df):
    print(df.head())
    print(df.shape)

def print_evaluation(y_test, y_pred):
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")

def load_large_csv(file_name, chunksize=20000):
    # Read the CSV file
    mylist = []

    for chunk in pd.read_csv(file_name, chunksize=chunksize):
        mylist.append(chunk)

    gene_exp_df = pd.concat(mylist, axis = 0)
    
    del mylist
    return gene_exp_df

'''
MAIN PROGRAM
'''

# Load the data
X = load_large_csv(f"{dataset_path}histone_features.csv")
print_df_info(X)

y = load_large_csv(f"{dataset_path}value_1_df.csv")
print_df_info(y)

# Prepare the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Train the model
# Set up the parameters and train the model
params = {
    'objective': 'reg:squarederror',
    'max_depth': 5,
    'learning_rate': 0.1,
    'seed': 42
}

num_round = 50

bst = xgb.train(params, dtrain, num_round)

# Prediction
y_pred = bst.predict(dtest)

# Evaluation
print_evaluation(y_test, y_pred)