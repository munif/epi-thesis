'''
References: 
1. https://xgboost.readthedocs.io/en/stable/python/examples/cross_validation.html

'''

import pandas as pd
import numpy as np

import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

    for chunk in pd.read_csv(file_name, chunksize = chunksize):
        mylist.append(chunk)

    gene_exp_df = pd.concat(mylist, axis = 0)
    
    del mylist
    return gene_exp_df

# Load the data
X = load_large_csv(f"{dataset_path}histone_features.csv")
y = load_large_csv(f"{dataset_path}value_1_df.csv")

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# Train the model
params = {
            'n_estimators'     : 10000, 
            'colsample_bytree' : 0.8, 
            'max_depth'        : 10, 
            'subsample'        : 1.0, 
            'learning_rate'    : 0.5, 
            'objective'        : 'reg:squarederror', 
            'eval_metric'      : 'rmse',
            'seed'             : 42
         }

model = XGBRegressor(**params)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print the evaluation metrics
print_evaluation(y_test, y_pred)

y_test.to_csv("y_test_10000.csv", index=False)
pd.DataFrame(y_pred).to_csv("y_pred_10000.csv", index=False)