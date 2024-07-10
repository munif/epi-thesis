# Load the required libraries
import pandas as pd
import numpy as np
import polars as pl

import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

dataset_path = "/group/pmc021/amunif/epi-thesis/dataset/"

# User defined function
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

def get_eval_metric(n_estimators, y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return n_estimators, mse, rmse, mae, r2

# Load the data using polars
X = pl.read_csv(f"{dataset_path}histone_features.csv")
y = pl.read_csv(f"{dataset_path}value_1_df.csv")

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

n_estimators = [100, 200, 300, 500, 1000, 5000, 10000]
train_eval_metric = []
test_eval_metric = []
eval_metric = 'mae'

y_train.write_csv(f"output-csv/{eval_metric}/y_train.csv")

for i in n_estimators:
    prefix = f"output-csv/{eval_metric}/xgboost_{i}_"
    # Train the model
    params = {
                'n_estimators'     : i, 
                'colsample_bytree' : 0.8, 
                'max_depth'        : 10, 
                'subsample'        : 1.0, 
                'learning_rate'    : 0.5, 
                'objective'        : 'reg:squarederror', 
                'eval_metric'      : eval_metric,
                'seed'             : 42
            }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    # Make prediction for y_train
    y_train_pred = model.predict(X_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Save all data files
    # y_train and y_train_pred
    pd.DataFrame(y_train_pred).to_csv(f"{prefix}y_train_pred.csv", index=False, header=['h_value_1'])

    # y_test and y_pred
    y_test.write_csv(f"{prefix}y_test.csv")
    pd.DataFrame(y_pred).to_csv(f"{prefix}y_pred.csv", index=False, header=['h_value_1'])

    # Evaluation metric
    train_eval_metric.append(get_eval_metric(i, y_train, y_train_pred))
    test_eval_metric.append(get_eval_metric(i, y_test, y_pred))

# Save the evaluation metric
pd.DataFrame(train_eval_metric, columns=['n_estimators', 'mse', 'rmse', 'mae', 'r2_score']).to_csv(f"output-csv/{eval_metric}/train_eval_metric.csv", index=False)

pd.DataFrame(test_eval_metric, columns=['n_estimators', 'mse', 'rmse', 'mae', 'r2_score']).to_csv(f"output-csv/{eval_metric}/test_eval_metric.csv", index=False)