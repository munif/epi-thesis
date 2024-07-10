import pandas as pd
import numpy as np

import xgboost as xgb
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

print("Finding the best parameters using grid_search")
# Load the data
X = load_large_csv(f"{dataset_path}histone_features.csv")
# print_df_info(X)

y = load_large_csv(f"{dataset_path}value_1_df.csv")
# print_df_info(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Initialize the XGBoost regressor
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Define the parameter grid
# param_grid = {
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'n_estimators': [100, 200, 300],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.8, 1.0]
# }

param_grid = {
    'max_depth': [5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.5, 0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=10, verbose=2, n_jobs=8)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"Best parameters found: {best_params}")
print(f"Best Mean Squared Error from GridSearchCV: {best_score}")

# Train the model with the best parameters using DMatrix
final_model = xgb.train(best_params, dtrain, num_boost_round=100)

# Make predictions with DMatrix
y_pred = final_model.predict(dtest)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R2 Score: {r2}")