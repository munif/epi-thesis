'''
References: 
1. https://xgboost.readthedocs.io/en/stable/python/examples/cross_validation.html

'''

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

# Load the data
X = load_large_csv(f"{dataset_path}histone_features.csv")
# print_df_info(X)

y = load_large_csv(f"{dataset_path}value_1_df.csv")
# print_df_info(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'max_depth': 10,
    'learning_rate': 0.1,  # learning rate
    'objective': 'reg:squarederror',  # for regression
    'eval_metric': 'rmse',  # evaluation metric
    'seed': 42
}

# Number of folds and boosting rounds
num_boost_round = 100
nfold = 10

cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_boost_round,
    nfold=nfold,
    metrics={'rmse'},
    # early_stopping_rounds=10,
    as_pandas=True,
    seed=42
)

best_num_boost_round = cv_results['test-rmse-mean'].idxmin()

# Train the final model
final_model = xgb.train(params, dtrain, num_boost_round=best_num_boost_round)

# Make predictions on the test set
y_pred = final_model.predict(dtest)


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R2 Score: {r2}\n")

print(f"Test set Mean Squared Error: {mse}")
print(f"Test set R-squared: {r2}")

# Evaluate the result
print(cv_results)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(cv_results['train-rmse-mean'], label='Train RMSE')
plt.plot(cv_results['test-rmse-mean'], label='Test RMSE')
plt.fill_between(cv_results.index, 
                 cv_results['train-rmse-mean'] - cv_results['train-rmse-std'], 
                 cv_results['train-rmse-mean'] + cv_results['train-rmse-std'], alpha=0.1)
plt.fill_between(cv_results.index, 
                 cv_results['test-rmse-mean'] - cv_results['test-rmse-std'], 
                 cv_results['test-rmse-mean'] + cv_results['test-rmse-std'], alpha=0.1)
plt.xlabel('Number of Boosting Rounds')
plt.ylabel('RMSE')
plt.title('XGBoost Cross-Validation RMSE')
plt.legend()

# Save the plot to a file
plt.savefig('07_xgboost_cv_rmse.png')