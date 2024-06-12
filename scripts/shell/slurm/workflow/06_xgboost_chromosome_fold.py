import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

dataset_path = "/group/pmc021/amunif/epi-thesis/dataset/"

def print_df_info(df):
    print(df.head())
    print(df.shape)

def load_large_csv(file_name, chunksize=20000):
    # Read the CSV file
    mylist = []

    for chunk in pd.read_csv(file_name, chunksize = chunksize):
        mylist.append(chunk)

    gene_exp_df = pd.concat(mylist, axis = 0)
    
    del mylist
    return gene_exp_df

chrom_list = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']

# Loading data
chrom_df = pd.read_csv(f"{dataset_path}chrom_df.csv")
print_df_info(chrom_df)

value_1_df = pd.read_csv(f"{dataset_path}value_1_df.csv")
print_df_info(value_1_df)

# Load the data
histone_features_df = load_large_csv(f"{dataset_path}histone_features.csv")
print_df_info(histone_features_df)

# Combine all data into single dataframe
all_data_df = pd.concat([chrom_df, value_1_df, histone_features_df], axis = 1)
del chrom_df
del value_1_df
del histone_features_df

# Do the XGBoost chromosome fold modeling
for current_chrom in chrom_list:
    print(f"{current_chrom} as the test. The rest are for training.")
    X_train = all_data_df[all_data_df['h_chrom'] != current_chrom]
    y_train = X_train[['h_value_1']]
    X_train = X_train.drop(columns = ['h_chrom', 'h_value_1'])

    X_test = all_data_df[all_data_df['h_chrom'] == current_chrom]
    y_test = X_test[['h_value_1']]
    X_test = X_test.drop(columns=['h_chrom', 'h_value_1'])

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'reg:squarederror',
        'max_depth': 5,
        'learning_rate': 0.1,
        'seed': 42
    }

    num_round = 100

    bst = xgb.train(params, dtrain, num_round)

    # Make predictions
    y_pred = bst.predict(dtest)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")

    del X_train
    del y_train
    del X_test
    del y_test