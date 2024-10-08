import os
import glob
import pandas as pd

DATASET_PATH = '/group/pmc021/amunif/epi-thesis/workflow/07_deepchrome/dataset/E066/'

# Get CSV files list from a folder
csv_files = glob.glob(os.path.join(DATASET_PATH, "experiments", "gpu", "min-avg-max", "*.csv"))
print(csv_files)

# Read each CSV file into DataFrame
# This creates a list of dataframes
df_list = (pd.read_csv(file) for file in csv_files)

# Concatenate all DataFrames
big_df = pd.concat(df_list, ignore_index=True)

# Save to file
big_df.to_csv(os.path.join(DATASET_PATH, "experiments", "gpu", "all_results.csv"), header=True, index=False)