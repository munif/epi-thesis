import pandas as pd

dataset_path = "/group/pmc021/amunif/epi-thesis/dataset/"

chrom_list = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']

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

for chrom in chrom_list:
    print(f"Processing {chrom} ...")
    df = all_data_df[all_data_df['h_chrom'] == chrom]
    df.to_csv(f"{dataset_path}chromosome-fold/{chrom}.csv", index=False)
    del df