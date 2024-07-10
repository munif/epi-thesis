import pandas as pd
import numpy as np

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

dataset_path = "/group/pmc021/amunif/epi-thesis/dataset/"

# Read the CSV file
mylist = []

for chunk in pd.read_csv(f'{dataset_path}gene_exp.csv', chunksize=20000):
    mylist.append(chunk)

gene_exp_df = pd.concat(mylist, axis = 0)
del mylist


# Convert string representation back to arrays
gene_exp_df.loc[:, 'h3k4me3']  = gene_exp_df['h3k4me3'].apply(lambda x: np.array(list(map(float, x.split(',')))))
gene_exp_df.loc[:, 'h3k9ac']   = gene_exp_df['h3k9ac'].apply(lambda x: np.array(list(map(float, x.split(',')))))
gene_exp_df.loc[:, 'h3k9me3']  = gene_exp_df['h3k9me3'].apply(lambda x: np.array(list(map(float, x.split(',')))))
gene_exp_df.loc[:, 'h3k27ac']  = gene_exp_df['h3k27ac'].apply(lambda x: np.array(list(map(float, x.split(',')))))
gene_exp_df.loc[:, 'h3k27me3'] = gene_exp_df['h3k27me3'].apply(lambda x: np.array(list(map(float, x.split(',')))))


# Explode the column
h3k4me3_df = pd.DataFrame(gene_exp_df['h3k4me3'].tolist())
h3k4me3_df.columns = [f'h3k4me3_{i}' for i in range(h3k4me3_df.shape[1])]

h3k9ac_df = pd.DataFrame(gene_exp_df['h3k9ac'].tolist())
h3k9ac_df.columns = [f'h3k9ac_{i}' for i in range(h3k9ac_df.shape[1])]

h3k9me3_df = pd.DataFrame(gene_exp_df['h3k9me3'].tolist())
h3k9me3_df.columns = [f'h3k9me3_{i}' for i in range(h3k9me3_df.shape[1])]

h3k27ac_df = pd.DataFrame(gene_exp_df['h3k27ac'].tolist())
h3k27ac_df.columns = [f'h3k27ac_{i}' for i in range(h3k27ac_df.shape[1])]

h3k27me3_df = pd.DataFrame(gene_exp_df['h3k27me3'].tolist())
h3k27me3_df.columns = [f'h3k27me3_{i}' for i in range(h3k27me3_df.shape[1])]

chrom_df = gene_exp_df[['h_chrom']]
value_1_df = gene_exp_df[['h_value_1']]

# # Saving all into the files
# h3k4me3_df.to_csv(f"{dataset_path}h3k4me3_df.csv", index=False)
# h3k9ac_df.to_csv(f"{dataset_path}h3k9ac_df.csv", index=False)
# h3k9me3_df.to_csv(f"{dataset_path}h3k9me3_df.csv", index=False)
# h3k27ac_df.to_csv(f"{dataset_path}h3k27ac_df.csv", index=False)
# h3k27me3_df.to_csv(f"{dataset_path}h3k27me3_df.csv", index=False)

X = pd.concat([h3k4me3_df, h3k9ac_df, h3k9me3_df, h3k27ac_df, h3k27me3_df], axis=1)
X.to_csv(f"{dataset_path}histone_features.csv", index=False)

chrom_df.to_csv(f"{dataset_path}chrom_df.csv", index=False)
value_1_df.to_csv(f"{dataset_path}value_1_df.csv", index=False)