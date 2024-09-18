import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm

tqdm.pandas()


DATASET_PATH = '../dataset/E066/'

def encode_histone_exp(row, histone):
    arr = []
    start = row['start']
    end = row['end']
    gene_id = row['gene_id']
    count = 1

    for i in range(start, end, 100):
        histone_df = histone[
                    # The same gene ID
                    (histone['gene_id'] == gene_id) &
                    (
                        # The window is inside the histone
                        ((histone['chromStart'] <= i) & (histone['chromEnd'] >= i + 100)) |
                        # The start of histone is inside the window
                        ((histone['chromStart'] >= i) & (histone['chromStart'] <= i + 100) & (histone['chromEnd'] >= i + 100)) |
                        # The end of histone is inside the window
                        ((histone['chromStart'] <= i) & (histone['chromEnd'] >= i) & (histone['chromEnd'] <= i + 100))
                    )
                    ]
        size = histone_df.shape[0]
        if (size > 0):
            avg = histone_df['signalValue'].mean()
        else:
            avg = 0.0
        
        # print(f"Bin {count}: {i} to {i + 100}: {histone_df.shape[0]}, average: {avg}")
        arr.append(avg)
        count += 1
    return arr

# Load the gene expression file
print("Loading the gene expression file")

column_names = ['chromosome_name',
                'start',
                'end',
                'gene_id',
                'E066',
                'strand',
                'label',
                'external_gene_name',
                'start_position',
                'end_position',
                'tss' 
                ]

E066_df = pd.read_csv(os.path.join(DATASET_PATH, "E066.bed"), sep="\t", names = column_names)
print(f"E066 file: {E066_df.shape}")

# Load the histone data
print("Loading the histone file")
H3K4me1_df = pd.read_csv(os.path.join(DATASET_PATH, "E066_H3K4me1_df.csv"))
H3K4me3_df = pd.read_csv(os.path.join(DATASET_PATH, "E066_H3K4me3_df.csv"))
H3K9me3_df = pd.read_csv(os.path.join(DATASET_PATH, "E066_H3K9me3_df.csv"))
H3K27me3_df = pd.read_csv(os.path.join(DATASET_PATH, "E066_H3K27me3_df.csv"))
H3K36me3_df = pd.read_csv(os.path.join(DATASET_PATH, "E066_H3K36me3_df.csv"))

print(H3K4me1_df.shape)
print(H3K4me3_df.shape)
print(H3K9me3_df.shape)
print(H3K27me3_df.shape)
print(H3K36me3_df.shape)

# Generate the histone column
print("Generate histone column")
E066_df.loc[:, 'H3K4me1'] = E066_df.progress_apply(lambda row: encode_histone_exp(row, H3K4me1_df), axis = 1)
E066_df.loc[:, 'H3K4me3'] = E066_df.progress_apply(lambda row: encode_histone_exp(row, H3K4me3_df), axis = 1)
E066_df.loc[:, 'H3K9me3'] = E066_df.progress_apply(lambda row: encode_histone_exp(row, H3K9me3_df), axis = 1)
E066_df.loc[:, 'H3K27me3'] = E066_df.progress_apply(lambda row: encode_histone_exp(row, H3K27me3_df), axis = 1)
E066_df.loc[:, 'H3K36me3'] = E066_df.progress_apply(lambda row: encode_histone_exp(row, H3K36me3_df), axis = 1)

# Saving to file
print("Saving to parquet file")
E066_df.to_parquet(os.path.join(DATASET_PATH, "E066_with_histone_2.parquet"))