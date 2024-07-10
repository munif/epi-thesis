import pandas as pd
import numpy as np

dataset_path = "/group/pmc021/amunif/epi-thesis/dataset/"

def load_histone_data(file_name, type):
    columns = ['chrom', 'chromStart', 'chromEnd', 'name']
    df = pd.read_csv(file_name, sep="\t", header=None, names=columns)
    df["type"] = type
    return df

def encode_histone(row, histone_df, threshold = 0.8, histone_length = 146):  
  chrom = row["h_chrom"]
  tss_start = row['tss_start']
  tss_end = row['tss_end']

  hist_df = histone_df.loc[(histone_df['chrom'] == chrom) & 
                            (
                                # Inside the +/- 2KB from TSS
                                ((histone_df['chromStart'] >= tss_start) & (histone_df['chromEnd'] <= tss_end)) |

                                # Overlap at the start
                                ((histone_df['chromStart'] < tss_start) & 
                                (histone_df['chromEnd'] > tss_start) & 
                                (histone_df['chromEnd'] < tss_end) & 
                                ((histone_df['chromEnd'] - tss_start)/ histone_length >= threshold)) |

                                # Overlap at the end
                                ((histone_df['chromStart'] > tss_start) &
                                (histone_df['chromStart'] < tss_end) &
                                (histone_df['chromEnd'] > tss_end) &
                                ((tss_end - histone_df['chromStart'])/histone_length >= threshold))
                            )
                          ]

  start_idx = hist_df[['chromStart']].to_numpy()
  arr = np.zeros(4000)
  idx = [x for x in start_idx - tss_start]
  arr[idx] = 1

  return arr


'''
MAIN PROGRAM
'''

# Load gene data
print("Loading data")
gene_df = pd.read_csv(f"{dataset_path}histone_count_overlap80.csv", sep="\t")
gene_exp_df = gene_df[['h_chrom', 'h_chromStart', 'h_chromEnd', 'h_value_1', 'n_strand', 'tss', 'tss_start', 'tss_end']]
# print(gene_exp_df.head())

# Load Histone
print("Loading histone data")
h3k4me3_df = load_histone_data(f"{dataset_path}HepG2_Male.histone.H3K4me3.peak.bed", "h3k4me3")
h3k9ac_df = load_histone_data(f"{dataset_path}HepG2_Male.histone.H3K9ac.peak.bed", "h3k9ac")
h3k9me3_df = load_histone_data(f"{dataset_path}HepG2_Male.histone.H3K9me3.peak.bed", "h3k9me3")
h3k27ac_df = load_histone_data(f"{dataset_path}HepG2_Male.histone.H3K27ac.peak.bed", "h3k27ac")
h3k27me3_df = load_histone_data(f"{dataset_path}HepG2_Male.histone.H3K27me3.peak.bed", "h3k27me3")

# Encoding the histone position
print("Encoding histone column")
gene_exp_df.loc[:, 'h3k4me3'] = gene_exp_df.apply(lambda row: encode_histone(row, h3k4me3_df), axis = 1)
gene_exp_df.loc[:, 'h3k9ac'] = gene_exp_df.apply(lambda row: encode_histone(row, h3k9ac_df), axis = 1)
gene_exp_df.loc[:, 'h3k9me3'] = gene_exp_df.apply(lambda row: encode_histone(row, h3k9me3_df), axis = 1)
gene_exp_df.loc[:, 'h3k27ac'] = gene_exp_df.apply(lambda row: encode_histone(row, h3k27ac_df), axis = 1)
gene_exp_df.loc[:, 'h3k27me3'] = gene_exp_df.apply(lambda row: encode_histone(row, h3k27me3_df), axis = 1)

# # Save the data to parquet
# gene_exp_df.to_parquet(f"{dataset_path}gene_exp.parquet")

# Save the data to CSV
gene_exp_df.loc[:, 'h3k4me3']  = gene_exp_df['h3k4me3'].apply(lambda x: ','.join(map(str, x)))
gene_exp_df.loc[:, 'h3k9ac']   = gene_exp_df['h3k9ac'].apply(lambda x: ','.join(map(str, x)))
gene_exp_df.loc[:, 'h3k9me3']  = gene_exp_df['h3k9me3'].apply(lambda x: ','.join(map(str, x)))
gene_exp_df.loc[:, 'h3k27ac']  = gene_exp_df['h3k27ac'].apply(lambda x: ','.join(map(str, x)))
gene_exp_df.loc[:, 'h3k27me3'] = gene_exp_df['h3k27me3'].apply(lambda x: ','.join(map(str, x)))

gene_exp_df.to_csv(f"{dataset_path}gene_exp.csv", index=False)