import pandas as pd

dataset_path = "/group/pmc021/amunif/project/dataset/"

gene_df = pd.read_csv(f"{dataset_path}histone_count_overlap80.csv", sep="\t")

slice_df = gene_df.head(10)

slice_df.to_csv(f"{dataset_path}slice_10.csv", header=True, index=False)