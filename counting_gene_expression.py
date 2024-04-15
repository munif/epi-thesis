import pandas as pd
from tqdm.auto import tqdm

import epi_utils as eu

histone_path = "dataset/histone_count/"
hepG2_path = "dataset/HepG2/merged/"
output_path = "dataset/gene_count/"

chrom_list = ['chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16',
       'chr17', 'chr18', 'chr19', 'chr1', 'chr2', 'chr3', 'chrX', 'chr4',
       'chrY', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr20', 'chr21',
       'chr22', 'chrMT', 'chrUn']

tqdm.pandas()

def data_processing():
    for chrom in chrom_list:
        print(f"--- Processing {chrom} ---")

        # Reading input files
        gene_chr_df = pd.read_csv(f"dataset/HepG2/merged/{chrom}.csv")
        ncbiRefSeq_chr_df = pd.read_csv(f"dataset/histone_count/{chrom}.csv")

        # Counting gene expression
        ncbiRefSeq_chr_df.loc[:, "gene_count"] = ncbiRefSeq_chr_df.progress_apply(\
                                lambda row: eu.count_gene(gene_chr_df, row), axis=1)

        # Saving the data
        ncbiRefSeq_chr_df.to_csv(f"{output_path}{chrom}.csv", index=False)
    
    # chrUn
    chrom = 'chrUn'
    processing_additional_chrom(chrom)

    # chrMT
    chrom = 'chrMT'
    processing_additional_chrom(chrom)

def processing_additional_chrom(chrom):
    print(f"--- Processing {chrom} ---")
    ncbiRefSeq_chr_df = pd.read_csv(f"{histone_path}{chrom}.csv")
    ncbiRefSeq_chr_df.loc[:, "gene_count"] = 0
    ncbiRefSeq_chr_df.to_csv(f"{output_path}{chrom}.csv", index=False)

def main():
    data_processing()

if __name__ == "__main__":
    main()