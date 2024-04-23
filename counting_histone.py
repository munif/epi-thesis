import pandas as pd
from tqdm.auto import tqdm

import epi_utils as eu

histone_path = "dataset/histone/"
ncbiRefSeq_path = "dataset/ncbiRefSeq/merged/"
output_path = "dataset/histone_count/"

tqdm.pandas()

def data_processing():
    # Omitting the 'chrMT', 'chrUn' chromosome
    chrom_list = pd.read_csv("dataset/chrom_list.csv")["Chromosome"].to_list()

    for chrom in chrom_list:
        print(f"--- Processing {chrom} ---")

        # Reading file
        histone_chr_df = pd.read_csv(f"{histone_path}{chrom}.csv")
        ncbiRefSeq_chr_df = pd.read_csv(f"{ncbiRefSeq_path}{chrom}.csv")

        # Counting Histone
        ncbiRefSeq_chr_df.loc[:, "histone_count"] = ncbiRefSeq_chr_df.progress_apply(\
                                lambda row: eu.count_histone(histone_chr_df, row), axis=1)
        
        # Saving the result
        ncbiRefSeq_chr_df.to_csv(f"{output_path}{chrom}.csv", index=False)

    # # chrUn
    # chrom = 'chrUn'
    # processing_additional_chrom(chrom)

    # # chrMT
    # chrom = 'chrMT'
    # processing_additional_chrom(chrom)

# def processing_additional_chrom(chrom):
#     print(f"--- Processing {chrom} ---")
#     ncbiRefSeq_chr_df = pd.read_csv(f"{ncbiRefSeq_path}{chrom}.csv")
#     ncbiRefSeq_chr_df.loc[:, "histone_count"] = 0
#     ncbiRefSeq_chr_df.to_csv(f"{output_path}{chrom}.csv", index=False)

def main():
    data_processing()

if __name__ == "__main__":
    main()