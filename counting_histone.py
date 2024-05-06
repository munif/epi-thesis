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
        print(f"Processing '{chrom}'")
        
        # Loading histone
        histone_df = pd.read_csv(f"dataset/histone/{chrom}.csv")

        # Loading NCBI Refseq
        refseq_df = pd.read_csv(f"dataset/ncbiRefSeq/merged/{chrom}.csv")

        # Counting histone
        refseq_df.loc[:, "hist_count_dict"] = refseq_df.progress_apply(lambda row: eu.count_histone(histone_df, row), axis = 1)

        # Normalize the JSON column
        refseq_df = pd.concat([refseq_df, pd.json_normalize(refseq_df['hist_count_dict'])], axis = 1)

        # Handling null values and correcting the data types
        refseq_df.drop(columns=["hist_count_dict"], inplace=True)
        refseq_df.fillna(0, inplace=True)
        refseq_df = refseq_df.astype({'h3k4me3': 'int32', 'h3k9ac':'int32', 'h3k9me3':'int32', 'h3k27ac':'int32', 'h3k27me3':'int32', 'histone_count_total':'int32'})

        refseq_df = refseq_df[['Chromosome', 'Source', 'Feature', 'Start', 'End', 'Score', 'Strand',
        'Frame', 'gene_id', 'transcript_id', 'gene_name', 'exon_number',
        'exon_id', 'tss', 'h3k4me3',
        'h3k9ac', 'h3k27ac', 'h3k27me3', 'h3k9me3', 'histone_count_total']]

        # Saving the result
        refseq_df.to_csv(f"dataset/histone_count/{chrom}.csv", header=True, index=False)

def main():
    data_processing()

if __name__ == "__main__":
    main()