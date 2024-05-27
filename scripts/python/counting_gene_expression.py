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
        # Loading NCBI Refseq with histone count
        refseq_df = pd.read_csv(f"dataset/histone_count/{chrom}.csv")

        # Loading HepG2 data
        hepg2 = pd.read_csv(f"dataset/HepG2/merged/{chrom}.csv")

        # Counting the histone on the gene expression
        hepg2.loc[:, 'hepg2_refseq'] = hepg2.progress_apply(lambda row: eu.hepg2_intersect_refseq(refseq_df, row), axis = 1)

        # Normalize the JSON column and selecting the required column
        hepg2_refseq_df = pd.json_normalize(hepg2['hepg2_refseq'])
        hepg2_histone = pd.concat([hepg2, hepg2_refseq_df], axis = 1)

        hepg2_histone = hepg2_histone[['test_id', 'gene_id', 'gene', 'locus', 'sample_1', 'sample_2', 'status',
        'value_1', 'value_2', 'log2(fold_change)', 'test_stat', 'p_value',
        'q_value', 'significant', 'chrom', 'chromStart', 'chromEnd',
        'h3k4me3', 'h3k9ac', 'h3k9me3', 'h3k27ac', 'h3k27me3', 'histone_count_total']]
        
        # Correcting the data types
        hepg2_histone = hepg2_histone.astype({'h3k4me3': 'int32', 'h3k9ac':'int32', 'h3k9me3':'int32', 'h3k27ac':'int32', 'h3k27me3':'int32', 'histone_count_total':'int32'})

        # Save the result
        hepg2_histone.to_csv(f"dataset/HepG2/hepg2_{chrom}_with_histone_count.csv", header=True, index=False)



def main():
    data_processing()

if __name__ == "__main__":
    main()