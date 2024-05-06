import pandas as pd
import numpy as np
import pyranges as pr

# Please refer to BED file format
# https://en.wikipedia.org/wiki/BED_(file_format)

def load_histone_data(file_name, type):
    columns = ['chrom', 'chromStart', 'chromEnd', 'name']
    df = pd.read_csv(file_name, sep="\t", header=None, names=columns)
    df.loc[:, "length"] = df["chromEnd"] - df["chromStart"]
    df.loc[:, "type"] = type
    return df

def load_histone_files():
    h3k4me3_df = load_histone_data("dataset/HepG2_Male.histone.H3K4me3.peak.bed", "h3k4me3")
    h3k9ac_df = load_histone_data("dataset/HepG2_Male.histone.H3K9ac.peak.bed", "h3k9ac")
    h3k27ac_df = load_histone_data("dataset/HepG2_Male.histone.H3K27ac.peak.bed", "h3k27ac")
    h3k27me3_df = load_histone_data("dataset/HepG2_Male.histone.H3K27me3.peak.bed", "h3k27me3")
    h3k9me3_df = load_histone_data("dataset/HepG2_Male.histone.H3K9me3.peak.bed", "h3k9me3")

    frames = [h3k4me3_df, h3k9ac_df, h3k27ac_df, h3k27me3_df, h3k9me3_df]
    histone_df = pd.concat(frames)

    return histone_df

def count_histone(histone_df, row, threshold = 0.8, histone_length = 146):
    tss = row["tss"]

    hist_df = histone_df.loc[# Inside the +/- 2k from TSS
                             (((histone_df["chromStart"] >= tss - 2000) & (histone_df["chromEnd"] <= tss + 2000)) |
                             # Intersect with -2k or +2 from TSS
                             (((histone_df["chromEnd"] - (tss - 2000))/histone_length).between(threshold, 1.0)) |
                             ((((tss + 2000) - histone_df["chromStart"])/histone_length).between(threshold, 1.0)))]
    
    hist_count_dict = hist_df["type"].value_counts().to_dict()
    hist_count_dict.update({'histone_count_total': len(hist_df.index)})

    return hist_count_dict

def hepg2_intersect_refseq(refseq_df, row):
    hepg2_refseq = refseq_df[(refseq_df['tss'] >= row['chromStart']) & (refseq_df['tss'] <= row['chromEnd'])]
    hepg2_dict = hepg2_refseq.sum(numeric_only=True).to_dict()

    if (len(hepg2_refseq) == 0):
        hepg2_dict.update({'status_refseq': 0})
    else:
      hepg2_dict.update({'status_refseq': 1})
    
    return hepg2_dict

def count_gene(gene_df, row, threshold = 0.8, tss_length = 4000):
    tss = row["tss"]
    hist_df = gene_df.loc[  # Inside the +/- 2k from TSS
                             (((gene_df["chromStart"] >= tss - 2000) & (gene_df["chromEnd"] <= tss + 2000)) |
                             
                             # Intersect with -2k or +2 from TSS
                             (((gene_df["chromEnd"] - (tss - 2000))/tss_length).between(threshold, 1.0)) |
                             ((((tss + 2000) - gene_df["chromStart"])/tss_length).between(threshold, 1.0))) |
                             
                             # TSS inside the gene
                             ((tss + 2000 <= gene_df["chromEnd"]) & (tss - 2000 >= gene_df["chromStart"]))
                        ]
    
    return len(hist_df.index)

def get_tss(row):
    if row['Strand'] == "+":
        return row['Start']
    return row['End']

def load_knownGene():
    hg19knownGene_df = pr.read_gtf("dataset/hg19.knownGene.gtf", as_df=True)
    hg19knownGene_df.loc[:, "tss"] = hg19knownGene_df.apply(lambda row: get_tss(row), axis=1)
    return hg19knownGene_df

def load_ncbiRefSeq():
    hg19ncbiRefSeq_df = pr.read_gtf("dataset/hg19.ncbiRefSeq.gtf", as_df=True)
    hg19ncbiRefSeq_df.loc[:, "tss"] = hg19ncbiRefSeq_df.apply(lambda row: get_tss(row), axis=1)
    return hg19ncbiRefSeq_df

# Flatten the array
def flatten_concatenation(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list