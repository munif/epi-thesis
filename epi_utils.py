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
    hist_df = histone_df[((histone_df["chromStart"] >= tss - 2000) & 
                     (histone_df["chromEnd"] <= tss + 2000)) |
                     (((histone_df["chromEnd"] - (tss - 2000))/histone_length).between(threshold, 1.0)) |
                     ((((tss + 2000) - histone_df["chromStart"])/histone_length).between(threshold, 1.0))]
    
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