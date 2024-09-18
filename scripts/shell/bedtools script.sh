bedtools intersect -a hepg2_df.bed -b ncbi_histone_df.bed -wa -wb -f 0.80

bedtools intersect -a hepg2_df.bed -b ncbi_refseq_hg38.bed -wa -wb -f 0.80 > hepg2_ncbirefseq_hg38_overlap80.bed

bedtools intersect -a hepg2_df.bed -b ncbi_refseq_hg38.bed -wa -wb > hepg2_ncbirefseq_hg38_overlap80.bed

# NCBI RefSeq hg19
bedtools intersect -a hepg2_exp.bed -b ncbirefseq_hg19.bed -wa -wb -f 0.80 > hepg2_ncbi_hg19_overlap.bed
bedtools intersect -a hepg2_exp.bed -b ncbirefseq_hg19.bed -v > hepg2_ncbi_hg19_nonoverlap.bed

# NCBI RefSeq hg38
bedtools intersect -a hepg2_exp.bed -b ncbirefseq_hg38.bed -wa -wb -f 0.80 > hepg2_ncbi_hg38_overlap.bed
bedtools intersect -a hepg2_exp.bed -b ncbirefseq_hg38.bed -v > hepg2_ncbi_hg38_nonoverlap.bed

# HepG2 + tss from NCBI RefSeq hg19 overlap with histone peak
bedtools intersect -a overlap_2188.csv -b HepG2_Male.histone.H3K4me3.peak.bed -c > overlap_histone1.bed
bedtools intersect -a overlap_histone1.bed -b HepG2_Male.histone.H3K9ac.peak.bed -c > overlap_histone2.bed
bedtools intersect -a overlap_histone2.bed -b HepG2_Male.histone.H3K9me3.peak.bed -c > overlap_histone3.bed
bedtools intersect -a overlap_histone3.bed -b HepG2_Male.histone.H3K27ac.peak.bed -c > overlap_histone4.bed
bedtools intersect -a overlap_histone4.bed -b HepG2_Male.histone.H3K27me3.peak.bed -c > overlap_histone5.bed


# DeepChrome Processing
## Without threshold F 0.80
bedtools intersect -a E066.bed -b E066-H3K4me1.narrowPeak -wa -wb > E066_H3K4me1.bed
bedtools intersect -a E066.bed -b E066-H3K4me3.narrowPeak -wa -wb > E066_H3K4me3.bed
bedtools intersect -a E066.bed -b E066-H3K9me3.narrowPeak -wa -wb > E066_H3K9me3.bed
bedtools intersect -a E066.bed -b E066-H3K27me3.narrowPeak -wa -wb > E066_H3K27me3.bed
bedtools intersect -a E066.bed -b E066-H3K36me3.narrowPeak -wa -wb > E066_H3K36me3.bed

## With threshold F
### -F: with regards b files
### -f: with regards a files
bedtools intersect -a E066.bed -b E066-H3K4me1.narrowPeak -wa -wb -F 0.80 > E066_H3K4me1.bed