bedtools intersect -a hepg2_df.bed -b ncbi_histone_df.bed -wa -wb -f 0.80

bedtools intersect -a hepg2_df.bed -b ncbi_refseq_hg38.bed -wa -wb -f 0.80 > hepg2_ncbirefseq_hg38_overlap80.bed

bedtools intersect -a hepg2_df.bed -b ncbi_refseq_hg38.bed -wa -wb > hepg2_ncbirefseq_hg38_overlap80.bed

# NCBI RefSeq hg19
bedtools intersect -a hepg2_exp.bed -b ncbirefseq_hg19.bed -wa -wb -f 0.80 > hepg2_ncbi_hg19_overlap.bed
bedtools intersect -a hepg2_exp.bed -b ncbirefseq_hg19.bed -v > hepg2_ncbi_hg19_nonoverlap.bed

# NCBI RefSeq hg38
bedtools intersect -a hepg2_exp.bed -b ncbirefseq_hg38.bed -wa -wb -f 0.80 > hepg2_ncbi_hg38_overlap.bed
bedtools intersect -a hepg2_exp.bed -b ncbirefseq_hg38.bed -v > hepg2_ncbi_hg38_nonoverlap.bed