#!/bin/bash

### Sort the bed files and find unique values ###
# H3K27ac.4.bed  H3K27me3.4.bed  H3K4me3.4.bed  H3K9ac.4.bed  H3K9me3.4.bed

sort -k1,1 -k2,2n H3K4me3.4.bed > H3K4me3.4.sorted.bed
uniq H3K4me3.4.sorted.bed > H3K4me3.4.sorted.uniq.bed

sort -k1,1 -k2,2n H3K9ac.4.bed > H3K9ac.4.sorted.bed
uniq H3K9ac.4.sorted.bed > H3K9ac.4.sorted.uniq.bed

sort -k1,1 -k2,2n H3K9me3.4.bed > H3K9me3.4.sorted.bed
uniq H3K9me3.4.sorted.bed > H3K9me3.4.sorted.uniq.bed

sort -k1,1 -k2,2n H3K27ac.4.bed > H3K27ac.4.sorted.bed
uniq H3K27ac.4.sorted.bed > H3K27ac.4.sorted.uniq.bed

sort -k1,1 -k2,2n H3K27me3.4.bed > H3K27me3.4.sorted.bed
uniq H3K27me3.4.sorted.bed > H3K27me3.4.sorted.uniq.bed


### Subsampling ###
# Check the number of lines
wc -l H3K4me3.4.sorted.uniq.bed
wc -l H3K9ac.4.sorted.uniq.bed
wc -l H3K9me3.4.sorted.uniq.bed
wc -l H3K27ac.4.sorted.uniq.bed
wc -l H3K27me3.4.sorted.uniq.bed

### Find the fragment length ###
# Convert the input control into tagAlign
awk 'BEGIN{OFS="\t"}{print $1,$2,$3,".",$5,$6}' Input.4.bed > Input.4.tagAlign

# Convert the bed files into tagAlign
awk 'BEGIN{OFS="\t"}{print $1,$2,$3,".",$5,$6}' H3K4me3.4.sorted.uniq.bed > H3K4me3.4.tagAlign
awk 'BEGIN{OFS="\t"}{print $1,$2,$3,".",$5,$6}' H3K9ac.4.sorted.uniq.bed > H3K9ac.4.tagAlign
awk 'BEGIN{OFS="\t"}{print $1,$2,$3,".",$5,$6}' H3K9me3.4.sorted.uniq.bed > H3K9me3.4.tagAlign
awk 'BEGIN{OFS="\t"}{print $1,$2,$3,".",$5,$6}' H3K27ac.4.sorted.uniq.bed > H3K27ac.4.tagAlign
awk 'BEGIN{OFS="\t"}{print $1,$2,$3,".",$5,$6}' H3K27me3.4.sorted.uniq.bed > H3K27me3.4.tagAlign

# Fixing the H3K9me3 without score column
awk 'BEGIN{OFS="\t"}{print $1,$2,$3,".","1",$5}' H3K9me3.4.sorted.uniq.bed > H3K9me3.4.tagAlign


### Run the fragment length findings ###
run_spp.R -c=H3K4me3.4.tagAlign -p=4 -savp -out=H3K4me3.4.spp.out
# Top 3 estimates for fragment length 0,810 -> choose 810 -> 405

run_spp.R -c=H3K9ac.4.tagAlign -p=4 -savp -out=H3K9ac.4.spp.out
# Top 3 estimates for fragment length 0,560,640 -> choose 560 -> 280

run_spp.R -c=H3K9me3.4.tagAlign -p=4 -savp -out=H3K9me3.4.spp.out
# Top 3 estimates for fragment length -15,290,315 -> 290 -> 185

run_spp.R -c=H3K27ac.4.tagAlign -p=4 -savp -out=H3K27ac.4.spp.out
# Top 3 estimates for fragment length 245 -> choose 245 -> 122

run_spp.R -c=H3K27me3.4.tagAlign -p=4 -savp -out=H3K27me3.4.spp.out
# Top 3 estimates for fragment length 220,310,375 -> choose 310 -> 155


### Call Peaks ###
mkdir peaks_out

# H3K4me3
macs2 callpeak \
    -t H3K4me3.4.tagAlign \
    -c Input.4.tagAlign \
    -f BED \
    -g hs \
    -n H3K4me3 \
    --nomodel \
    --outdir peaks_out \
    -q 0.01 \
    --broad \
    --broad-cutoff 0.1 \
    --shift -405 \
    --extsize 810

# H3K9ac
macs2 callpeak \
    -t H3K9ac.4.tagAlign \
    -c Input.4.tagAlign \
    -f BED \
    -g hs \
    -n H3K9ac \
    --nomodel \
    --outdir peaks_out \
    -q 0.01 \
    --broad \
    --broad-cutoff 0.1 \
    --shift -280 \
    --extsize 560

# H3K9me3
macs2 callpeak \
    -t H3K9me3.4.tagAlign \
    -c Input.4.tagAlign \
    -f BED \
    -g hs \
    -n H3K9me3 \
    --nomodel \
    --outdir peaks_out \
    -q 0.01 \
    --broad \
    --broad-cutoff 0.1 \
    --shift -145 \
    --extsize 290

# H3K27ac
macs2 callpeak \
    -t H3K27ac.4.tagAlign \
    -c Input.4.tagAlign \
    -f BED \
    -g hs \
    -n H3K27ac \
    --nomodel \
    --outdir peaks_out \
    -q 0.01 \
    --broad \
    --broad-cutoff 0.1 \
    --shift -122 \
    --extsize 245

# H3K27me3
macs2 callpeak \
    -t H3K27me3.4.tagAlign \
    -c Input.4.tagAlign \
    -f BED \
    -g hs \
    -n H3K27me3 \
    --nomodel \
    --outdir peaks_out \
    -q 0.01 \
    --broad \
    --broad-cutoff 0.1 \
    --shift -155 \
    --extsize 310


## Doing the intersection with the gene expressions
bedtools intersect -a ../healthy_liver.bed -b H3K4me3_peaks.gappedPeak -wa -wb > H3K4me3.bed
bedtools intersect -a ../healthy_liver.bed -b H3K9ac_peaks.gappedPeak -wa -wb > H3K9ac.bed
bedtools intersect -a ../healthy_liver.bed -b H3K9me3_peaks.gappedPeak -wa -wb > H3K9me3.bed
bedtools intersect -a ../healthy_liver.bed -b H3K27ac_peaks.gappedPeak -wa -wb > H3K27ac.bed
bedtools intersect -a ../healthy_liver.bed -b H3K27me3_peaks.gappedPeak -wa -wb > H3K27me3.bed









