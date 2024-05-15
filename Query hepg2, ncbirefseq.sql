-- HepG2_exp = 22,460 rows
SELECT DISTINCT chrom, chromStart, chromEnd, gene 
FROM common.hepg2_exp_transformed h 
WHERE gene <> '-' 
ORDER BY chrom, chromStart, chromEnd ASC;

-- NCIB RefSeq hg19 = 46,468 rows
SELECT DISTINCT  chrom, txStart, txEnd, name2, score
FROM hg19.ncbirefseq n

-- NCIB RefSeq hg38 = 98,332 rows
SELECT DISTINCT  chrom, txStart, txEnd, name2, score
FROM hg38.ncbirefseq n

-- HepG2_exp with NCBI RefSeq hg19
CREATE TABLE hg19.hepg2_ncbi_hg19_overlap
(
	hepg2_chrom VARCHAR(50),
	hepg2_chromStart INT(11),
	hepg2_chromEnd INT(11),
	hepg2_name VARCHAR(100),
	ncbi_chrom VARCHAR(50),
	ncbi_txStart INT(11),
	ncbi_txEnd INT(11),
	ncbi_name VARCHAR(100),
	ncbi_score INTEGER
)

CREATE TABLE hg19.hepg2_ncbi_hg19_nonoverlap
(
	hepg2_chrom VARCHAR(50),
	hepg2_chromStart INT(11),
	hepg2_chromEnd INT(11),
	hepg2_name VARCHAR(100)
)

-- 24,071 rows
SELECT *
FROM hg19.hepg2_ncbi_hg19_overlap

-- 17,442 rows
SELECT DISTINCT hepg2_chrom, hepg2_chromStart, hepg2_chromEnd, hepg2_name 
FROM hg19.hepg2_ncbi_hg19_overlap


-- HepG2_exp with NCBI RefSeq hg38
CREATE TABLE hg38.hepg2_ncbi_hg38_overlap
(
	hepg2_chrom VARCHAR(50),
	hepg2_chromStart INT(11),
	hepg2_chromEnd INT(11),
	hepg2_name VARCHAR(100),
	ncbi_chrom VARCHAR(50),
	ncbi_txStart INT(11),
	ncbi_txEnd INT(11),
	ncbi_name VARCHAR(100),
	ncbi_score INTEGER
)

CREATE TABLE hg38.hepg2_ncbi_hg38_nonoverlap
(
	hepg2_chrom VARCHAR(50),
	hepg2_chromStart INT(11),
	hepg2_chromEnd INT(11),
	hepg2_name VARCHAR(100)
)

-- 23,986 rows
SELECT *
FROM hg38.hepg2_ncbi_hg38_overlap

-- 8,089 rows
SELECT DISTINCT hepg2_chrom, hepg2_chromStart, hepg2_chromEnd, hepg2_name 
FROM hg38.hepg2_ncbi_hg38_overlap

SELECT chrom, txStart, txEnd, name2
FROM hg19.ncbirefseq n 
WHERE name2 LIKE 'ALB%'
ORDER BY chrom, txStart, txEnd

UNION
SELECT chrom, chromStart, chromStart , gene
FROM common.hepg2_exp_transformed n 
WHERE gene LIKE 'ALB%'
ORDER BY chrom, chromStart, chromStart

SELECT *
FROM hg38.ncbirefseq n 
WHERE txStart >= 13474052
AND txEnd <= 13477569

SELECT *
FROM hg19.hepg2_ncbi_hg19_overlap hnho 
WHERE ncbi_name LIKE 'ALB%'


SELECT count(*)
FROM (
SELECT h.chrom AS hepg2_chrom, h.chromStart, h.chromEnd, h.gene,
		n.chrom ncbi_chrom, n.txStart, n.txEnd, n.name2 
FROM common.hepg2_exp_transformed h, hg19.ncbirefseq n
WHERE h.chrom = n.chrom 
AND h.chromStart = n.txStart 
AND h.chromEnd = n.txEnd) table1


SELECT bin, name, chrom, strand, txStart, txEnd, cdsStart, cdsEnd, name2
from hg19.ncbirefseq
WHERE name2 = 'ALB'

-- Restructuring ncbirefseq and hepg2 for bedtools alignment

SELECT `chrom`
, `txStart`
, `txEnd`
, `name2`
, `score`
, `bin`
, `name`
, `strand`
, `cdsStart`
, `cdsEnd`
, `exonCount`
, `exonStarts`
, `exonEnds`
, `cdsStartStat`
, `cdsEndStat`
, `exonFrames`
FROM hg19.ncbirefseq n 

SELECT *
FROM common.hepg2_exp_transformed het 

