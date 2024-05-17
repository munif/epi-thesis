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


SELECT h.chrom AS hepg2_chrom, h.chromStart, h.chromEnd, h.gene,
		n.chrom ncbi_chrom, n.txStart, n.txEnd, n.name2 
FROM common.hepg2_exp_transformed h, hg19.ncbirefseq n
WHERE h.chrom = n.chrom 
AND h.chromStart = n.txStart 
AND h.chromEnd = n.txEnd
AND h.gene <> n.name2 


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

SELECT    chrom
        , chromStart
        , chromEnd
        , gene
        , test_id
        , gene_id
        , locus
        , sample_1
        , sample_2
        , status
        , value_1
        , value_2
        , `log2(fold_change)`
        , test_stat
        , p_value
        , q_value
        , significant
FROM common.hepg2_exp_transformed;



-- Preparing data with the same chromosome, start and end
SELECT  
          h.chrom               AS h_chrom
        , CASE
	        WHEN n.strand = '+' THEN n.txStart - 2000
        	WHEN n.strand = '-' THEN n.txEnd - 2000
        END AS tss_start
        , CASE
	        WHEN n.strand = '+' THEN n.txStart + 2000
        	WHEN n.strand = '-' THEN n.txEnd + 
        	2000
        END AS tss_end
        , h.chromStart          AS h_chromStart
        , h.chromEnd            AS h_chromEnd
        , h.gene                AS h_gene
        , h.test_id             AS h_test_id
        , h.gene_id             AS h_gene_id
        , h.locus               AS h_locus
        , h.sample_1            AS h_sample_1
        , h.sample_2            AS h_sample_2
        , h.status              AS h_status
        , h.value_1             AS h_value_1
        , h.value_2             AS h_value_2
        , h.`log2(fold_change)` AS `h_log2(fold_change)`
        , h.test_stat           AS h_test_stat
        , h.p_value             AS h_p_value
        , h.q_value             AS h_q_value
        , h.significant         AS h_significant
        , n.chrom             AS n_chrom
        , n.txStart           AS n_txStart
        , n.txEnd             AS n_txEnd
        , n.name2             AS n_name2
        , n.score             AS n_score
        , n.bin               AS n_bin
        , n.name              AS n_name
        , n.strand            AS n_strand
        , n.cdsStart          AS n_cdsStart
        , n.cdsEnd            AS n_cdsEnd
        , n.exonCount         AS n_exonCount
        , n.exonStarts        AS n_exonStarts
        , n.exonEnds          AS n_exonEnds
        , n.cdsStartStat      AS n_cdsStartStat
        , n.cdsEndStat        AS n_cdsEndStat
        , n.exonFrames        AS n_exonFrames
        , CASE
	        WHEN n.strand = '+' THEN n.txStart 
        	WHEN n.strand = '-' THEN n.txEnd 
        END AS tss
FROM common.hepg2_exp_transformed h
INNER JOIN hg19.ncbirefseq n
ON  h.chrom         = n.chrom
AND h.chromStart    = n.txStart
AND h.chromEnd      = n.txEnd

-- Table for histone peak (from bedfile)
CREATE TABLE common.`h3k4me3` (
  `chrom` varchar(100) NOT NULL,
  `chromStart` int(11) NOT NULL,
  `chromEnd` int(11) NOT NULL,
  `name` varchar(100) NOT NULL,
  KEY `h3k4me3_chrom_IDX` (`chrom`,`chromStart`,`chromEnd`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

CREATE TABLE common.`H3K9ac` (
  `chrom` varchar(100) NOT NULL,
  `chromStart` int(11) NOT NULL,
  `chromEnd` int(11) NOT NULL,
  `name` varchar(100) NOT NULL,
  KEY `H3K9ac_chrom_IDX` (`chrom`,`chromStart`,`chromEnd`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;


CREATE TABLE common.`H3K9me3` (
  `chrom` varchar(100) NOT NULL,
  `chromStart` int(11) NOT NULL,
  `chromEnd` int(11) NOT NULL,
  `name` varchar(100) NOT NULL,
  KEY `H3K9me3_chrom_IDX` (`chrom`,`chromStart`,`chromEnd`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;


CREATE TABLE common.`H3K27ac` (
  `chrom` varchar(100) NOT NULL,
  `chromStart` int(11) NOT NULL,
  `chromEnd` int(11) NOT NULL,
  `name` varchar(100) NOT NULL,
  KEY `H3K27ac_chrom_IDX` (`chrom`,`chromStart`,`chromEnd`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

CREATE TABLE common.`H3K27me3` (
  `chrom` varchar(100) NOT NULL,
  `chromStart` int(11) NOT NULL,
  `chromEnd` int(11) NOT NULL,
  `name` varchar(100) NOT NULL,
  KEY `H3K27me3_chrom_IDX` (`chrom`,`chromStart`,`chromEnd`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;


-- Checking the histone count result
-- chr1	203272453	203276453

SELECT *
FROM common.h3k4me3 
WHERE chrom = 'chr1'	
AND chromStart  >= 203272453
AND chromEnd <= 203276453
UNION 
SELECT *
FROM common.h3k9ac
WHERE chrom = 'chr1'	
AND chromStart  >= 203272453
AND chromEnd <= 203276453
UNION 
SELECT *
FROM common.h3k9me3  
WHERE chrom = 'chr1'	
AND chromStart  >= 203272453
AND chromEnd <= 203276453
UNION 
SELECT *
FROM common.H3K27ac  
WHERE chrom = 'chr1'	
AND chromStart  >= 203272453
AND chromEnd <= 203276453
UNION 
SELECT *
FROM common.H3K27me3  
WHERE chrom = 'chr1'	
AND chromStart  >= 203272453
AND chromEnd <= 203276453

-- Table for saving the histone count

CREATE TABLE common.histone_count
(
	      h_chrom					VARCHAR(100) NOT NULL
        , tss_start					INTEGER NOT NULL
        , tss_end					INTEGER NOT NULL
        , h_chromStart				INTEGER NOT NULL
        , h_chromEnd				INTEGER NOT NULL
        , h_gene					VARCHAR(100) NOT NULL
        , h_test_id					VARCHAR(100) NOT NULL
        , h_gene_id					VARCHAR(100) NOT NULL
        , h_locus					VARCHAR(100) NOT NULL
        , h_sample_1				VARCHAR(100) NOT NULL
        , h_sample_2				VARCHAR(100) NOT NULL
        , h_status					VARCHAR(100) NOT NULL
        , h_value_1					DOUBLE NOT NULL
        , h_value_2					DOUBLE NOT NULL
        , `h_log2(fold_change)`		VARCHAR(100) NOT NULL
        , h_test_stat				DOUBLE NOT NULL
        , h_p_value					DOUBLE NOT NULL
        , h_q_value					DOUBLE NOT NULL
        , h_significant				VARCHAR(100) NOT NULL
        , n_chrom					VARCHAR(100) NOT NULL
        , n_txStart					INTEGER NOT NULL
        , n_txEnd					INTEGER NOT NULL
        , n_name2					VARCHAR(100) NOT NULL
        , n_score					INTEGER NOT NULL
        , n_bin						SMALLINT NOT NULL
        , n_name					VARCHAR(100) NOT NULL
        , n_strand					CHAR(1) NOT NULL
        , n_cdsStart				INTEGER NOT NULL
        , n_cdsEnd					INTEGER NOT NULL
        , n_exonCount				INTEGER NOT NULL
        , n_exonStarts				INTEGER NOT NULL
        , n_exonEnds				INTEGER NOT NULL
        , n_cdsStartStat			ENUM('none','unk','incmpl','cmpl') NOT NULL
        , n_cdsEndStat				enum('none','unk','incmpl','cmpl') NOT NULL	
        , n_exonFrames				LONGBLOB NOT NULL
        , tss						INTEGER NOT NULL
        , H3K4me3					INTEGER NOT NULL
        , H3K9ac					INTEGER NOT NULL
        , H3K9me3					INTEGER NOT NULL
        , H3K27ac					INTEGER NOT NULL
        , H3K27me3					INTEGER NOT NULL
);

-- Import the data

-- Then alter the table by adding a new total column
ALTER TABLE common.histone_count 
ADD COLUMN histone_total_count INTEGER NOT NULL;

-- Calculate the histone total count
UPDATE common.histone_count 
SET histone_total_count = H3K4me3 + H3K9ac + H3K9me3 + H3K27ac + H3K27me3 

-- Select the overlap with different start or end
SELECT *
FROM hg19.hepg2_ncbi_hg19_overlap
WHERE hepg2_chromStart <> ncbi_txStart 
OR hepg2_chromEnd <> ncbi_txEnd 



SELECT hepg2_chrom, hepg2_name, hepg2_chromStart, ncbi_name, ncbi_txStart, 
	   hepg2_chromStart - ncbi_txStart AS diff_start,
	   hepg2_chromEnd, ncbi_txEnd,
	   hepg2_chromEnd - ncbi_txEnd AS diff_end,
	   (hepg2_chromEnd - hepg2_chromStart) - (ncbi_txEnd - ncbi_txStart) AS diff_len
FROM hg19.hepg2_ncbi_hg19_overlap hnho 
WHERE (hepg2_chromEnd - hepg2_chromStart) - (ncbi_txEnd - ncbi_txStart) = 0


-- https://www.ncbi.nlm.nih.gov/datasets/gene/GCF_000001405.40/?search=ALB
-- NC_000004.12:73404287-73421482 --> hg38

-- https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/genes/
-- https://www.ncbi.nlm.nih.gov/datasets/gene/GCF_000001405.25/?search=ALB
-- NC_000004.11:74270004-74287199 --> hg19 used in NCBI Browser ?


