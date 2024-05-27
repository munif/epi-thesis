-- Create table HepG2 overlap with NCBI RefSeq
CREATE TABLE hg19.hepg2_ncbi_overlap80
(
  `h_chrom` varchar(100) NOT NULL,
  `h_chromStart` int(11) NOT NULL,
  `h_chromEnd` int(11) NOT NULL,
  `h_gene` varchar(100) NOT NULL,
  `h_test_id` varchar(100) NOT NULL,
  `h_gene_id` varchar(100) NOT NULL,
  `h_locus` varchar(100) NOT NULL,
  `h_sample_1` varchar(100) NOT NULL,
  `h_sample_2` varchar(100) NOT NULL,
  `h_status` varchar(100) NOT NULL,
  `h_value_1` double NOT NULL,
  `h_value_2` double NOT NULL,
  `h_log2(fold_change)` varchar(100) NOT NULL,
  `h_test_stat` double NOT NULL,
  `h_p_value` double NOT NULL,
  `h_q_value` double NOT NULL,
  `h_significant` varchar(100) NOT NULL,
  `n_chrom` varchar(100) NOT NULL,
  `n_txStart` int(11) NOT NULL,
  `n_txEnd` int(11) NOT NULL,
  `n_name2` varchar(100) NOT NULL,
  `n_score` int(11) NOT NULL,
  `n_bin` smallint(6) NOT NULL,
  `n_name` varchar(100) NOT NULL,
  `n_strand` char(1) NOT NULL,
  `n_cdsStart` int(11) NOT NULL,
  `n_cdsEnd` int(11) NOT NULL,
  `n_exonCount` int(11) NOT NULL,
  `n_exonStarts` longblob NOT NULL,
  `n_exonEnds` longblob NOT NULL,
  `n_cdsStartStat` enum('none','unk','incmpl','cmpl') NOT NULL,
  `n_cdsEndStat` enum('none','unk','incmpl','cmpl') NOT NULL,
  `n_exonFrames` longblob NOT NULL
)

-- Select data as TSS

SELECT  
        h_chrom
        , CASE
	        WHEN n_strand = '+' THEN n_txStart - 2000
        	WHEN n_strand = '-' THEN n_txEnd - 2000
        END AS tss_start
        , CASE
	        WHEN n_strand = '+' THEN n_txStart + 2000
        	WHEN n_strand = '-' THEN n_txEnd + 2000
        END AS tss_end
        , h_chromStart
        , h_chromEnd
        , h_gene
        , h_test_id
        , h_gene_id
        , h_locus
        , h_sample_1
        , h_sample_2
        , h_status
        , h_value_1
        , h_value_2
        , `h_log2(fold_change)`
        , h_test_stat
        , h_p_value
        , h_q_value
        , h_significant
        , n_chrom
        , n_txStart
        , n_txEnd
        , n_name2
        , n_score
        , n_bin
        , n_name
        , n_strand
        , n_cdsStart
        , n_cdsEnd
        , n_exonCount
        , n_exonStarts
        , n_exonEnds
        , n_cdsStartStat
        , n_cdsEndStat
        , n_exonFrames
        , CASE
	        WHEN n_strand = '+' THEN n_txStart 
        	WHEN n_strand = '-' THEN n_txEnd 
        END AS tss
FROM hg19.hepg2_ncbi_overlap80



CREATE TABLE common.histone_count_overlap80
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
ALTER TABLE common.histone_count_overlap80 
ADD COLUMN histone_total_count INTEGER NOT NULL;

-- Calculate the histone total count
UPDATE common.histone_count_overlap80 
SET histone_total_count = H3K4me3 + H3K9ac + H3K9me3 + H3K27ac + H3K27me3 



SELECT 	h_chrom,
		H3K4me3,
  		H3K9ac,
  		H3K9me3,
  		H3K27ac,
  		H3K27me3,
  		histone_total_count,
  		h_value_1,
  		h_value_2
FROM common.histone_count_overlap80 hco 
WHERE 
		H3K4me3 > 0 AND
  		H3K9ac > 0 AND
  		H3K9me3 > 0 AND
  		H3K27ac > 0 AND
  		H3K27me3 > 0 AND 
		h_value_1 >= 1
ORDER BY histone_total_count DESC
