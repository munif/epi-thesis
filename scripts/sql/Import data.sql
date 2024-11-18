LOAD DATA INFILE 'D:/Downloads/gwasCatalog.txt' INTO TABLE gwascatalog;
LOAD DATA INFILE 'D:/Downloads/hg38/GeneReviews/geneReviews.txt' INTO TABLE geneReviews;
LOAD DATA INFILE 'D:/Downloads/hg38/GeneReviews/geneReviewsDetail.txt' INTO TABLE genereviewsdetail;

LOAD DATA INFILE D:/Downloads/ncbiRefSeq.txt INTO TABLE ncbirefseq;

LOAD DATA INFILE 'D:/xampp/mysql/bin/ncbiRefSeq_hg38.txt' INTO TABLE ncbirefseq;

LOAD DATA INFILE 'D:/xampp/mysql/bin/ncbiRefSeq_hg38.txt' INTO TABLE ncbirefseq;

LOAD DATA INFILE 'D:/Repo/epi-thesis/dataset/hepg2_exp_transformed.csv' into table hepg2_exp_transformed;

mysql.exe -u root hg38 < all_mrna.sql
mysql.exe -u root hg38 < knownGene.sql
mysql.exe -u root hg38 < knownGeneExt.sql
mysql.exe -u root hg38 < knownGeneMrna.sql
mysql.exe -u root hg38 < knownToMrna.sql
mysql.exe -u root hg38 < knownToMrnaSingle.sql
mysql.exe -u root hg38 < mrnaOrientInfo.sql
mysql.exe -u root hg38 < spMrna.sql
mysql.exe -u root hg38 < tRNAs.sql
mysql.exe -u root hg38 < wgRna.sql
mysql.exe -u root hg38 < xenoMrna.sql


SELECT `COLUMN_NAME` 
FROM `INFORMATION_SCHEMA`.`COLUMNS` 
WHERE `TABLE_SCHEMA`='hg19' 
    AND `TABLE_NAME`='ncbirefseq';

SELECT 'bin', 'name', 'chrom', 'strand', 'txStart', 'txEnd', 'cdsStart', 'cdsEnd', 'exonCount', 'exonStarts', 'exonEnds', 'score', 'name2', 'cdsStartStat', 'cdsEndStat', 'exonFrames'
UNION
SELECT bin, name, chrom, strand, txStart, txEnd, cdsStart, cdsEnd, exonCount, exonStarts, exonEnds, score, name2, cdsStartStat, cdsEndStat, exonFrames

-- HepG2 exploded gene name

CREATE TABLE `gsm3718064_hepg2_exp_exploded_gene_name` (
  `test_id` varchar(50) DEFAULT NULL,
  `gene_id` varchar(50) DEFAULT NULL,
  `gene` varchar(50) DEFAULT NULL,
  `locus` varchar(50) DEFAULT NULL,
  `sample_1` varchar(50) DEFAULT NULL,
  `sample_2` varchar(50) DEFAULT NULL,
  `status` varchar(50) DEFAULT NULL,
  `value_1` double DEFAULT NULL,
  `value_2` double DEFAULT NULL,
  `log2(fold_change)` double DEFAULT NULL,
  `test_stat` double DEFAULT NULL,
  `p_value` double DEFAULT NULL,
  `q_value` double DEFAULT NULL,
  `significant` varchar(50) DEFAULT NULL,
  `chromosome` varchar(50) DEFAULT NULL,
  `start` int(11) DEFAULT NULL,
  `end` int(11) DEFAULT NULL,
  `chr` varchar(50) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;