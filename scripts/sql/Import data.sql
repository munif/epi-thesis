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

