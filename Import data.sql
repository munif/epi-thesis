LOAD DATA INFILE 'D:/Downloads/gwasCatalog.txt' INTO TABLE gwascatalog;
LOAD DATA INFILE 'D:/Downloads/hg38/GeneReviews/geneReviews.txt' INTO TABLE geneReviews;
LOAD DATA INFILE 'D:/Downloads/hg38/GeneReviews/geneReviewsDetail.txt' INTO TABLE genereviewsdetail;

LOAD DATA INFILE D:/Downloads/ncbiRefSeq.txt INTO TABLE ncbirefseq;


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


D:/Repo/hg38/database/