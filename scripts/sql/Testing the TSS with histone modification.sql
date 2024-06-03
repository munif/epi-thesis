SELECT 	h_chrom, h_chromStart, h_chromEnd, n_name2 , tss, tss_start, tss_end, h_value_1, 
		H3K4me3 , H3K9ac , H3K9me3 , H3K27ac, H3K27me3 
FROM common.histone_count_overlap80 hco
WHERE tss  = 17721238

-- WHERE H3K4me3 >= 10
-- ORDER BY H3K4me3 DESC, h_value_1 DESC 

SELECT min(h_value_1), max(h_value_1)
FROM common.histone_count_overlap80 hco 

SELECT * -- , chromEnd - chromStart  AS length
FROM common.h3k4me3 hkm
WHERE chromStart >= 17721238 - 2000
AND chromEnd <= 17721238 + 2000
AND chrom = 'chr2'


-- T: H3K4me3, H3K9ac, H3K27ac
-- D: H3K4me3, H3K9ac
-- A: H3K4me3

CREATE VIEW common.tda AS
SELECT 	H3K4me3 + H3K9ac + H3K27ac AS T,
		H3K4me3 + H3k9ac AS D,
		H3K4me3 AS A,
		h_value_1  AS value_1
FROM common.histone_count_overlap80 hco

SELECT 	H3K4me3 + H3K9ac + H3K27ac AS T,
		H3K4me3 + H3k9ac AS D,
		H3K4me3 AS A,
		h_value_1  AS value_1
FROM common.histone_count_overlap80 hco


-- Find the histone marker
SELECT 	h_chrom,
		tss,
		h_chromStart,
		h_chromEnd,
		h_value_1,
		H3K4me3,
		H3K9ac,
		H3K9me3,
		H3K27ac,
		H3K27me3,
		histone_total_count 
FROM common.histone_count_overlap80 hco 
WHERE tss = 17721238
-- tss = 9294832

SELECT *
FROM common.h3k4me3 hkm
WHERE chrom = 'chr1'
AND (
((chromStart >= 9294832 - 2000) AND (chromEnd <= 9294832 + 2000))
OR
((chromStart <= 9294832 - 2000) AND (chromEnd <= 9294832 + 2000) AND (chromEnd >= 9294832 - 2000) AND (((chromEnd - (9294832 - 2000))/146) >= 0.8))
)


SELECT *
FROM histone_count_overlap80 
WHERE h_chrom = 'chr1'
AND h_chromStart = 24741636