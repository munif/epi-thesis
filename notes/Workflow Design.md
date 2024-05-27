1. Do the alignment between HepG2 and NCBI Refseq to find the match gene
2. Calculate the +/-2k from TSS
3. Do the intersect counting between the hepg2_ncbi and every histone marker
4. Import to the database
5. Do the linear regression analysis
    Possible improvement:
    - scaling (?)
    - removing the outliers

- Data pre-procesing
    - value: average, single data (v1, v2)
    - chromosome -> one hot encoding ??

- Visualization
    - histone count + value

- Linear regression
    - histone count (5 columns of histone modification) -> value1, value2
    - add another features (chromosome, location, etc)


ALB
- HepG2 expression  = chr4:74269689-74287767
- NCBI RefSeq       = chr4:74270003-74287199