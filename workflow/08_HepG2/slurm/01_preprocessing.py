import polars as pl
from pybiomart import Server
import os
import numpy as np
import pandas as pd

WORKING_DIR = '/group/pmc021/amunif/epi-thesis/workflow/08_HepG2'
DATASET_DIR = '/group/pmc021/amunif/epi-thesis/dataset'

df = pl.read_csv(os.path.join(DATASET_DIR, 'GSM3718064_HepG2_exp.txt'), separator="\t")

# Split locus column into 3 columns: chr, start, end
df = df.with_columns([
    pl.col("locus").str.split(":").list.get(0).alias("chromosome"),
    pl.col("locus").str.split(":").list.get(1).str.split_exact("-", 1).struct.rename_fields(["start", "end"]).alias("position")
])

# Unnest the column and convert into integer
df = df.unnest("position")

df = df.with_columns([
    pl.col("start").cast(pl.Int64),
    pl.col("end").cast(pl.Int64)
])

df = df.with_columns([
    pl.col('chromosome').str.replace('chr', '').alias('chr')
])

exploded_df = df.with_columns([
    pl.col("gene").str.split(",")
]).explode("gene")

def query_ensembl_by_location(item):

    chrom = item[17]
    start = item[15]
    end = item[16]
    gene_name = item[2]
    test_id = item[0]
    locus = item[3]
    
    # Connect to the GRCh37 archive server
    server = Server(host='http://grch37.ensembl.org')
    
    # Select the human dataset
    dataset = (server.marts['ENSEMBL_MART_ENSEMBL']
               .datasets['hsapiens_gene_ensembl'])

    result = dataset.query(attributes=['chromosome_name', 
                                   'start_position', 
                                   'end_position', 
                                   'strand',
                                   'external_gene_name', 
                                   'ensembl_gene_id',
                                   'gene_biotype'],
                       filters={'chromosome_name': chrom,
                                'start': start,
                                'end': end})
    
    result['tss'] = result.apply(lambda row: row['Gene start (bp)'] if row['Strand'] == 1 else row['Gene end (bp)'], axis=1)
    result['tss -2kb'] = result['tss'] - 2000
    result['tss +2kb'] = result['tss'] + 2000
    result['test_id'] = test_id
    result['orig_gene_name'] = gene_name
    result['locus'] = locus
    result['orig_start'] = start
    result['orig_end'] = end
    return result

exploded_np = exploded_df.to_numpy()

output_path = os.path.join(WORKING_DIR, 'dataset', 'ensembl_slurm.csv')

i = 1
total = exploded_np.shape[0]

for item in exploded_np:

    if ((i % 100 == 0) or (i == total)):
        print(f"{i}/{total} - {item[0]} - {item[2]}")
    
    result = query_ensembl_by_location(item)
    result.to_csv(output_path, mode='a', header=(not os.path.exists(output_path)), index=False)
    i += 1