import pandas as pd
import numpy as np

# Please refer to BED file format
# https://en.wikipedia.org/wiki/BED_(file_format)

def load_data(file_name, type):
    columns = ['chrom', 'chromStart', 'chromEnd', 'name']
    df = pd.read_csv(file_name, sep="\t", header=None, names=columns)
    df.loc[:, "length"] = df["chromEnd"] - df["chromStart"]
    df.loc[:, "type"] = type
    return df

