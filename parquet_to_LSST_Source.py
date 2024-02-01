import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa

from LSST_Source import LSST_Source

def get_LSST_objects(parquet_file_path):

    df = pl.read_parquet(parquet_file_path)
    row = df[0]

    temp = LSST_Source(row, 'KN_B19')
    temp.plot_lightcurve()
    print(temp)
    

if __name__=='__main__':

    parquet_file_path = 'data/data/elasticc2_train/parquet/KN_B19.parquet'
    get_LSST_objects(parquet_file_path)
