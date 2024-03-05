import numpy as np
import polars as pl

from pathlib import Path
from astropy.io import ascii

from LSST_Source import LSST_Source
from argparse import ArgumentParser

def parse_args(argv=None):
    parser = ArgumentParser(
        prog='Class summaries',
        description='Print Class summaries',
    )
    parser.add_argument('input', type=Path, 
                        help='Path to the parquet file')

    return parser.parse_args(argv)

def main(argv=None):

    args = parse_args(argv)

    parquet_file_path = args.input
    object_class = str(parquet_file_path).split('.')[0].split('/')[1]

    df = pl.read_parquet(parquet_file_path)

    ts_lengths = []
    

    for i in range(df.shape[0]):

        row = df[i]
        source = LSST_Source(row, object_class)
        if len(source.MJD) == 0:
            print(len(source.MJD), source.MJD)
        ts_lengths.append(len(source.MJD))

    print(f"{len(ts_lengths)}  {source.ELASTICC_class} objects: Min TS Length {min(ts_lengths)} | Mean TS Length {np.mean(ts_lengths)} | Max TS Length {max(ts_lengths)}")
        

if __name__=='__main__':

    main()
