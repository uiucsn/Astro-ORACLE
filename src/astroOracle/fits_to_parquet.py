#!/usr/bin/env python

from argparse import ArgumentParser
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
from sncosmo import read_snana_fits


def parse_args(argv=None):
    parser = ArgumentParser(
        prog='SNANA2parquet',
        description='Convert SNANA simulated light curves from FITS to parquet',
    )
    parser.add_argument('input', type=Path, 
                        help='Path to folder containing {HEAD,PHOT}.FITS.gz files')
    parser.add_argument('output', type=Path,
                        help='Output parquet file path, typically has .parquet extension')

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    head = sorted(args.input.glob('*HEAD.FITS.gz'))
    phot = sorted(args.input.glob('*PHOT.FITS.gz'))
    assert len(head) > 0
    assert len(head) == len(phot)

    lcs = list(chain.from_iterable(
        read_snana_fits(h, p) for h, p in zip(head, phot)
    ))

    # Fix band srtings - trim extra space
    for lc in lcs:
        lc['BAND'] = lc['BAND'].astype('U1')

    head_columns = list(lc.meta)
    phot_columns = list(lc.columns)

    df = pl.from_records(
        [
            lc.meta
            | {column: pl.Series(lc[column].newbyteorder().byteswap()) for column in lc.columns}
            for lc in lcs
        ],
    )
    count_lcs = len(lcs)
    del lcs

    # Make mandatory transformations
    df = df.with_columns(pl.col('SNID').cast(int))

    print(args.output, len(df), '/', count_lcs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output)

    return df, head_columns, phot_columns


if __name__ == '__main__':
    main()