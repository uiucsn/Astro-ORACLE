from argparse import ArgumentParser
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
from sncosmo import read_snana_fits

seed = 42


def parse_args(argv=None):
    parser = ArgumentParser(
        prog='Combine parquet',
        description='Combine all parquet files into one big parquet file',
    )
    parser.add_argument('input', type=Path, 
                        help='Path to folder containing individual parquet files')
    parser.add_argument('train_output', type=Path,
                        help='Output parquet file path for training, typically has .parquet extension')
    parser.add_argument('test_output', type=Path,
                        help='Output parquet file path for testing, typically has .parquet extension')
    parser.add_argument('--train_frac', type=Path, default=0.7,
                        help='Fraction of all data used for training')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    file_paths = sorted(args.input.glob('*.parquet'))

    # Read each parquet file into a Polars DataFrame.
    train_frames = []
    test_frames = []

    for file_path in file_paths:

        object_class = str(file_path).split('/')[-1].split('.')[0]
        print(f'Opening {object_class}', flush=True)

        # Load file, add label, and shuffle
        dataframe = pl.read_parquet(file_path)
        dataframe = dataframe.with_columns(pl.lit(object_class).alias('ELASTICC_class'))
        dataframe = dataframe.sample(fraction=1, shuffle=True, seed=seed)

        # Split intro train and test sets
        test_size = int((1 - args.train_frac) * dataframe.shape[0])
        test, train = dataframe.head(test_size), dataframe.tail(-test_size)

        print(f"Total DF: {dataframe.shape[0]} | Train DF: {train.shape[0]} | Test DF: {test.shape[0]}")

        train_frames.append(train)
        test_frames.append(test)

    # Combine the DataFrames into a single DataFrame.
    combined_train = pl.concat(train_frames, how='diagonal')
    combined_test = pl.concat(test_frames, how='diagonal')

    print(f"=========\nTrain {combined_train.shape[0]} | Test {combined_test.shape[0]}", flush=True)

    # Write the combined DataFrame to a new parquet file.
    combined_train.write_parquet(args.train_output)
    combined_test.write_parquet(args.test_output)

    print("Saved!")

if __name__ == '__main__':
    main()