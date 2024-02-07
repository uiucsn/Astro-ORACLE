import polars as pl
from pathlib import Path

from LSST_Source import LSST_Source
from argparse import ArgumentParser

def parse_args(argv=None):
    parser = ArgumentParser(
        prog='Parquet2LSST_Source',
        description='Convert parquet files based on SNANA FITS to LSST_Source object tensors',
    )
    parser.add_argument('input', type=Path, 
                        help='Path to folder containing parquet files')
    # parser.add_argument('output', type=Path,
    #                     help='Output tensors from the feature extraction of LSST_Source objects')

    return parser.parse_args(argv)

def main(argv=None):

    args = parse_args(argv)

    parquet_file_path = args.input
    object_class = str(parquet_file_path).split('.')[0].split('/')[1]

    df = pl.read_parquet(parquet_file_path)
    print(f'Starting data augmentation for {parquet_file_path} Number of objects: {df.shape[0]} Feature length: {df.shape[1]}')

    for i in range(df.shape[0]):

        row = df[i]
        temp = LSST_Source(row, object_class)
        l = temp.get_augmented_sources()

    print(f"Completed conversion: {parquet_file_path}")
        

if __name__=='__main__':

    main()
