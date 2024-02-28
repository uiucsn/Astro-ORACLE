import polars as pl

from pathlib import Path
from astropy.io import ascii

from LSST_Source import LSST_Source
from argparse import ArgumentParser

def parse_args(argv=None):
    parser = ArgumentParser(
        prog='Parquet2LSST_Source',
        description='Convert parquet files based on SNANA FITS to LSST_Source object tensors',
    )
    parser.add_argument('input', type=Path, 
                        help='Path to the parquet file')
    parser.add_argument('output', type=Path,
                        help='Output folder for astropy tables from the feature extraction of LSST_Source objects')

    return parser.parse_args(argv)

def main(argv=None):

    args = parse_args(argv)

    parquet_file_path = args.input
    table_file_path = args.output
    object_class = str(parquet_file_path).split('.')[0].split('/')[1]

    df = pl.read_parquet(parquet_file_path)
    print(f'Starting data augmentation for {parquet_file_path} Number of objects: {df.shape[0]} SNANA Feature length: {df.shape[1]}')

    for i in range(df.shape[0]):

        row = df[i]
        source = LSST_Source(row, object_class)
        source.plot_flux_curve()
        t  = source.get_event_table()

        # Make directory
        table_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the event table
        file_name = str(table_file_path) + f'/{source.SNID}_{source.ELASTICC_class}.ecsv'
        t.write(file_name, overwrite = False) 

    print(f"Completed conversion: {parquet_file_path}")
        

if __name__=='__main__':

    main()
