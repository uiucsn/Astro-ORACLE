import pickle
import os
import numpy as np
import pandas as pd

from dataloader import LSSTSourceDataSet

from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm


def save(save_path , obj):
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f)

def parse_args(argv=None):
    parser = ArgumentParser(
        prog='Prepare data arrays for ingesting',
        description='Convert parquet files into arrays for training and testing sets',
    )
    parser.add_argument('input_path', type=Path,
                        help='Path to the parquet file.')
    parser.add_argument('output_dir', type=Path, 
                        help='Save the weights here.')
    return parser.parse_args(argv)

def main(argv=None):

    # ML parameters
    args = parse_args(argv)
    input_path = args.input_path
    output_dir = args.output_dir

    # Data loader for training
    train_set = LSSTSourceDataSet(input_path)

    print('Loading data...\n', flush=True)

    X_ts = [] # info => for each time step, store time, median passband wavelength, flux, flux error
    X_static = [] # static features
    Y = [] # store labels
    astrophysical_classes = []
    elasticc_classes = []
    lengths = []

    for i in tqdm(range(train_set.get_len())):
    
        source, labels = train_set.get_item(i)
        table = source.get_event_table()

        meta_data = table.meta
        ts_data = pd.DataFrame(np.array(table)) # astropy table to pandas dataframe

        # Append data for ML
        X_ts.append(ts_data)
        X_static.append(meta_data)
        Y.append(labels)

        # Append other useful data
        astrophysical_classes.append(source.astrophysical_class)
        elasticc_classes.append(source.ELASTICC_class)
        lengths.append(ts_data.shape[0])
    
    print("\nDumping to pickle...")

    # Make a directory and save the data
    os.makedirs(f"{output_dir}", exist_ok=True)       

    save(f"{output_dir}/y.pkl", Y)
    save(f"{output_dir}/x_ts.pkl", X_ts)
    save(f"{output_dir}/x_static.pkl", X_static)
    save(f"{output_dir}/a_labels.pkl", astrophysical_classes)
    save(f"{output_dir}/e_label.pkl", elasticc_classes)
    save(f"{output_dir}/lengths.pkl", lengths)

if __name__=='__main__':

    main()