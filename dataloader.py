import os
import torch
import math

import numpy as np
import polars as pl

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import tqdm

from LSTM_model import LSTMClassifier
from LSST_Source import LSST_Source
from taxonomy import get_classification_labels, get_astrophysical_class, get_taxonomy_tree

# All samples in the same batch need to have consistent sequence length. This adds padding for sequences shorter than sequence_length and truncates light sequences longer than sequence_length 
sequence_length = 500

# Value used for padding tensors to make them the correct length
padding_constant = 0

def reduce_length_uniform(np_ts):

    ts_length = np_ts.shape[0]

    # Fraction of ts to retain
    new_ts_length = int(math.ceil(np.random.uniform(low=0, high=1) * ts_length))
    np_ts = np_ts[:new_ts_length, ]

    return np_ts

class LSSTSourceDataSet(Dataset):


    def __init__(self, path, length_transform=None):
        """
        Arguments:
            path (string): Directory with all the astropy tables.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        print(f'Loading parquet dataset: {path}', flush=True)

        self.path = path
        self.parquet = pl.read_parquet(path)
        self.num_sample = self.parquet.shape[0]
        self.length_transform = length_transform

        print(f"Number of sources: {self.num_sample}")

    def __len__(self):

        return self.num_sample

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.parquet[idx]
        source = LSST_Source(row)
        table = source.get_event_table()

        astrophysical_class = get_astrophysical_class(source.ELASTICC_class)
        _, class_labels = get_classification_labels(astrophysical_class)

        ts_np = table.to_pandas().to_numpy()

        # Shorten the length of the time series data so the classifier learn to classify partial phase light curves
        if self.length_transform:
            ts_np = self.length_transform(ts_np)
            

        if ts_np.shape[0] < sequence_length:

            final_seq_length = ts_np.shape[0] 
            padding_length = sequence_length - ts_np.shape[0] 

            # If the length of the TS is less than sequence_length, add padding
            ts_np = np.pad(ts_np, [(0, padding_length), (0, 0)], mode='constant', constant_values=padding_constant)

        elif ts_np.shape[0] >= sequence_length:

            final_seq_length = sequence_length

            # If the length of the TS is more than sequence_length, keep the first sequence_length number of data points
            ts_np = ts_np[:sequence_length, :]

        # Getting the static features from the table
        static_np = np.array(list(table.meta.values()))

        # Replace flag values like 999 and -9999 with -9
        static_np[static_np == -9999] = -9
        static_np[static_np == 999] = -9
        static_np[static_np == -999] = -9

        return ts_np, static_np, class_labels, final_seq_length
    
    def get_dimensions(self):

        idx = 0
        ts_np, static_np, class_labels, _ = self.__getitem__(idx)

        dims = {
            'ts': ts_np.shape[1],
            'static': static_np.shape[0],
            'labels': class_labels.shape[0]
        }

        return dims
    
    def get_labels(self):

        ELASTICC_labels = self.parquet['ELASTICC_class']
        astrophysical_labels = []

        for idx in range(self.num_sample):

            elasticc_class = ELASTICC_labels[idx]
            astrophysical_class = get_astrophysical_class(elasticc_class)
            astrophysical_labels.append(astrophysical_class)
        
        return astrophysical_labels

if __name__=='__main__':
    
    # Simple test to verify data loader
    data_set = LSSTSourceDataSet('data/data/elasticc2_train/train_parquet.parquet', length_transform=reduce_length_uniform)

    print(data_set.get_dimensions())
    loader = DataLoader(data_set, shuffle=True, batch_size = 4)
    for i, (X_ts, X_static, Y, sequence_lengths) in enumerate(tqdm(loader)):
        pass
        #print(X_ts.shape, X_static.shape, Y.shape, sequence_lengths.shape)
