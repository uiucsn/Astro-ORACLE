import os
import torch
import glob
import math

import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from astropy.io import ascii

from LSTM_model import LSTMClassifier
from taxonomy import get_classification_labels, get_astrophysical_class

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


    def __init__(self, root_dir, length_transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the astropy tables.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.file_names = glob.glob(f"{root_dir}/*")
        self.length_transform = length_transform

    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        elasticc_class = ''.join(self.file_names[idx].split('/')[-1].split('.')[0].split('_')[1:])
        astrophysical_class = get_astrophysical_class(elasticc_class)
        _, class_labels = get_classification_labels(astrophysical_class)

        table_path = os.path.join(self.file_names[idx])
        table = ascii.read(table_path)
        ts_np = table.to_pandas().to_numpy()

        # Shorten the length of the time series data so the classifier learn to classify partial phase light curves
        if self.length_transform:
            ts_np = self.length_transform(ts_np)

        if ts_np.shape[0] < sequence_length:

            padding_length = sequence_length - ts_np.shape[0] 

            # If the length of the TS is less than sequence_length, add padding
            ts_np = np.pad(ts_np, [(padding_length, 0), (0, 0)], mode='constant', constant_values=padding_constant)

        elif ts_np.shape[0] < sequence_length:

            # If the length of the TS is more than sequence_length, keep the first sequence_length number of data points
            ts_np = ts_np[:sequence_length, :]

        # Getting the static features from the table
        static_np = np.array(list(table.meta.values()))

        # Replace flag values like 999 and -9999 with -9
        static_np[static_np == -9999] = -9
        static_np[static_np == 999] = -9

        return ts_np, static_np, class_labels
    
    def get_dimensions(self):

        idx = 0

        elasticc_class = self.file_names[idx].split('/')[-1].split('.')[0].split('_')[1]
        astrophysical_class = get_astrophysical_class(elasticc_class)
        _, class_labels = get_classification_labels(astrophysical_class)

        table_path = os.path.join(self.file_names[idx])
        table = ascii.read(table_path)
        ts_np = table.to_pandas().to_numpy()

        # Getting the static features from the table
        static_np = np.array(list(table.meta.values()))

        dims = {
            'ts': ts_np.shape[1],
            'static': static_np.shape[0],
            'labels': class_labels.shape[0]
        }

        return dims
    
    def get_labels(self):

        astrophysical_labels = []
        for idx in range(len(self.file_names)):

            elasticc_class = self.file_names[idx].split('/')[-1].split('.')[0].split('_')[1]
            astrophysical_class = get_astrophysical_class(elasticc_class)
            astrophysical_labels.append(astrophysical_class)
        
        return astrophysical_labels

if __name__=='__main__':
    
    # Simple test to verify data loader
    data_set = LSSTSourceDataSet('data/data/elasticc2_train/event_tables', length_transform=reduce_length_uniform)

    print(data_set.get_dimensions())
    loader = DataLoader(data_set, shuffle=True, batch_size = 4)
    for X_ts, X_static, Y in loader:
        print(X_ts.shape, X_static.shape, Y.shape)
        print(X_ts[1, :, :])
