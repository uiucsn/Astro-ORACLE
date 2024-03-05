import os
import torch
import glob

import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from astropy.io import ascii

from LSTM_model import LSTMClassifier
from taxonomy import get_classification_labels, get_astrophysical_class

# All samples in the same batch need to have consistent sequence length. This adds padding for sequences shorter than sequence_length and truncates light sequences longer than sequence_length 
sequence_length = 500

# Value used for padding tensors to make them the correct length
padding_constant = -9

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
        
        elasticc_class = self.file_names[idx].split('/')[-1].split('.')[0].split('_')[1]
        astrophysical_class = get_astrophysical_class(elasticc_class)
        _, class_labels = get_classification_labels(astrophysical_class)

        table_path = os.path.join(self.file_names[idx])
        table = ascii.read(table_path)
        ts_np = table.to_pandas().to_numpy()

        # Number of rows in original time series data
        original_ts_sequence_length = ts_np.shape[0]
        
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

        return ts_np, static_np, class_labels

if __name__=='__main__':
    
    # Simple test to verify data loader
    data_set = LSSTSourceDataSet('data/data/elasticc2_train/event_tables')
    loader = DataLoader(data_set, shuffle=True, batch_size = 4)
    for X_ts, X_static, Y in loader:
        print(X_ts.shape, X_static.shape, Y.shape)

        print(X_ts[1, :, :])
        break
