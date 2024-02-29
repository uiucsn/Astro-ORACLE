import os
import torch
import glob

import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from astropy.io import ascii

from LSTM_model import LSTMClassifier

# All samples in the same batch need to have consistent sequence length. This adds padding for sequences shorter than sequence_length and truncates light sequences longer than sequence_length 
sequence_length = 1000

class LSSTSourceDataSet(Dataset):


    def __init__(self, root_dir, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the astropy tables.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.file_names = glob.glob(f"{root_dir}/*")
        self.transform = transform

    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        table_path = os.path.join(self.file_names[idx])
        table = ascii.read(table_path)
        ts_np = table.to_pandas().to_numpy()

        # if len(table) < sequence_length:
        #     ts_np = np.add()
        # elif len(table) < sequence_length:
        #     ts_np = 

        sample = {'ts': ts_np, 'static': np.array(list(table.meta.values()))}

        if self.transform:
            sample = self.transform(sample)

        return np.random.rand(sequence_length, 10),sample['static']

if __name__=='__main__':
    
    # Simple test to verify data loader
    data_set = LSSTSourceDataSet('data/data/elasticc2_train/event_tables')
    loader = DataLoader(data_set, shuffle=True, batch_size = 4)
    for X_ts, X_static in loader:
        print(X_ts.shape, X_static.shape)
        break
