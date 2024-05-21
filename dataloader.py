import pickle
import numpy as np
import polars as pl

from LSST_Source import LSST_Source
from taxonomy import get_classification_labels, get_astrophysical_class

ts_length = 500

def load(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def augment_ts_length(X_ts, fraction, add_padding=True):

    augmented_list = []

    # Loop through all the data
    for ind in range(len(X_ts)):

        print(f"{(ind/len(X_ts) * 100):.3f} %", end="\r")

        # Multiply the fraction by the original length of the time series to get the new length
        new_length = int(fraction * X_ts[ind].to_numpy().shape[0])

        # Make sure there is at least 1 observation in the data
        new_length = max(1, new_length)
        
        # Slice the data appropriately, Keep the first new_length number of observations and all columns
        augmented_list.append(X_ts[ind].to_numpy()[:new_length, :])

        # Optionally - Pad for TF masking layer
        if add_padding:
            augmented_list[ind] = np.pad(augmented_list[ind], ((0, ts_length - augmented_list[ind].shape[0]), (0, 0)))
        
    return augmented_list

def get_augmented_data(X_ts, X_static, Y, a_classes, fractions):

    # New lists to save the augmented data
    X_ts_aug = []
    X_static_aug = []
    Y_aug = []
    astrophysical_classes_aug = []
    lc_fraction_aug = []

    # Augment the length of the ts data
    for f in fractions:
        
        print(f"Augmenting light curve to {f * 100:.2f}%")
        
        X_ts_aug += augment_ts_length(X_ts, f)
        X_static_aug += X_static
        Y_aug += Y
        astrophysical_classes_aug += a_classes
        lc_fraction_aug += ([f] * len(X_ts))

    # Squeeze data into homogeneously shaped numpy arrays
    X_ts = np.squeeze(X_ts_aug)
    X_static = np.squeeze(X_static_aug)
    Y = np.squeeze(Y_aug)
    astrophysical_classes = np.squeeze(astrophysical_classes_aug)
    lc_fraction = np.squeeze(lc_fraction_aug)

    return X_ts, X_static, Y, astrophysical_classes, lc_fraction

class LSSTSourceDataSet():


    def __init__(self, path):
        """
        Arguments:
            path (string): Parquet file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        print(f'Loading parquet dataset: {path}', flush=True)

        self.path = path
        self.parquet = pl.read_parquet(path)
        self.num_sample = self.parquet.shape[0]

        print(f"Number of sources: {self.num_sample}")

    def get_len(self):

        return self.num_sample

    def get_item(self, idx):
        
        row = self.parquet[idx]
        source = LSST_Source(row)
        table = source.get_event_table()

        astrophysical_class = get_astrophysical_class(source.ELASTICC_class)
        _, class_labels = get_classification_labels(astrophysical_class)
        class_labels = np.array(class_labels)

        return source, class_labels
    
    def get_dimensions(self):

        idx = 0
        source, class_labels = self.get_item(idx)
        table = source.get_event_table()

        ts_np = table.to_pandas().to_numpy()
        static_np = np.array(list(table.meta.values()))

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
    data_set = LSSTSourceDataSet('data/data/elasticc2_train/test_parquet.parquet')
    print(data_set.get_dimensions())

    source, class_labels = data_set.get_item(0)
    table = source.get_event_table()
    print(source.astrophysical_class)
    print(table.meta)
    print(table)
    print(class_labels)
    
