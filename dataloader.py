import pickle
import numpy as np
import polars as pl

from tqdm import tqdm
from tensorflow.keras.utils import pad_sequences

from LSST_Source import LSST_Source
from taxonomy import get_classification_labels, get_astrophysical_class


ts_length = 500

# Features to be used in the model
static_feature_list = ['MWEBV', 'MWEBV_ERR', 'REDSHIFT_HELIO', 'REDSHIFT_HELIO_ERR', 'HOSTGAL_PHOTOZ', 'HOSTGAL_PHOTOZ_ERR', 'HOSTGAL_SPECZ', 'HOSTGAL_SPECZ_ERR', 'HOSTGAL_RA', 'HOSTGAL_DEC', 'HOSTGAL_SNSEP', 'HOSTGAL_ELLIPTICITY', 'HOSTGAL_MAG_u', 'HOSTGAL_MAG_g', 'HOSTGAL_MAG_r', 'HOSTGAL_MAG_i', 'HOSTGAL_MAG_z', 'HOSTGAL_MAG_Y', 'MW_plane_flag', 'ELAIS_S1_flag', 'XMM-LSS_flag', 'Extended_Chandra_Deep_Field-South_flag', 'COSMOS_flag']

# Flag values for missing data of static feature according to elasticc
missing_data_flags = [-9, -99, -999, -9999, 999]

# Flag value for masking used in the ML model
static_flag_value = -9
ts_flag_value = 0.

def load(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    
def augment_ts_length_to_days_since_trigger(X_ts, X_static, Y, a_classes, days):

    # Augment the length of the ts data
    X_ts = get_ts_upto_days_since_trigger(X_ts, days=days)

    # Squeeze data into homogeneously shaped numpy arrays
    X_ts = np.squeeze(X_ts)
    X_static = np.squeeze(X_static)
    Y = np.squeeze(Y).astype(np.float32)
    astrophysical_classes = np.squeeze(a_classes)

    return X_ts, X_static, Y, astrophysical_classes

def get_ts_upto_days_since_trigger(X_ts, days, add_padding=True):

    augmented_list = []

    # Loop through all the data
    for ind in tqdm(range(len(X_ts)), desc ="TS Augmentation: ", disable=True):

        times = X_ts[ind]['scaled_time_since_first_obs'].to_numpy()

        # Get the idx of the first detection
        first_detection_idx = np.where(X_ts[ind]['detection_flag'].to_numpy() == 1)[0][0]
        first_detection_t = times[first_detection_idx]

        if len(np.where((times - first_detection_t) * 100 <= days)[0]) == 0:
            augmented_list.append(np.zeros_like(X_ts[ind].to_numpy()))
        else:
            # Get the index of the the last observation between the mjd(first detection) and  mjd(first detection)
            last_observation_idx = np.where((times - first_detection_t) * 100 <= days)[0][-1]
            
            # Slice the data appropriately, Keep the first new_length number of observations and all columns
            augmented_list.append(X_ts[ind].to_numpy()[:(last_observation_idx + 1), :])

    # Optionally - Pad for TF masking layer
    if add_padding:
        augmented_list = pad_sequences(augmented_list, maxlen=ts_length,  dtype='float32', padding='post', value=ts_flag_value)

    return augmented_list



def augment_ts_length(X_ts, add_padding=True, fraction=None):

    augmented_list = []
    
    if fraction == None:
        fractions = np.random.rand(len(X_ts))
    else:
        fractions = [fraction] * len(X_ts)
        fractions = np.array([fractions])

    # Loop through all the data
    for ind in tqdm(range(len(X_ts)), desc ="TS Augmentation: "):

        # If no fraction is mentioned, pick a random number between 0 and 1
        if fraction == None:
            fraction = fractions[ind]

        # Multiply the fraction by the original length of the time series to get the new length
        new_length = int(fraction * X_ts[ind].to_numpy().shape[0])

        # Make sure there is at least 1 observation in the data
        new_length = max(1, new_length)
        
        # Slice the data appropriately, Keep the first new_length number of observations and all columns
        augmented_list.append(X_ts[ind].to_numpy()[:new_length, :])

    # Optionally - Pad for TF masking layer
    if add_padding:
        augmented_list = pad_sequences(augmented_list, maxlen=ts_length,  dtype='float32', padding='post', value=ts_flag_value)

    return augmented_list, fractions

def get_augmented_data(X_ts, X_static, Y, a_classes, fraction=None):

    # Augment the length of the ts data
    X_ts, fractions = augment_ts_length(X_ts, fraction=fraction)

    # Squeeze data into homogeneously shaped numpy arrays
    X_ts = np.squeeze(X_ts)
    X_static = np.squeeze(X_static)
    Y = np.squeeze(Y).astype(np.float32)
    astrophysical_classes = np.squeeze(a_classes)
    fractions = np.squeeze(fractions).astype(np.float32)

    return X_ts, X_static, Y, astrophysical_classes, fractions

def get_static_features(y, feature_list=static_feature_list):

    static_features = []

    # Get the necessary static features from the ordered dictionary
    for feature in feature_list:

        val = y[feature]
        if val in missing_data_flags:
            static_features.append(static_flag_value)
        else:
            static_features.append(val)

    return static_features

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
    
