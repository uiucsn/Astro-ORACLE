import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 
import networkx as nx

from tensorflow import keras
from tqdm import tqdm
from pathlib import Path
from typing import List
from tensorflow.keras.utils import pad_sequences

from astroOracle.taxonomy import get_taxonomy_tree, source_node_label
from astroOracle.interpret_results import get_conditional_probabilites
from astroOracle.dataloader import get_static_features, ts_length, ts_flag_value
from astroOracle.LSST_Source import LSST_Source

flux_scaling_const = 1000
time_scaling_const = 100
pb_wavelengths = {
    'u': (320 + 400) / (2 * 1000),
    'g': (400 + 552) / (2 * 1000),
    'r': (552 + 691) / (2 * 1000),
    'i': (691 + 818) / (2 * 1000),
    'z': (818 + 922) / (2 * 1000),
    'Y': (950 + 1080) / (2 * 1000),
}

class ORACLE():

    def __init__(self, model_path=Path("models/lsst_alpha_0.5/best_model.h5")):
        
        """
        Initialize the ORACLE class.
        """
        self.model_path = model_path
        self.model = keras.models.load_model(self.model_path, compile=False)
        print(f"Model loaded from {model_path}")

        self.tree = get_taxonomy_tree()

    def prep_dataframes(self, x_ts_list:List[pd.DataFrame]):

        # Assert that columns names are correct

        augmented_arrays = []

        for ind in tqdm(range(len(x_ts_list)), desc ="TS Processing: "):

            df = x_ts_list[ind]

            # Scale the flux and flux error values
            df['scaled_FLUXCAL'] = df['FLUXCAL'] / flux_scaling_const
            df['scaled_FLUXCALERR'] = df['FLUXCALERR']/ flux_scaling_const

            # Subtract off the time of first observation and divide by scale factor
            df['scaled_time_since_first_obs'] = df['MJD'] / time_scaling_const

            # Remove saturations
            saturation_mask = np.where((df['PHOTFLAG'] & 1024) == 0)[0]
            df = df.iloc[saturation_mask].copy()

            # 1 if it was a detection, zero otherwise
            df.loc[:,'detection_flag'] = np.where((df['PHOTFLAG'] & 4096 != 0), 1, 0)

            # Encode pass band information correctly 
            df['band_label'] = [pb_wavelengths[pb] for pb in df['BAND']]
            
            df = df[['scaled_time_since_first_obs', 'detection_flag', 'scaled_FLUXCAL', 'scaled_FLUXCALERR', 'band_label']]
            
            # Truncate array if too long
            arr = df.to_numpy()
            if arr.shape[0]>ts_length:
                arr = arr[:ts_length, :]

            augmented_arrays.append(arr)
            
        augmented_arrays = pad_sequences(augmented_arrays, maxlen=ts_length,  dtype='float32', padding='post', value=ts_flag_value)

        return augmented_arrays

    def prep_static_features(self, x_static_list:List[pd.DataFrame]):

        for i in tqdm(range(len(x_static_list)), desc ="Static Processing: "):        
            x_static_list[i] = get_static_features(x_static_list[i])
        
        if len(x_static_list) == 1:
            x_static_list = np.expand_dims(x_static_list[0], axis=0)
        else:
            x_static_list = np.squeeze(x_static_list)
        return x_static_list

    def predict(self, x_ts_list:List[pd.DataFrame], x_static_list:List[pd.DataFrame]):

        # Prep data
        x_ts_tensors = self.prep_dataframes(x_ts_list)
        x_static_tensors = self.prep_static_features(x_static_list)
    
        # Make predictions
        logits = self.model.predict([x_ts_tensors, x_static_tensors], verbose=0)
        _, pseudo_conditional_probabilities = get_conditional_probabilites(logits, self.tree)

        level_order_nodes = nx.bfs_tree(self.tree, source=source_node_label).nodes()
        columns_names =  list(level_order_nodes)

        df = pd.DataFrame(pseudo_conditional_probabilities, columns=columns_names)
        
        return df
    
    def predict_classes(self, x_ts_list:List[pd.DataFrame], x_static_list:List[pd.DataFrame]):

        probs_df = self.predict(x_ts_list, x_static_list)

        # Find the leaf nodes
        leaf_nodes = np.array([x for x in self.tree.nodes() if self.tree.out_degree(x)==0 and self.tree.in_degree(x)==1])
        
        # Get the probabilities for the leaf nodes
        probs_df = probs_df[leaf_nodes]
        class_idx = np.argmax(probs_df.to_numpy(), axis=1)

        # Get the class labels
        class_labels = leaf_nodes[class_idx]

        return class_labels
        
class ORACLE_lite():

    def __init__(self, model_path=Path("models/lsst_alpha_0.5_no_md/best_model.h5")):
        
        """
        Initialize the ORACLE class.
        """
        self.model_path = model_path
        self.model = keras.models.load_model(self.model_path, compile=False)
        print(f"Model loaded from {model_path}")

        self.tree = get_taxonomy_tree()

    def prep_dataframes(self, x_ts_list:List[pd.DataFrame]):

        # Assert that columns names are correct

        augmented_arrays = []

        for ind in tqdm(range(len(x_ts_list)), desc ="TS Processing: "):

            df = x_ts_list[ind]

            # Scale the flux and flux error values
            df['scaled_FLUXCAL'] = df['FLUXCAL'] / flux_scaling_const
            df['scaled_FLUXCALERR'] = df['FLUXCALERR']/ flux_scaling_const

            # Subtract off the time of first observation and divide by scale factor
            df['scaled_time_since_first_obs'] = df['MJD'] / time_scaling_const

            # Remove saturations
            saturation_mask = np.where((df['PHOTFLAG'] & 1024) == 0)[0]
            df = df.iloc[saturation_mask].copy()

            # 1 if it was a detection, zero otherwise
            df.loc[:,'detection_flag'] = np.where((df['PHOTFLAG'] & 4096 != 0), 1, 0)

            # Encode pass band information correctly 
            df['band_label'] = [pb_wavelengths[pb] for pb in df['BAND']]
            
            df = df[['scaled_time_since_first_obs', 'detection_flag', 'scaled_FLUXCAL', 'scaled_FLUXCALERR', 'band_label']]
            
            # Truncate array if too long
            arr = df.to_numpy()
            if arr.shape[0]>ts_length:
                arr = arr[:ts_length, :]

            augmented_arrays.append(arr)
            
        augmented_arrays = pad_sequences(augmented_arrays, maxlen=ts_length,  dtype='float32', padding='post', value=ts_flag_value)

        return augmented_arrays


    def predict(self, x_ts_list:List[pd.DataFrame]):

        # Prep data
        x_ts_tensors = self.prep_dataframes(x_ts_list)

        # Making up fake data for the disconnected branch. This could be anything
        x_static_tensors = np.zeros((x_ts_tensors.shape[0], 23))
    
        # Make predictions
        logits = self.model.predict([x_ts_tensors, x_static_tensors], verbose=0)
        _, pseudo_conditional_probabilities = get_conditional_probabilites(logits, self.tree)

        level_order_nodes = nx.bfs_tree(self.tree, source=source_node_label).nodes()
        columns_names =  list(level_order_nodes)

        df = pd.DataFrame(pseudo_conditional_probabilities, columns=columns_names)
        
        return df
    
    def predict_classes(self, x_ts_list:List[pd.DataFrame]):

        probs_df = self.predict(x_ts_list)

        # Find the leaf nodes
        leaf_nodes = np.array([x for x in self.tree.nodes() if self.tree.out_degree(x)==0 and self.tree.in_degree(x)==1])
        
        # Get the probabilities for the leaf nodes
        probs_df = probs_df[leaf_nodes]
        class_idx = np.argmax(probs_df.to_numpy(), axis=1)

        # Get the class labels
        class_labels = leaf_nodes[class_idx]

        return class_labels
        
                                          

def main():

    m =  ORACLE()
    print(m)