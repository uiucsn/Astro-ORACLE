import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Masking, concatenate

from dataloader import ts_length, static_flag_value, ts_flag_value

def get_ensemble_model(output_dim, models):

    # Use the output of the pre trained models as input
    inputs = []
    for i, model in enumerate(models):
        inputs.append(Input((output_dim), name=f'model_{i}_output'))

    # Concatenate the outputs of the pre trained models and combine them.
    merged_results = concatenate(inputs) 
    output = Dense(output_dim)(merged_results)

    model = keras.Model(inputs=inputs, outputs=output)
    
    return model

