import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Masking, concatenate

from dataloader import ts_length, static_flag_value, ts_flag_value

def get_ensemble_model(ts_dim, static_dim, output_dim, models):

    input_1 = Input((ts_length, ts_dim), name='light curve input') 
    input_2 = Input(shape = (static_dim, ), name='static features input') 

    # Run inference on all models and then concatenate the outputs from all the models
    models_results = [model([input_1, input_2], training=False) for model in models]
    merged_results = concatenate(models_results) 

    output = Dense(output_dim)(merged_results)

    model = keras.Model(inputs=[input_1, input_2], outputs=output)
    
    return model

class EnsembleModel(tf.keras.Model):
    def __init__(self, models, output_dim, **kwargs):
        super(EnsembleModel, self).__init__(**kwargs)
        self.models = models #where you've made this a list of your individual models
        self.dense_output = Dense(output_dim)

    def call(self, inputs):
        input_1, input_2 = inputs
        results = [model([input_1, input_2], training=False) for model in self.models]
        merged_results = concatenate(results)
        return self.dense_output(merged_results)
