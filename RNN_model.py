import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense, Masking, concatenate, GRU

from dataloader import ts_length, static_flag_value, ts_flag_value

def get_RNN_model(ts_dim, static_dim, output_dim, latent_size):

    input_1 = Input((ts_length, ts_dim)) 
    masking_input1 = Masking(mask_value=ts_flag_value)(input_1)

    lstm1 = GRU(100, return_sequences=True, activation='tanh')(masking_input1)
    lstm2 = GRU(100, return_sequences=False, activation='tanh')(lstm1)

    dense1 = Dense(100, activation='tanh')(lstm2)

    input_2 = Input(shape = (static_dim,)) 
    masking_input2 = Masking(mask_value=static_flag_value)(input_2)

    dense2 = Dense(10)(masking_input2)

    merge1 = concatenate([dense1, dense2])

    dense3 = Dense(100, activation='relu')(merge1)

    dense4 = Dense(latent_size, activation='relu')(dense3)

    output = Dense(output_dim)(dense4)

    model = keras.Model(inputs=[input_1, input_2], outputs=output)
    
    return model

if __name__=='__main__':

    ts_dim = 5
    static_dim = 2
    latent_size = 64
    output_dim = 26

    batch_size = 4

    model = get_RNN_model(ts_dim, static_dim, output_dim, latent_size)
    print(model.summary())

    input_ts = np.random.randn(batch_size, ts_length, ts_dim)
    input_static = np.random.randn(batch_size, static_dim)

    outputs = model.predict([input_ts, input_static])

    print('Input Sizes:', input_ts.shape, input_static.shape)
    print('Output Size:', outputs.shape)

    tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
    plt.show()

