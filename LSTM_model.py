import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense, Masking, concatenate, GRU

from dataloader import ts_length

def get_LSTM_Classifier(ts_dim, static_dim, output_dim, latent_size, loss_func):

    input_1 = Input((ts_length, ts_dim), name='light curve') 
    masking_input1 = Masking(mask_value=0.)(input_1)

    lstm1 = GRU(100, return_sequences=True, activation='tanh')(masking_input1)
    lstm2 = GRU(100, return_sequences=False, activation='tanh')(lstm1)

    dense1 = Dense(100, activation='tanh')(lstm2)

    input_2 = Input(shape = (static_dim, ), name='static features') # CHANGE

    dense2 = Dense(10)(input_2)

    merge1 = concatenate([dense1, dense2])

    dense3 = Dense(100, activation='relu')(merge1)

    dense4 = Dense(latent_size, activation='relu', name='latent')(dense3)

    output = Dense(output_dim)(dense4)

    model = keras.Model(inputs=[input_1, input_2], outputs=output)

    model.compile(loss = loss_func, optimizer="adam")
    
    return model

if __name__=='__main__':

    ts_dim = 5
    static_dim = 2
    latent_size = 64
    output_dim = 26

    batch_size = 4

    model = get_LSTM_Classifier(ts_dim, static_dim, output_dim, latent_size,  "categorical_crossentropy")

    input_ts = np.random.randn(batch_size, ts_length, ts_dim)
    input_static = np.random.randn(batch_size, static_dim)

    outputs = model.predict([input_ts, input_static])

    print('Input Sizes:', input_ts.shape, input_static.shape)
    print('Output Size:', outputs.shape)

    tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
    plt.show()

