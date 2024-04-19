import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense, Masking, concatenate, GRU

from dataloader import ts_length

def get_LSTM_Classifier(ts_dim, static_dim, output_dim, latent_size):

    input_1 = Input((ts_length, ts_dim), name='lc') 
    masking_input1 = Masking(mask_value=0.)(input_1)

    lstm1 = GRU(100, return_sequences=True, activation='tanh')(masking_input1)
    lstm2 = GRU(100, return_sequences=False, activation='tanh')(lstm1)

    dense1 = Dense(100, activation='tanh')(lstm2)

    input_2 = Input(shape = (static_dim, ), name='host_features') # CHANGE

    dense2 = Dense(10)(input_2)

    merge1 = concatenate([dense1, dense2])

    dense3 = Dense(100, activation='relu')(merge1)

    dense4 = Dense(latent_size, activation='relu', name='latent')(dense3)

    output = Dense(output_dim)(dense4)

    # TODO: add masked softmax here

    model = keras.Model(inputs=[input_1, input_2], outputs=output)

    model.compile(loss = "categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    
    return model

if __name__=='__main__':

    ts_dim = 5
    static_dim = 15
    latent_size = 10
    output_dim = 15

    batch_size = 4

    model = get_LSTM_Classifier(ts_dim, static_dim, output_dim, latent_size)

    input_ts = np.random.randn(batch_size, ts_length, ts_dim)
    input_static = np.random.randn(batch_size, static_dim)

    outputs = model.predict([input_ts, input_static])
    print(outputs)

    # print('Input:', input_ts.size(), input_static.size())
    # print('Output:', outputs.size())

