import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class LSTMClassifier(nn.Module):

    def __init__(self, ts_input_dim, static_input_dim, lstm_hidden_dim, output_dim, lstm_num_layers, dropout=0.1): 

        super(LSTMClassifier, self).__init__()

        self.ts_input_dim = ts_input_dim
        self.static_input_dim = static_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.output_dim = output_dim
        self.lstm_num_layers = lstm_num_layers
        self.dropout = dropout

        # LSTM block to process some time series input
        self.lstm = nn.LSTM(input_size=ts_input_dim, hidden_size=lstm_hidden_dim, num_layers=lstm_num_layers, batch_first=True)
        
        # Fully connected layer(s) for merging LSTM output with static features
        # self.fc1 = nn.Linear(lstm_hidden_dim, 128)
        self.fc1 = nn.Linear(lstm_hidden_dim + static_input_dim, 128)


        # Layer to output the class probabilities.
        self.fc_final = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x_ts, x_static):
        
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.lstm_num_layers, x_static.size(0), self.lstm_hidden_dim).requires_grad_()
        c0 = torch.zeros(self.lstm_num_layers, x_static.size(0), self.lstm_hidden_dim).requires_grad_()

        # Pass the time series data through the LSTM
        lstm_out, (hn, cn) = self.lstm(x_ts, (h0.detach(), c0.detach()))
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Concatenate LSTM output with the static features
        concat_tensor = torch.concat((unpacked[:, -1, :], x_static), dim=1)

        # Pass the concatenated tensors through the fully connected layers
        concat_tensor = self.relu(concat_tensor)
        out = self.fc1(concat_tensor) 
        out = self.relu(out)
        out = self.fc_final(out)
        return out
    
if __name__=='__main__':

    learning_rate = 0.001
    ts_input_dim = 5
    static_input_dim = 5
    lstm_hidden_dim = 64
    output_dim = 7
    lstm_num_layers = 4
    batch_size = 4

    model = LSTMClassifier(ts_input_dim, static_input_dim, lstm_hidden_dim, output_dim, lstm_num_layers)

    ts_length = 5
    input_ts = torch.randn(batch_size, ts_length, ts_input_dim)
    input_static = torch.randn(batch_size, static_input_dim)

    outputs = model(input_ts, input_static)

    print('Input:', input_ts.size(), input_static.size())
    print('Output:', outputs.size())

