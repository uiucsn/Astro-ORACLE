import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import Dataset, DataLoader

from LSTM_model import LSTMClassifier
from dataloader import LSSTSourceDataSet, reduce_length_uniform
from loss import WHXE_Loss
from taxonomy import get_taxonomy_tree

from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

def parse_args(argv=None):
    parser = ArgumentParser(
        prog='Train LSTM',
        description='Script to train LSTM model',
    )
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--num_dl_workers', type=int, default=4,
                        help='Number of workers for Dataloader.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--output_path', type=Path, default='models/LSTM.pt',
                        help='Save the weights here.')
    parser.add_argument('--train_path', type=Path, default='data/data/elasticc2_train/train_parquet.parquet',
                        help='Path to the training parquet file.')

    return parser.parse_args(argv)

def main(argv=None):

    # ML parameters
    args = parse_args(argv)
    learning_rate = args.lr
    batch_size = args.batch_size
    num_epochs = args.epochs
    num_dl_workers = args.num_dl_workers
    training_path = args.train_path

    output_path = args.output_path

    # Data loader for training
    data_set = LSSTSourceDataSet(training_path, length_transform=reduce_length_uniform)
    loader = DataLoader(data_set, shuffle=True, batch_size=batch_size, num_workers=num_dl_workers)

    # These might change - Should come from the LSST Source Tensor shapes.
    dims = data_set.get_dimensions()
    n_ts_features = dims['ts']
    n_static_features = dims['static']
    n_outputs = dims['labels']

    # Inputs for model
    ts_input_dim = n_ts_features
    static_input_dim = n_static_features
    output_dim = n_outputs

    lstm_hidden_dim = 64
    lstm_num_layers = 4

    # Initialize models
    model = LSTMClassifier(ts_input_dim, static_input_dim, lstm_hidden_dim, output_dim, lstm_num_layers)
    
    # Loss and optimizer
    tree = get_taxonomy_tree()
    loss_object = WHXE_Loss(tree, data_set.get_labels()) 
    criterion = loss_object.compute_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('Starting Training Loop...', flush=True)

    # Training loop
    for epoch in range(num_epochs):
        for i, (X_ts, X_static, labels, sequence_lengths) in enumerate(tqdm(loader)):
        
            pack = torch.nn.utils.rnn.pack_padded_sequence(X_ts, sequence_lengths, batch_first=True, enforce_sorted=False)
            # Forward pass
            outputs = model(pack.float(), X_static.float())
            loss = criterion(outputs, labels.float())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', flush=True)

        # Save the model. TODO change this to save the best model only.
        torch.save(model.state_dict(), output_path)

if __name__=='__main__':

    main()