import pickle
import numpy as np
import pandas as pd

from LSTM_model import get_LSTM_Classifier
from dataloader import LSSTSourceDataSet, ts_length
#from loss import WHXE_Loss
from taxonomy import get_taxonomy_tree

from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

def save(save_path , obj):
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f)

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
    train_set = LSSTSourceDataSet(training_path)

    # These might change - Should come from the LSST Source Tensor shapes.
    dims = train_set.get_dimensions()
    n_ts_features = dims['ts']
    n_static_features = dims['static']
    n_outputs = dims['labels']

    # Inputs for model
    ts_dim = n_ts_features
    static_dim = n_static_features
    output_dim = n_outputs
    latent_size = 10

    # Initialize models
    model = get_LSTM_Classifier(ts_dim, static_dim, output_dim, latent_size)
    
    # Loss and optimizer
    tree = get_taxonomy_tree()
    # loss_object = WHXE_Loss(tree, train_set.get_labels()) 
    # criterion = loss_object.compute_loss

    print('Started loading data...', flush=True)

    X_ts = [] # info => for each time step, store time, median passband wavelength, flux, flux error
    X_static = [] # static features
    Y = [] # store labels
    astrophysical_classes = []
    elasticc_classes = []
    lengths = []

    for i in tqdm(range(train_set.get_len())):
    
        source, labels = train_set.get_item(i)
        table = source.get_event_table()

        meta_data = np.array(list(table.meta.values()))
        ts_data = pd.DataFrame(np.array(table)) # astropy table to pandas dataframe

        # Append data for ML
        X_ts.append(ts_data)
        X_static.append(meta_data)
        Y.append(labels)

        # Append other useful data
        astrophysical_classes.append(source.astrophysical_class)
        elasticc_classes.append(source.ELASTICC_class)
        lengths.append(ts_data.shape[0])

    # Pad for TF masking layer
    for ind in range(len(X_ts)):
        X_ts[ind] = np.pad(X_ts[ind], ((0, ts_length - len(X_ts[ind])), (0, 0)))


    # Split into train and validation
    X_ts_train, X_ts_val, X_static_train, X_static_val, Y_train, Y_val, astrophysical_classes_train, astrophysical_classes_val, elasticc_classes_train, elasticc_classes_val = train_test_split(X_ts, X_static, Y, astrophysical_classes, elasticc_classes, random_state = 40, test_size = 0.1)

    # Do some processing for tensorflow
    X_ts_train = np.squeeze(np.array(X_ts_train))
    X_ts_val = np.squeeze(np.array(X_ts_val))

    X_static_train = np.squeeze(np.array(X_static_train))
    X_static_val = np.squeeze(np.array(X_static_val))

    Y_train = np.squeeze(np.array(Y_train))
    Y_val = np.squeeze(np.array(Y_val))

    print('Start training...')
    early_stopping = EarlyStopping(
                              patience=5,
                              min_delta=0.001,                               
                              monitor="val_loss",
                              restore_best_weights=True
                              )

    try:
        history = model.fit(x = [X_ts_train, X_static_train],  y = Y_train, validation_data=([X_ts_val, X_static_val], Y_val), epochs=num_epochs, batch_size = batch_size, callbacks=[early_stopping])
    except Exception as e:
        print(e)

    model.save(f"models/RedshiftLatent_{latent_size}")
    save(f"models/RedshiftLatent_{latent_size}_history", history)

if __name__=='__main__':

    main()