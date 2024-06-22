import argparse
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random 

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tqdm import tqdm
from pathlib import Path

from Ensemble_model import get_ensemble_model
from train_RNN import train_step
from dataloader import load, get_augmented_data, get_static_features, ts_length
from loss import WHXE_Loss
from taxonomy import get_taxonomy_tree

default_seed = 42
default_val_fraction = 0.01

default_num_epochs = 50
default_batch_size = 1024
default_learning_rate=5e-4

default_model_paths = []
default_train_dir = Path("processed/train")
default_model_dir = Path("models/ensemble")
default_max_class_count = 30000



def parse_args():
    '''
    Get commandline options
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=default_num_epochs, help='Number of epochs to train the model for.')
    parser.add_argument('--batch_size', type=int, default=default_batch_size, help='Batch size used for training.')
    parser.add_argument('--lr', type=float, default=default_learning_rate, help='Learning rate used for training.')
    parser.add_argument('--max_class_count', type=int, default=default_max_class_count, help='Maximum number of samples in each class.')
    parser.add_argument('--model_paths', nargs='+', default=default_model_paths, help='List of models to be used in the ensemble')
    parser.add_argument('--train_dir', type=Path, default=default_train_dir, help='Directory which contains the training data.')
    parser.add_argument('--model_dir', type=Path, default=default_model_dir, help='Directory for saving the models and best model during training.')

    args = parser.parse_args()
    return args

@tf.function
def train_step(pretrained_outputs, y, model, criterion, optimizer):
    with tf.GradientTape() as tape:
        logits = model(pretrained_outputs, training=True)
        loss_value = criterion(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss_value


def train_ensemble_model(models, num_epochs=default_num_epochs, batch_size=default_batch_size, learning_rate=default_learning_rate, max_class_count=default_max_class_count, train_dir=default_train_dir, model_dir=default_model_dir):

    random.seed(default_seed)
    os.mkdir(f"{model_dir}")

    # This step takes a while because it has load from disc to memory...
    print("Loading training data from disc...")
    X_ts = load(f"{train_dir}/x_ts.pkl")
    X_static = load(f"{train_dir}/x_static.pkl")
    Y = load(f"{train_dir}/y.pkl")
    astrophysical_classes = load(f"{train_dir}/a_labels.pkl")


    print("Summary of all training data")
    a, b = np.unique(astrophysical_classes, return_counts=True)
    print(f"Total sample count = {np.sum(b)}")
    print(pd.DataFrame(data = {'Class': a, 'Count': b}))

    # Small step to convert X_static from a dictionary to an array
    for i in tqdm(range(len(X_static)), desc="Converting dictionaries to arrays"):        
        X_static[i] = get_static_features(X_static[i])

    X_ts_balanced = []
    X_static_balanced = []
    Y_balanced = []
    astrophysical_classes_balanced = []

    # Balance the data set in some way
    for c in np.unique(astrophysical_classes):

        idx = list(np.where(np.array(astrophysical_classes) == c)[0])
        
        if len(idx) > max_class_count:
            idx = random.sample(idx, max_class_count)
    
        X_ts_balanced += [X_ts[i] for i in idx]
        X_static_balanced += [X_static[i] for i in idx]
        Y_balanced += [Y[i] for i in idx]
        astrophysical_classes_balanced += [astrophysical_classes[i] for i in idx]

    # Print summary of the data set used for training and validation
    print("Summary of training data used")
    a, b = np.unique(astrophysical_classes_balanced, return_counts=True)
    data_summary = pd.DataFrame(data = {'Class': a, 'Count': b})
    training_set_size = np.sum(b)
    print(data_summary)

    # Free up some memory
    del X_ts, X_static, Y, astrophysical_classes

    # Split into train and validation
    X_ts_train, X_ts_val, X_static_train, X_static_val, Y_train, Y_val, astrophysical_classes_train, astrophysical_classes_val = train_test_split(X_ts_balanced, X_static_balanced, Y_balanced, astrophysical_classes_balanced, shuffle=True, random_state = default_seed, test_size = default_val_fraction)
    
    # Free up some memory
    del X_ts_balanced, X_static_balanced, Y_balanced, astrophysical_classes_balanced


    tree = get_taxonomy_tree()
    loss_object = WHXE_Loss(tree, astrophysical_classes_train, alpha=0) 
    criterion = loss_object.compute_loss

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    ts_dim = X_ts_train[0].shape[1]
    static_dim = len(X_static_train[0])
    output_dim = len(Y_train[0])

    print(f"TS Input Dim: {ts_dim} | Static Input Dim: {static_dim} | Output Dim: {output_dim}")
    model = get_ensemble_model(output_dim, models)
    model.compile(optimizer=optimizer, loss=criterion)
    print(model.summary())

    keras.utils.plot_model(model, to_file=f'{model_dir}/lstm.pdf', show_shapes=True, show_layer_names=True)

    avg_train_losses = []
    for epoch in range(num_epochs):
        
        print(f"\nStart of epoch {epoch}:\n")
        start_time = time.time()
        
        # Create the augmented data set for training
        X_ts_train_aug, X_static_train_aug, Y_train_aug, astrophysical_classes_train_aug = get_augmented_data(X_ts_train, X_static_train, Y_train, astrophysical_classes_train)
        train_dataset =  tf.data.Dataset.from_tensor_slices((X_ts_train_aug, X_static_train_aug, Y_train_aug, astrophysical_classes_train_aug)).batch(batch_size)
        
        # Array to keep tracking of the training loss
        train_loss_values = []
        
        pbar = tqdm(desc="Training Model", leave=True, total=int(np.ceil(training_set_size/batch_size)))
        # Iterate over the batches of the dataset.
        for step, (x_ts_batch_train, x_static_batch_train, y_batch_train, a_class_batch_train) in enumerate(train_dataset):
            
            pretrained_outputs = []
            for m in models:
                pretrained_outputs.append(m.predict([x_ts_batch_train, x_static_batch_train]))

            loss_value = train_step(pretrained_outputs, y_batch_train, model, criterion, optimizer)
            train_loss_values.append(float(loss_value))
            pbar.update()
        pbar.close()
        
        
        # Log the avg train loss
        avg_train_loss = np.mean(train_loss_values)
        avg_train_losses.append(avg_train_loss)
        print(f"Avg training loss: {float(avg_train_loss):.4f}")
        
        print(f"Time taken: {time.time() - start_time:.2f}s")
        model.save(f"{model_dir}/lstm_epoch_{epoch}.h5")
        
        # Save the model with the smallest training loss
        best_model_epoch = np.argmin(avg_train_losses)
        if best_model_epoch == epoch:
            print(f"Best model is at epoch {epoch} Saving")
            model.save(f"{model_dir}/best_model.h5")
            
        print("==========")

    print(f'Running inference for validation...')
    best_model = keras.models.load_model(f"{model_dir}/best_model.h5", compile=False)

    x1, x2, y_true, _ = get_augmented_data(X_ts_val, X_static_val, Y_val, astrophysical_classes_val, fraction=0.5)
    
    # Run inference on these
    pretrained_outputs = []
    for m in models:
        pretrained_outputs.append(m.predict([x_ts_batch_train, x_static_batch_train]))
    y_pred = best_model.predict(pretrained_outputs)

    plt.plot(list(range(len(avg_train_losses))), avg_train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Avg log loss for training")
    plt.savefig(f"{model_dir}/training_history.pdf")
    plt.close()
 
if __name__=='__main__':
    args = parse_args()

    assert args.model_paths != default_model_paths, "Model list cannot be empty"

    # Loading all the models

    models = []
    for i, path in enumerate(args.model_paths):

        print(f"Loading {path}")
        loaded_model = keras.models.load_model(path, compile=False)
        loaded_model._name = f"model_{i}"

        for layer in loaded_model.layers:
            layer._name = layer._name + str(f"_{i}")
        models.append(loaded_model)

    train_ensemble_model(
                models,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr, 
                max_class_count=args.max_class_count, 
                train_dir=args.train_dir, 
                model_dir=args.model_dir)
