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

from RNN_model import get_RNN_model
from dataloader import load, get_augmented_data, get_static_features
from loss import WHXE_Loss
from taxonomy import get_taxonomy_tree

default_seed = 40
default_val_fraction = 0.05
val_fractions = [0.1, 0.4, 0.6, 1]

default_num_epochs = 100
default_batch_size = 1024
default_learning_rate=5e-4
default_latent_size = 64
default_alpha = 0.5

default_train_dir = Path("processed/train")
default_model_dir = Path("models/test")
default_max_class_count = 40000



def parse_args():
    '''
    Get commandline options
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=default_num_epochs, help='Number of epochs to train the model for.')
    parser.add_argument('--batch_size', type=int, default=default_batch_size, help='Batch size used for training.')
    parser.add_argument('--lr', type=float, default=default_learning_rate, help='Learning rate used for training.')
    parser.add_argument('--latent_size', type=int, default=default_latent_size, help='Dimension of the final latent layer of the neural net.')
    parser.add_argument('--alpha', type=float, default=default_alpha, help='Alpha value used for the loss function. See Villar et al. (2024) for more information [https://arxiv.org/abs/2312.02266]')
    parser.add_argument('--max_class_count', type=int, default=default_max_class_count, help='Maximum number of samples in each class.')
    parser.add_argument('--train_dir', type=Path, default=default_train_dir, help='Directory which contains the training data.')
    parser.add_argument('--model_dir', type=Path, default=default_model_dir, help='Directory for saving the models and best model during training.')

    args = parser.parse_args()
    return args

@tf.function
def train_step(x_ts, x_static, y, model, criterion, optimizer):
    with tf.GradientTape() as tape:
        logits = model((x_ts, x_static), training=True)
        loss_value = criterion(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss_value

@tf.function
def test_step(x_ts, x_static, y, model, criterion):
    val_logits = model([x_ts, x_static], training=False)
    loss_value = criterion(y, val_logits)
    return loss_value


def train_model(num_epochs=default_num_epochs, batch_size=default_batch_size, learning_rate=default_learning_rate, latent_size=default_latent_size, alpha=default_alpha, max_class_count=default_max_class_count, train_dir=default_train_dir, model_dir=default_model_dir):

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


    # Free up some memory
    del X_ts, X_static, Y, astrophysical_classes

    # Split into train and validation
    X_ts_train, X_ts_val, X_static_train, X_static_val, Y_train, Y_val, astrophysical_classes_train, astrophysical_classes_val = train_test_split(X_ts_balanced, X_static_balanced, Y_balanced, astrophysical_classes_balanced, shuffle=True, random_state = default_seed, test_size = default_val_fraction)
    
    # Free up some memory
    del X_ts_balanced, X_static_balanced, Y_balanced, astrophysical_classes_balanced

    # Print summary of the data set used for training and validation
    print("Summary of training data used")
    a, b = np.unique(astrophysical_classes_train, return_counts=True)
    data_summary = pd.DataFrame(data = {'Class': a, 'Count': b})
    data_summary.to_csv(f"{model_dir}/train_sample.csv")
    print(data_summary)

    print("Summary of validation data used")
    a, b = np.unique(astrophysical_classes_val, return_counts=True)
    data_summary = pd.DataFrame(data = {'Class': a, 'Count': b})
    data_summary.to_csv(f"{model_dir}/validation_sample.csv")
    print(data_summary)


    tree = get_taxonomy_tree()
    loss_object = WHXE_Loss(tree, astrophysical_classes_train, alpha=alpha) 
    criterion = loss_object.compute_loss

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=10000, decay_rate=0.9)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    ts_dim = X_ts_train[0].shape[1]
    static_dim = len(X_static_train[0])
    output_dim = len(Y_train[0])

    print(f"TS Input Dim: {ts_dim} | Static Input Dim: {static_dim} | Output Dim: {output_dim}")

    model = get_RNN_model(ts_dim, static_dim, output_dim, latent_size)
    model.compile(optimizer=optimizer, loss=criterion)

    #keras.utils.plot_model(model, to_file=f'{model_dir}/lstm.pdf', show_shapes=True, show_layer_names=True)

    # Create an augmented data set for validation
    print("Creating augmented validation data set")
    X_ts_val_aug, X_static_val_aug, Y_val_aug, astrophysical_classes_val_aug, fractions_val_aug = [], [], [], [], []

    for f in val_fractions:    
        x1_, x2_, y_, a_, f_ = get_augmented_data(X_ts_val, X_static_val, Y_val, astrophysical_classes_val, fraction=f)
        X_ts_val_aug.append(x1_)
        X_static_val_aug.append(x2_)
        Y_val_aug.append(y_)
        astrophysical_classes_val_aug.append(a_)
        fractions_val_aug.append(f_)

    del X_ts_val, X_static_val, Y_val, astrophysical_classes_val

    X_ts_val_aug = np.concatenate(X_ts_val_aug)
    X_static_val_aug = np.concatenate(X_static_val_aug)
    Y_val_aug = np.concatenate(Y_val_aug)
    astrophysical_classes_val_aug = np.concatenate(astrophysical_classes_val_aug)
    fractions_val_aug = np.concatenate(fractions_val_aug)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_ts_val_aug, X_static_val_aug, Y_val_aug, astrophysical_classes_val_aug, fractions_val_aug)).batch(batch_size)
    val_set_size = len(astrophysical_classes_val_aug)

    del X_ts_val_aug, X_static_val_aug, Y_val_aug, astrophysical_classes_val_aug, fractions_val_aug

    avg_train_losses = []
    avg_val_losses = []

    for epoch in range(num_epochs):
        
        print(f"\nStart of epoch {epoch}:\n")
        start_time = time.time()
        
        # Create the augmented data set for training
        X_ts_train_aug, X_static_train_aug, Y_train_aug, astrophysical_classes_train_aug, fractions_train_aug = get_augmented_data(X_ts_train, X_static_train, Y_train, astrophysical_classes_train)
        train_dataset =  tf.data.Dataset.from_tensor_slices((X_ts_train_aug, X_static_train_aug, Y_train_aug, astrophysical_classes_train_aug, fractions_train_aug)).batch(batch_size)
        
        # Array to keep tracking of the training loss
        train_loss_values = []
        val_loss_values = []
        
        pbar = tqdm(desc="Training Model", leave=True, total=int(np.ceil(len(astrophysical_classes_train)/batch_size)))
        # Iterate over the batches of the dataset.
        for step, (x_ts_batch_train, x_static_batch_train, y_batch_train, a_class_batch_train, fractions_batch_train) in enumerate(train_dataset):
            loss_value = train_step(x_ts_batch_train, x_static_batch_train, y_batch_train, model, criterion, optimizer)
            train_loss_values.append(float(loss_value))
            pbar.update()
        pbar.close()


        pbar = tqdm(desc="Validate Model", leave=True, total=int(np.ceil(val_set_size/batch_size)))
        # Iterate over the batches of the dataset.
        for step, (x_ts_batch_val, x_static_batch_val, y_batch_val, a_class_batch_val, fractions_batch_val) in enumerate(val_dataset):
            loss_value = test_step(x_ts_batch_val, x_static_batch_val, y_batch_val, model, criterion)
            val_loss_values.append(float(loss_value))
            pbar.update()
        pbar.close()
        
        
        # Log the avg train loss
        avg_train_loss = np.mean(train_loss_values)
        avg_train_losses.append(avg_train_loss)
        print(f"Avg training loss: {float(avg_train_loss):.4f}")


        # Log the avg val loss
        avg_val_loss = np.mean(val_loss_values)
        avg_val_losses.append(avg_val_loss)
        print(f"Avg val loss: {float(avg_val_loss):.4f}")

        if np.isnan(avg_train_loss) == True:

            print("Training loss was nan. Exiting the loop.")
            break
        
        print(f"Time taken: {time.time() - start_time:.2f}s")
        #model.save(f"{model_dir}/rnn_epoch_{epoch}.h5")
        
        # Save the model with the smallest training loss
        best_model_epoch = np.argmin(avg_val_losses)
        if best_model_epoch == epoch:
            print(f"Best model is at epoch {epoch}. Saving...")
            model.save(f"{model_dir}/best_model.h5")
            
        print("==========")

    pd.DataFrame({'Avg_train_loss': avg_train_losses, 'Avg_val_loss': avg_val_losses}).to_csv(f"{model_dir}/loss_history.csv")

 
if __name__=='__main__':
    args = parse_args()
    train_model(num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr, 
                latent_size=args.latent_size, 
                alpha=args.alpha, 
                max_class_count=args.max_class_count, 
                train_dir=args.train_dir, 
                model_dir=args.model_dir)
