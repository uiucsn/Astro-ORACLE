import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 

from tensorflow import keras
from tqdm import tqdm
from pathlib import Path


from dataloader import LSSTSourceDataSet, load, get_augmented_data, get_static_features, ts_length
from loss import WHXE_Loss
from taxonomy import get_taxonomy_tree
from vizualizations import make_gif
from interpret_results import get_conditional_probabilites, save_all_cf_and_rocs, save_leaf_cf_and_rocs
from train_RNN import default_batch_size

default_seed = 42

default_test_dir = Path("processed/test")
default_max_class_count = 1000

def parse_args():
    '''
    Get commandline options
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=Path, required=True, help='Directory containing best_mode.h5. Results will be stored in the same directory.')
    parser.add_argument('--test_dir', type=Path, default=default_test_dir, help='Directory which contains the testing data.')
    parser.add_argument('--max_class_count', type=int, default=default_max_class_count, help='Maximum number of samples in each class.')
    args = parser.parse_args()
    return args

def test_model(model_dir, test_dir=default_test_dir, max_class_count=default_max_class_count):

    random.seed(default_seed)

    # This step takes a while because it has load from disc to memory...
    print("Loading testing data from disc...")
    X_ts = load(f"{test_dir}/x_ts.pkl")
    X_static = load(f"{test_dir}/x_static.pkl")
    Y = load(f"{test_dir}/y.pkl")
    astrophysical_classes = load(f"{test_dir}/a_labels.pkl")

    a, b = np.unique(astrophysical_classes, return_counts=True)
    print(f"Total sample count = {np.sum(b)}")
    print(pd.DataFrame(data = {'Class': a, 'Count': b}))

    for i in tqdm(range(len(X_static))):        
        X_static[i] = get_static_features(X_static[i])

    X_ts_balanced = []
    X_static_balanced = []
    Y_balanced = []
    astrophysical_classes_balanced = []

    for c in np.unique(astrophysical_classes):

        idx = list(np.where(np.array(astrophysical_classes) == c)[0])
        
        if len(idx) > max_class_count:
            idx = random.sample(idx, max_class_count)
    
        X_ts_balanced += [X_ts[i] for i in idx]
        X_static_balanced += [X_static[i] for i in idx]
        Y_balanced += [Y[i] for i in idx]
        astrophysical_classes_balanced += [astrophysical_classes[i] for i in idx]

    # Print summary of the data set used for training and validation
    a, b = np.unique(astrophysical_classes_balanced, return_counts=True)
    data_summary = pd.DataFrame(data = {'Class': a, 'Count': b})
    print(data_summary)

    del X_ts, X_static, Y, astrophysical_classes

    tree = get_taxonomy_tree()
    best_model = keras.models.load_model(f"{model_dir}/best_model.h5", compile=False)

    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for f in fractions:

        print(f'Running inference for {int(f*100)}% light curves...')

        x1, x2, y_true, _ = get_augmented_data(X_ts_balanced, X_static_balanced, Y_balanced, astrophysical_classes_balanced, fraction=f)
        
        # Run inference on these
        y_pred = best_model.predict([x1, x2], batch_size=default_batch_size)

        # Get the conditional probabilities
        _, pseudo_conditional_probabilities = get_conditional_probabilites(y_pred, tree)
        
        print(f'For {int(f*100)}% of the light curve, these are the statistics:')
        
        # TODO: create dirs in the model directory
        # Print all the stats and make plots...
        save_all_cf_and_rocs(y_true, pseudo_conditional_probabilities, tree, model_dir, f)
        save_leaf_cf_and_rocs(y_true, pseudo_conditional_probabilities, tree, model_dir, f)
        
        plt.close()
    
    # Make the gifs at leaf nodes
    cf_files = [f"{model_dir}/gif/leaf_cf/{f}.png" for f in fractions]
    make_gif(cf_files, f'{model_dir}/gif/leaf_cf/leaf_cf.gif')
    plt.close()

    roc_files = [f"{model_dir}/gif/leaf_roc/{f}.png" for f in fractions]
    make_gif(roc_files, f'{model_dir}/gif/leaf_roc/leaf_roc.gif')
    plt.close()

    # Make the gifs at the level 1 of the tree
    cf_files = [f"{model_dir}/gif/level_1_cf/{f}.png" for f in fractions]
    make_gif(cf_files, f'{model_dir}/gif/level_1_cf/level_1_cf.gif')
    plt.close()

    roc_files = [f"{model_dir}/gif/level_1_roc/{f}.png" for f in fractions]
    make_gif(roc_files, f'{model_dir}/gif/level_1_roc/level_1_roc.gif')
    plt.close()

    # Make the gifs at the level 2 of the tree
    cf_files = [f"{model_dir}/gif/level_2_cf/{f}.png" for f in fractions]
    make_gif(cf_files, f'{model_dir}/gif/level_2_cf/level_2_cf.gif')
    plt.close()

    roc_files = [f"{model_dir}/gif/level_2_roc/{f}.png" for f in fractions]
    make_gif(roc_files, f'{model_dir}/gif/level_2_roc/level_2_roc.gif')
    plt.close()

if __name__=='__main__':
    args = parse_args()
    test_model(args.model_dir, 
               test_dir=args.test_dir, 
               max_class_count=args.max_class_count)