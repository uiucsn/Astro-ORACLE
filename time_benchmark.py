import argparse
import time
import numpy as np
import random 
import matplotlib.pyplot as plt

from tensorflow import keras
from tqdm import tqdm
from pathlib import Path

from dataloader import LSSTSourceDataSet, load, get_augmented_data, get_static_features, ts_length, get_ts_upto_days_since_trigger, ts_flag_value, static_flag_value, augment_ts_length_to_days_since_trigger
from loss import WHXE_Loss
from taxonomy import get_taxonomy_tree, source_node_label
from vizualizations import make_gif, plot_reliability_diagram, plot_data_set_composition, plot_day_vs_class_score, plot_lc, make_z_plots

default_seed = 40
np.random.seed(default_seed)

default_test_dir = Path("processed/test")
default_max_class_count = 10000

fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
days = 2 ** np.array(range(11))

# Uncomment for CPU testing 
# import os
# import tensorflow as tf
# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)
# os.environ["OMP_NUM_THREADS"] = "1"

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

def run_time_bench_mark(model_dir, test_dir=default_test_dir, max_class_count=default_max_class_count):

    # This step takes a while because it has load from disc to memory...
    print("Loading testing data from disc...")
    X_ts = load(f"{test_dir}/x_ts.pkl")
    X_static = load(f"{test_dir}/x_static.pkl")
    Y = load(f"{test_dir}/y.pkl")
    astrophysical_classes = load(f"{test_dir}/a_labels.pkl")

    a, b = np.unique(astrophysical_classes, return_counts=True)

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

    tree = get_taxonomy_tree()
    model = keras.models.load_model(f"{model_dir}/best_model.h5", compile=False)

    del X_ts, X_static, Y, astrophysical_classes

    x1, x2, y_true, _ = augment_ts_length_to_days_since_trigger(X_ts_balanced, X_static_balanced, Y_balanced, astrophysical_classes_balanced, 1024)
    
    # Total number of objects to classify 
    n = 20000

    # Run inference on these for batch size = 1
    batch_size = 1
    start = time.time()
    y_pred = model.predict([x1[:n, :, :], x2[:n, :]], batch_size=batch_size)
    end = time.time()
    print(f'Avg time for batch size = {batch_size}:', (end - start)/n, "s per LC")

    # Run inference on these for batch size = 2000
    batch_size = 2000
    start = time.time()
    model.predict([x1[n:2*n, :, :], x2[:n, :]], batch_size=batch_size)
    end = time.time()
    print(f'Avg time for batch size = {batch_size}:', (end - start)/n, "s per LC")

if __name__=='__main__':
    args = parse_args()
    run_time_bench_mark(args.model_dir, 
               test_dir=args.test_dir, 
               max_class_count=args.max_class_count)