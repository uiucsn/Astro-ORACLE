import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 
import networkx as nx

from tensorflow import keras
from tqdm import tqdm
from pathlib import Path


from dataloader import LSSTSourceDataSet, load, get_augmented_data, get_static_features, ts_length, get_ts_upto_days_since_trigger
from loss import WHXE_Loss
from taxonomy import get_taxonomy_tree, source_node_label
from vizualizations import make_gif, plot_reliability_diagram, plot_data_set_composition, plot_day_vs_class_score
from interpret_results import get_conditional_probabilites, save_all_cf_and_rocs, save_leaf_cf_and_rocs, save_all_phase_vs_accuracy_plot
from train_RNN import default_batch_size

default_seed = 40
np.random.seed(default_seed)

default_test_dir = Path("processed/test")
default_max_class_count = 1000

fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

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

def run_class_wise_analysis(model, tree, model_dir, X_ts, X_static, Y, astrophysical_classes):

    os.makedirs(f"{model_dir}/days_since_trigger", exist_ok=True)

    # Order the columns using level order traversal
    column_names = list(nx.bfs_tree(tree, source=source_node_label).nodes())

    df_list = []

    for c in np.unique(astrophysical_classes):

        idx = list(np.where(np.array(astrophysical_classes) == c)[0])

        X_ts_class = [X_ts[i] for i in idx]
        X_static_class = [X_static[i] for i in idx]
        Y_class = [Y[i] for i in idx]

        # Find the index for the first and last detections for each object in the sample
        first_detection_indices = [np.where(X_ts_class[i]['detection_flag'].to_numpy() == 1)[0][0] for i in range(len(X_ts_class))] 
        last_detection_indices = [np.where(X_ts_class[i]['detection_flag'].to_numpy() == 1)[0][-1] for i in range(len(X_ts_class))] 

        # Find the days of the first and last detection
        first_detection_days = np.array([X_ts_class[i]['scaled_time_since_first_obs'].to_numpy()[idx] * 100 for i, idx in enumerate(first_detection_indices)])
        last_detection_days = np.array([X_ts_class[i]['scaled_time_since_first_obs'].to_numpy()[idx] * 100 for i, idx in enumerate(last_detection_indices)])

        # Number of days between first and last detection + 1
        t_delta = last_detection_days - first_detection_days + 1
        max_ts_days = max(t_delta)
        day_range = np.arange(-20, max_ts_days, min(3, max_ts_days/100))

        print('Class: ', c, 'Days: ', max_ts_days)

        scores = np.zeros((len(day_range), 2*len(Y[0])))

        for i, d in enumerate(day_range):

            x1 = np.squeeze(get_ts_upto_days_since_trigger(X_ts_class, d))
            x2 = np.squeeze(X_static_class)

            y_pred = model.predict([x1, x2], batch_size=default_batch_size, verbose=0)
            
            # Get the conditional probabilities
            _, pseudo_conditional_probabilities = get_conditional_probabilites(y_pred, tree)

            means = np.mean(pseudo_conditional_probabilities, axis=0)
            std = np.std(pseudo_conditional_probabilities, axis=0)

            # Get the mean of each class scores over all samples at this time
            scores[i,:len(Y[0])] = means
            scores[i,len(Y[0]):] = std
             
        labels = [f"{n}_mean" for n in column_names] + [f"{n}_std" for n in column_names]
        df = pd.DataFrame(scores, columns=labels)
        df['days_since_trigger'] = day_range
        df['true_class'] = [c] * len(day_range)

        df_list.append(df)
    
    combined_df = pd.concat(df_list)
    combined_df.to_csv(f"{model_dir}/days_since_trigger/combined.csv")


def run_fractional_analysis(model, tree, model_dir, X_ts, X_static, Y, astrophysical_classes):


    all_predictions = []
    all_trues = []

    for f in fractions:

        print(f'Running inference for {int(f*100)}% light curves...')

        x1, x2, y_true, _, _ = get_augmented_data(X_ts, X_static, Y, astrophysical_classes, fraction=f)
        
        # Run inference on these
        y_pred = model.predict([x1, x2], batch_size=default_batch_size)

        # Get the conditional probabilities
        _, pseudo_conditional_probabilities = get_conditional_probabilites(y_pred, tree)
        
        print(f'For {int(f*100)}% of the light curve, these are the statistics:')
        
        # Print all the stats and make plots...
        save_all_cf_and_rocs(y_true, pseudo_conditional_probabilities, tree, model_dir, f)
        save_leaf_cf_and_rocs(y_true, pseudo_conditional_probabilities, tree, model_dir, f)

        all_predictions.append(pseudo_conditional_probabilities)
        all_trues.append(y_true)

        plt.close()


    all_predictions = np.concatenate(all_predictions)
    all_trues = np.concatenate(all_trues)

    plot_reliability_diagram(all_trues[:, 1:3], all_predictions[:, 1:3], title="Calibration at level 1", img_file=f"{model_dir}/level_1_cal.pdf")
    plot_reliability_diagram(all_trues[:, 3:8], all_predictions[:, 3:8], title="Calibration at level 2", img_file=f"{model_dir}/level_2_cal.pdf")
    plot_reliability_diagram(all_trues[:, -19:], all_predictions[:, -19:], title="Calibration at the leaves", img_file=f"{model_dir}/leaf_cal.pdf")
    plt.close()

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

    # Print summary of the data set used for testing
    a, b = np.unique(astrophysical_classes_balanced, return_counts=True)
    data_summary = pd.DataFrame(data = {'Class': a, 'Count': b})
    data_summary.to_csv(f"{model_dir}/test_sample.csv")
    print(data_summary)

    del X_ts, X_static, Y, astrophysical_classes

    tree = get_taxonomy_tree()
    best_model = keras.models.load_model(f"{model_dir}/best_model.h5", compile=False)

    # Run all the analysis code
    run_class_wise_analysis(best_model, tree, model_dir, X_ts_balanced, X_static_balanced, Y_balanced, astrophysical_classes_balanced)
    plot_day_vs_class_score(tree, model_dir)

    # Make plots of the scores
    run_fractional_analysis(best_model, tree, model_dir, X_ts_balanced, X_static_balanced, Y_balanced, astrophysical_classes_balanced)


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