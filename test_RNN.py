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

from tensorflow.keras.utils import pad_sequences
from dataloader import LSSTSourceDataSet, load, get_augmented_data, get_static_features, ts_length, get_ts_upto_days_since_trigger, ts_flag_value, static_flag_value, augment_ts_length_to_days_since_trigger
from loss import WHXE_Loss
from taxonomy import get_taxonomy_tree, source_node_label
from vizualizations import make_gif, plot_reliability_diagram, plot_data_set_composition, plot_day_vs_class_score, plot_lc, make_z_plots
from interpret_results import get_conditional_probabilites, save_all_cf_and_rocs, save_leaf_cf_and_rocs, save_all_phase_vs_accuracy_plot
from train_RNN import default_batch_size

default_seed = 40
np.random.seed(default_seed)

default_test_dir = Path("processed/test")
default_max_class_count = 1000

fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
days = 2 ** np.array(range(11))

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

def plot_some_lcs(model, X_ts, X_static, Y,  astrophysical_classes, class_count=20):
    tree = get_taxonomy_tree()

    for j, c in enumerate(np.unique(astrophysical_classes)):

        idx = list(np.where(np.array(astrophysical_classes) == c)[0])[:class_count]
        X_ts_class = [X_ts[i] for i in idx]
        X_static_class = [X_static[i] for i in idx]
        Y_class = [Y[i] for i in idx]

        for i in range(class_count):

            table = X_ts_class[i]
            static = X_static_class[i]
            target = Y_class[i]

            tables = []
            statics = []
            for k in range(1, table.to_numpy().shape[0] + 1):
                tables.append(table.to_numpy()[:k, :])
                statics.append(static)
            
            tables = pad_sequences(tables, maxlen=ts_length,  dtype='float32', padding='post', value=ts_flag_value)
            statics = np.squeeze(statics)

            true_class_idx = np.argmax(target[-19:])

            logits = model.predict([tables, statics], verbose=0)
            _, pseudo_conditional_probabilities = get_conditional_probabilites(logits, tree)
            leaf_probs = pseudo_conditional_probabilities[:, -19:]
            true_class_score = leaf_probs[:, true_class_idx]

            plot_lc(table, true_class_score, c, file_name=f"{j}-{i}")


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
        max_ts_days = 200 #max(t_delta) # use commented out section if you want to plot class scores for many more days. By default, it plots between -20 and +200 days
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

def run_day_wise_analysis(model, tree, model_dir, X_ts, X_static, Y, astrophysical_classes):
    

    all_predictions = []
    all_trues = []


    for d in days:

        print(f'Running inference for trigger + {d} days...')

        x1, x2, y_true, _ = augment_ts_length_to_days_since_trigger(X_ts, X_static, Y, astrophysical_classes, d)
        
        # Run inference on these
        y_pred = model.predict([x1, x2], batch_size=default_batch_size)

        # Get the conditional probabilities
        _, pseudo_conditional_probabilities = get_conditional_probabilites(y_pred, tree)
        
        print(f'For trigger + {d} days, these are the statistics:')

        plot_title = f"Trigger + {d} days"
        
        # Print all the stats and make plots...
        save_all_cf_and_rocs(y_true, pseudo_conditional_probabilities, tree, model_dir, plot_title)
        save_leaf_cf_and_rocs(y_true, pseudo_conditional_probabilities, tree, model_dir, plot_title)

        all_predictions.append(pseudo_conditional_probabilities)
        all_trues.append(y_true)

        plt.close()


    # Make the gifs at leaf nodes for days
    cf_files = [f"{model_dir}/gif/leaf_cf/Trigger + {d} days.png" for d in days]
    make_gif(cf_files, f'{model_dir}/gif/leaf_cf/leaf_cf_days.gif')
    plt.close()

    roc_files = [f"{model_dir}/gif/leaf_roc/Trigger + {d} days.png" for d in days]
    make_gif(roc_files, f'{model_dir}/gif/leaf_roc/leaf_roc_days.gif')
    plt.close()

    # Make the gifs at the level 1 of the tree
    cf_files = [f"{model_dir}/gif/level_1_cf/Trigger + {d} days.png" for d in days]
    make_gif(cf_files, f'{model_dir}/gif/level_1_cf/level_1_cf_days.gif')
    plt.close()

    roc_files = [f"{model_dir}/gif/level_1_roc/Trigger + {d} days.png" for d in days]
    make_gif(roc_files, f'{model_dir}/gif/level_1_roc/level_1_roc_days.gif')
    plt.close()

    # Make the gifs at the level 2 of the tree
    cf_files = [f"{model_dir}/gif/level_2_cf/Trigger + {d} days.png" for d in days]
    make_gif(cf_files, f'{model_dir}/gif/level_2_cf/level_2_cf_days.gif')
    plt.close()

    roc_files = [f"{model_dir}/gif/level_2_roc/Trigger + {d} days.png" for d in days]
    make_gif(roc_files, f'{model_dir}/gif/level_2_roc/level_2_roc_days.gif')
    plt.close()


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

        plot_title = f"{int(f*100)} percent"
        
        # Print all the stats and make plots...
        save_all_cf_and_rocs(y_true, pseudo_conditional_probabilities, tree, model_dir, plot_title)
        save_leaf_cf_and_rocs(y_true, pseudo_conditional_probabilities, tree, model_dir, plot_title)

        all_predictions.append(pseudo_conditional_probabilities)
        all_trues.append(y_true)

        plt.close()


    all_predictions = np.concatenate(all_predictions)
    all_trues = np.concatenate(all_trues)

    # plot_reliability_diagram(all_trues[:, 1:3], all_predictions[:, 1:3], title="Calibration at level 1", img_file=f"{model_dir}/level_1_cal.pdf")
    # plot_reliability_diagram(all_trues[:, 3:8], all_predictions[:, 3:8], title="Calibration at level 2", img_file=f"{model_dir}/level_2_cal.pdf")
    # plot_reliability_diagram(all_trues[:, -19:], all_predictions[:, -19:], title="Calibration at the leaves", img_file=f"{model_dir}/leaf_cal.pdf")
    # plt.close()

    cf_files = [f"{model_dir}/gif/leaf_cf/{int(f*100)} percent.png" for f in fractions]
    make_gif(cf_files, f'{model_dir}/gif/leaf_cf/leaf_cf_fraction.gif')
    plt.close()

    roc_files = [f"{model_dir}/gif/leaf_roc/{int(f*100)} percent.png" for f in fractions]
    make_gif(roc_files, f'{model_dir}/gif/leaf_roc/leaf_roc_fraction.gif')
    plt.close()

    # Make the gifs at the level 1 of the tree
    cf_files = [f"{model_dir}/gif/level_1_cf/{int(f*100)} percent.png" for f in fractions]
    make_gif(cf_files, f'{model_dir}/gif/level_1_cf/level_1_cf_fraction.gif')
    plt.close()

    roc_files = [f"{model_dir}/gif/level_1_roc/{int(f*100)} percent.png" for f in fractions]
    make_gif(roc_files, f'{model_dir}/gif/level_1_roc/level_1_roc_fraction.gif')
    plt.close()

    # Make the gifs at the level 2 of the tree
    cf_files = [f"{model_dir}/gif/level_2_cf/{int(f*100)} percent.png" for f in fractions]
    make_gif(cf_files, f'{model_dir}/gif/level_2_cf/level_2_cf_fraction.gif')
    plt.close()

    roc_files = [f"{model_dir}/gif/level_2_roc/{int(f*100)} percent.png" for f in fractions]
    make_gif(roc_files, f'{model_dir}/gif/level_2_roc/level_2_roc_fraction.gif')
    plt.close()

def run_pre_detection_comparison(model, tree, model_dir, X_ts, X_static, Y):

    # Run comparison for model performance before the first detection using 1. both non detections and host 2. Non detections only 3. Host galaxy only

    # Get data from upto 50 days before the trigger to upto the trigger, excluding the first detection
    x1 = np.squeeze(get_ts_upto_days_since_trigger(X_ts, 0))

    # Get all the static features and format them correctly
    x2 = np.squeeze(X_static)
    Y = np.squeeze(Y)

    # Run predictions and analysis for non detections + host
    y_pred_both = model.predict([x1, x2], batch_size=default_batch_size, verbose=0)
    _, pseudo_conditional_probabilities_both = get_conditional_probabilites(y_pred_both, tree)

    os.makedirs(f"{model_dir}/pre_trigger/both", exist_ok=True)
    plot_title = f"0 percent"
    save_all_cf_and_rocs(Y, pseudo_conditional_probabilities_both, tree, f"{model_dir}/pre_trigger/both", plot_title)
    save_leaf_cf_and_rocs(Y, pseudo_conditional_probabilities_both, tree, f"{model_dir}/pre_trigger/both", plot_title)

    # Run predictions and analysis for host only
    y_pred_host = model.predict([np.ones_like(x1) * ts_flag_value, x2], batch_size=default_batch_size, verbose=0)
    _, pseudo_conditional_probabilities_host = get_conditional_probabilites(y_pred_host, tree)

    os.makedirs(f"{model_dir}/pre_trigger/host", exist_ok=True)
    save_all_cf_and_rocs(Y, pseudo_conditional_probabilities_host, tree, f"{model_dir}/pre_trigger/host", plot_title)
    save_leaf_cf_and_rocs(Y, pseudo_conditional_probabilities_host, tree, f"{model_dir}/pre_trigger/host", plot_title)
    
    # Run predictions and analysis for non detections only
    y_pred_nd = model.predict([x1, np.ones_like(x2) * static_flag_value], batch_size=default_batch_size, verbose=0)
    _, pseudo_conditional_probabilities_nd = get_conditional_probabilites(y_pred_nd, tree)

    os.makedirs(f"{model_dir}/pre_trigger/nd", exist_ok=True)
    save_all_cf_and_rocs(Y, pseudo_conditional_probabilities_nd, tree, f"{model_dir}/pre_trigger/nd", plot_title)
    save_leaf_cf_and_rocs(Y, pseudo_conditional_probabilities_nd, tree, f"{model_dir}/pre_trigger/nd", plot_title)
    

def test_model(model_dir, test_dir=default_test_dir, max_class_count=default_max_class_count):

    random.seed(default_seed)

    # This step takes a while because it has load from disc to memory...
    print("Loading testing data from disc...")
    X_ts = load(f"{test_dir}/x_ts.pkl")
    X_static = load(f"{test_dir}/x_static.pkl")
    Y = load(f"{test_dir}/y.pkl")
    astrophysical_classes = load(f"{test_dir}/a_labels.pkl")

    z_arr = [X_static[i]['REDSHIFT_HELIO'] for i in range(len(X_static))]
    
    #make_z_plots(astrophysical_classes, z_arr, X_static, model_dir)

    a, b = np.unique(astrophysical_classes, return_counts=True)
    print(f"Total sample count = {np.sum(b)}")
    print(pd.DataFrame(data = {'Class': a, 'Count': b}))

    for i in tqdm(range(len(X_static))):        
        X_static[i] = get_static_features(X_static[i])

    X_ts_balanced = []
    X_static_balanced = []
    Y_balanced = []
    astrophysical_classes_balanced = []

    # Fix random seed while generating LCs for figure 19
    #random.seed(42)

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

    # Some general code to plot light curves
    #plot_some_lcs(best_model, X_ts_balanced, X_static_balanced, Y_balanced, astrophysical_classes_balanced)

    # Run all the analysis code
    # run_class_wise_analysis(best_model, tree, model_dir, X_ts_balanced, X_static_balanced, Y_balanced, astrophysical_classes_balanced)
    # plot_day_vs_class_score(tree, model_dir)

    # Run day wise analysis
    # run_day_wise_analysis(best_model, tree, model_dir, X_ts_balanced, X_static_balanced, Y_balanced, astrophysical_classes_balanced)

    # Make plots of the scores
    #run_fractional_analysis(best_model, tree, model_dir, X_ts_balanced, X_static_balanced, Y_balanced, astrophysical_classes_balanced)

    # Run pre trigger analysis
    # run_pre_detection_comparison(best_model, tree, model_dir, X_ts_balanced, X_static_balanced, Y_balanced)


if __name__=='__main__':
    args = parse_args()
    test_model(args.model_dir, 
               test_dir=args.test_dir, 
               max_class_count=args.max_class_count)