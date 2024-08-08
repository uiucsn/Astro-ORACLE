import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from test_RNN import fractions, days
from vizualizations import make_gif, plot_reliability_diagram, plot_data_set_composition
from interpret_results import save_all_phase_vs_accuracy_plot, merge_performance_tables

default_seed = 40
np.random.seed(default_seed)

def merge_sample_tables(model_dir):

    df_train = pd.read_csv(f"{model_dir}/train_sample.csv", index_col=0)
    df_train = df_train._append({'Class':'Total', 'Count':sum(df_train['Count'].to_numpy())}, ignore_index=True)
    df_train.rename(columns={'Count': 'Train_count'}, inplace=True)

    df_test = pd.read_csv(f"{model_dir}/test_sample.csv", index_col=0)
    df_test = df_test._append({'Class':'Total', 'Count':sum(df_test['Count'].to_numpy())}, ignore_index=True)
    df_test.rename(columns={'Count': 'Test_count'}, inplace=True)

    df_val = pd.read_csv(f"{model_dir}/validation_sample.csv", index_col=0)
    df_val = df_val._append({'Class':'Total', 'Count':sum(df_val['Count'].to_numpy())}, ignore_index=True)
    df_val.rename(columns={'Count': 'val_count'}, inplace=True)

    df_combined = df_train.merge(df_test, on='Class')
    df_combined = df_combined.merge(df_val, on='Class')

    df_combined.to_csv(f'{model_dir}/combined_sample.csv')
    print(df_combined.to_latex(index=False))
    
def make_training_history_plot(model_dir):

    df = pd.read_csv(f"{model_dir}/loss_history.csv")

    avg_train_losses = df['Avg_train_loss']
    avg_val_losses = df['Avg_val_loss']

    plt.plot(list(range(len(avg_train_losses))), np.log(avg_train_losses), label='Train Data')
    plt.plot(list(range(len(avg_val_losses))), np.log(avg_val_losses), label='Validation Data')

    plt.xlabel("Epoch", fontsize='x-large')
    plt.ylabel("Mean log loss", fontsize='x-large')

    plt.legend()

    plt.savefig(f"{model_dir}/training_history.pdf")
    plt.close()

def parse_args():
    '''
    Get commandline options
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=Path, required=True, help='Directory containing best_mode.h5. Results will be stored in the same directory.')
    args = parser.parse_args()
    return args

def run_analysis(model_dir):

    #make_training_history_plot(model_dir)

    save_all_phase_vs_accuracy_plot(model_dir, days=days)
    plt.close()

    # plot the make up of all the data sets
    plot_data_set_composition(model_dir)
    plt.close()

    merge_sample_tables(model_dir)

    merge_performance_tables(model_dir)

if __name__=='__main__':
    args = parse_args()
    run_analysis(args.model_dir)