import imageio

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score

from taxonomy import source_node_label

def plot_confusion_matrix(y_true, y_pred, labels, title=None, img_file=None):
    
    n_class = len(labels)
    font = {'size'   : 25}
    plt.rc('font', **font)
    
    cm = np.round(confusion_matrix(y_true, y_pred, labels=labels, normalize='true'),2)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    disp.im_.colorbar.remove()
    
    fig = disp.figure_
    if n_class > 10:
        plt.xticks(rotation=90)
    else:
        plt.yticks(rotation=90)
    
    fig.set_figwidth(18)
    fig.set_figheight(18)
    
    for label in disp.text_.ravel():
        if n_class > 10:
            label.set_fontsize(12)
        elif n_class <= 10 and n_class > 3:
            label.set_fontsize(25)
        else:
            label.set_fontsize(40)
    
    if title:
        disp.ax_.set_xlabel("Predicted Label", fontsize='x-large')
        disp.ax_.set_ylabel("True Label", fontsize='x-large')
        disp.ax_.set_title(title, fontsize='x-large')
    
    plt.tight_layout()

    if img_file:
        plt.savefig(img_file)

def plot_day_vs_class_score(tree, model_dir, show_uncertainties=False):

    column_names = list(nx.bfs_tree(tree, source=source_node_label).nodes())
    leaf_names = column_names[-19:]
    df_master = pd.read_csv(f"{model_dir}/days_since_trigger/combined.csv")

    # Plotting code
    for i, c in enumerate(leaf_names):

        df = df_master.loc[df_master['true_class'].eq(c)]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        fig.subplots_adjust(hspace=0.05)  # adjust space between Axes

        ax1.axvspan(-20, 0, color='gray', alpha=0.15)
        ax2.axvspan(-20, 0, color='gray', alpha=0.15)

        # plot the same data on both Axes
        for c_ in leaf_names:
            if c_ == c:
                ax1.plot(df['days_since_trigger'].to_numpy(), df[f"{c_}_mean"].to_numpy(), linewidth=3)
                ax2.plot(df['days_since_trigger'].to_numpy(), df[f"{c_}_mean"].to_numpy(), linewidth=3)
            else:
                ax1.plot(df['days_since_trigger'].to_numpy(), df[f"{c_}_mean"].to_numpy(), linewidth=1)
                ax2.plot(df['days_since_trigger'].to_numpy(), df[f"{c_}_mean"].to_numpy(), linewidth=1)

            if show_uncertainties:
                ax1.fill_between(df['days_since_trigger'].to_numpy(), np.minimum(1, df[f"{c_}_mean"].to_numpy() + df[f"{c_}_std"].to_numpy()), np.maximum(0, df[f"{c_}_mean"].to_numpy() - df[f"{c_}_std"].to_numpy()), alpha=0.2)
                ax2.fill_between(df['days_since_trigger'].to_numpy(), np.minimum(1, df[f"{c_}_mean"].to_numpy() + df[f"{c_}_std"].to_numpy()), np.maximum(0, df[f"{c_}_mean"].to_numpy() - df[f"{c_}_std"].to_numpy()), alpha=0.2)
        # High probability stuff - linear scale
        ax1.set_ylim(.2, 1.01)  # outliers only

        # Low probability stuff - log scale
        ax2.set_ylim(-0.01, .20)  # most of the data
        #ax2.set_yscale('log')

        # hide the spines between ax and ax2
        ax1.spines.bottom.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()

        # Now, let's turn towards the cut-out slanted lines.
        # We create line objects in axes coordinates, in which (0,0), (0,1),
        # (1,0), and (1,1) are the four corners of the Axes.
        # The slanted lines themselves are markers at those locations, such that the
        # lines keep their angle and position, independent of the Axes size or scale
        # Finally, we need to disable clipping.

        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                    linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

        ax2.set_xlabel('Time since first detection (in days)', fontsize='x-large')
        ax1.set_ylabel('Mean Class score', fontsize='x-large')
        ax1.set_title(f"True Class: {c}", fontsize='x-large')

        plt.tight_layout()
        plt.savefig(f"{model_dir}/days_since_trigger/{i}.pdf")
        plt.close()

def plot_roc_curves(y_true, y_pred, labels, title=None, img_file=None):

    chance = np.arange(0,1.01,0.01)
    plt.figure(figsize=(12,12))
    plt.plot(chance, chance, '--', color='black', label='Random Chance (AUC = 0.5)')

    for i, label in enumerate(labels):

        score = roc_auc_score(y_true[:, i], y_pred[:, i])
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        plt.plot(fpr, tpr, label=f"{label} (AUC = {score:.2f})")

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), fancybox=True, shadow=False, ncol=3, fontsize = 16)
    plt.title(title)
    plt.tight_layout()
    plt.gca().set_aspect('equal')

    if img_file:
        plt.savefig(img_file)

def plot_reliability_diagram(y_true, y_pred, title=None, img_file=None, n_bins=10):

    n_classes = y_true.shape[1]

    bins = np.linspace(0,1,n_bins+1)
    plt.plot(bins, bins, '--', color='black', label='Perfectly calibrated')

    for i in range(n_classes):

        prob_true, prob_pred = calibration_curve(y_true[:, i], y_pred[:, i], n_bins=n_bins)
        plt.scatter(prob_pred, prob_true)

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Empirical")

    plt.legend()
    plt.tight_layout()

    if img_file:
        plt.savefig(img_file)

    plt.close()

def plot_data_set_composition(model_dir):

    sets = ["test_sample", "train_sample", "validation_sample"]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    for s, c in zip(sets, colors):

        df = pd.read_csv(f"{model_dir}/{s}.csv")

        plt.bar(df['Class'].to_numpy(), df['Count'].to_numpy(), color=c)

        plt.ylabel("Count")

        plt.xticks(rotation=90)

        plt.tight_layout()
        plt.savefig(f"{model_dir}/{s}.pdf")
        plt.close()

def make_gif(files, gif_file=None):

    # Load the images
    images = []
    for filename in files:
        images.append(imageio.imread(filename))

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(18, 18))

    # Create the animation
    def animate(i):
        ax.clear()
        ax.axis('off')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.imshow(images[i])
        
    anim = animation.FuncAnimation(fig, animate, frames=len(images), interval=500)

    if gif_file:
        # Save the animation as a GIF
        anim.save(gif_file)

