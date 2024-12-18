import imageio

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score

from taxonomy import source_node_label
from LSST_Source import LSST_Source

cm = plt.get_cmap('gist_rainbow')
NUM_COLORS = 20
color_arr=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

def plot_legend(labels, filename=None, expand=[-5,-5,5,5]):

    for i, l in enumerate(labels):
        plt.plot([0], [0], label=l, color=color_arr[i])

    legend = plt.legend(loc=3, ncol=2, framealpha=1, frameon=True, fontsize=20)

    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    if filename != None:
        fig.savefig(filename, dpi="figure", bbox_inches=bbox)
    plt.close()

def plot_lc(table, true_class_score, class_name, file_name=None):

    def plot_lc_legend(filename=None, expand=[-5,-5,5,5]):

        legend = plt.legend(handles=get_legend_patches(), ncol=2)

        fig  = legend.figure
        fig.canvas.draw()
        bbox  = legend.get_window_extent()
        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        if filename != None:
            fig.savefig(filename, dpi="figure", bbox_inches=bbox)
        plt.close()

    def get_pb_color(wavelength):

        all_wavelengths = list(LSST_Source.pb_wavelengths.values())
        all_wavelengths.sort()
        idx = all_wavelengths.index(wavelength)

        return f"C{idx}"

    def get_legend_patches():

        patches = []

        all_wavelengths = list(LSST_Source.pb_wavelengths.values())
        all_wavelengths.sort()

        for passband in LSST_Source.pb_wavelengths:

            wavelength = LSST_Source.pb_wavelengths[passband]
            idx = all_wavelengths.index(wavelength)
            patch = mpatches.Patch(color=f"C{idx}", label=passband)
            patches.append(patch)

        return patches

    times = table['scaled_time_since_first_obs'].to_numpy()
    first_detection_idx = np.where(table['detection_flag'].to_numpy() == 1)[0][0]
    first_detection_t = times[first_detection_idx]
    days_since_trigger = (times - first_detection_t) * 100

    flux = table['scaled_FLUXCAL'].to_numpy() * LSST_Source.flux_scaling_const
    flux_err = table['scaled_FLUXCALERR'].to_numpy() * LSST_Source.flux_scaling_const
    detection_mask = table['detection_flag'].to_numpy() == 1
    non_detection_mask = table['detection_flag'].to_numpy() == 0
    colors = np.array([get_pb_color(w) for w in table['band_label'].to_numpy()])


    fig, ax1 = plt.subplots()

    # Plot the detections
    for c in np.unique(colors):

        color_mask = colors == c
        combined_mask = color_mask & detection_mask
        ax1.errorbar(x=days_since_trigger[combined_mask], y=flux[combined_mask], yerr=flux_err[combined_mask], color=c, fmt="*", label=c)


    # Plot the non detections
    ax1.scatter(days_since_trigger[non_detection_mask], flux[non_detection_mask], color=colors[non_detection_mask], marker="x", alpha=0.5)

    ax1.set_xlim(-20, 200)
    ax1.axvspan(-20, 0, color='gray', alpha=0.15)

    ax1.set_xlabel('Days from first detection', fontsize='xx-large')
    ax1.set_ylabel('Calibrated Flux', fontsize='xx-large')
    ax1.set_title(f"True class: {class_name}", fontsize='xx-large')

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

  
    ax2.set_ylabel('True Class Score', fontsize='xx-large') 
    ax2.plot(days_since_trigger, true_class_score, color='black', linestyle='dotted')

    #ax1.legend(handles=get_legend_patches())

    fig.tight_layout()

    if file_name == None:
        plt.show()
    else:
        plt.savefig(f"lcs/{file_name}.pdf")
    
    plt.close()
    plot_lc_legend(filename=f"lcs/legend.pdf")

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
        plt.yticks(rotation=45)
    
    fig.set_figwidth(18)
    fig.set_figheight(18)
    
    for label in disp.text_.ravel():
        if n_class > 10:
            label.set_fontsize(12)
        elif n_class <= 10 and n_class > 3:
            disp.ax_.tick_params(axis='both', labelsize=40)
            label.set_fontsize('xx-large')
        else:
            disp.ax_.tick_params(axis='both', labelsize=40)
            label.set_fontsize('xx-large')
    
    if title:
        disp.ax_.set_xlabel("Predicted Label", fontsize=60)
        disp.ax_.set_ylabel("True Label", fontsize=60)
        disp.ax_.set_title(title, fontsize=60)
    
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
        for j, c_ in enumerate(leaf_names):
            if c_ == c:
                ax1.plot(df['days_since_trigger'].to_numpy(), df[f"{c_}_mean"].to_numpy(), linewidth=3, linestyle='--', color=color_arr[j])
                ax2.plot(df['days_since_trigger'].to_numpy(), df[f"{c_}_mean"].to_numpy(), linewidth=3, linestyle='--', color=color_arr[j])
            else:
                ax1.plot(df['days_since_trigger'].to_numpy(), df[f"{c_}_mean"].to_numpy(), linewidth=1, color=color_arr[j])
                ax2.plot(df['days_since_trigger'].to_numpy(), df[f"{c_}_mean"].to_numpy(), linewidth=1, color=color_arr[j])

            if show_uncertainties:
                ax1.fill_between(df['days_since_trigger'].to_numpy(), np.minimum(1, df[f"{c_}_mean"].to_numpy() + df[f"{c_}_std"].to_numpy()), np.maximum(0, df[f"{c_}_mean"].to_numpy() - df[f"{c_}_std"].to_numpy()), alpha=0.2, color=color_arr[j])
                ax2.fill_between(df['days_since_trigger'].to_numpy(), np.minimum(1, df[f"{c_}_mean"].to_numpy() + df[f"{c_}_std"].to_numpy()), np.maximum(0, df[f"{c_}_mean"].to_numpy() - df[f"{c_}_std"].to_numpy()), alpha=0.2, color=color_arr[j])
        # High probability stuff - linear scale
        break_point = 0.25
        ax1.set_ylim(break_point, 1.01)  # outliers only

        # Low probability stuff - log scale
        ax2.set_ylim(-0.01, break_point)  # most of the data
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

        ax1.set_xlim(-20, 200)
        ax2.set_xlim(-20, 200)

        ax2.set_xlabel('Days from first detection', fontsize='xx-large')
        ax1.set_ylabel('Mean Class score', fontsize='x-large')
        ax1.set_title(f"True Class: {c}", fontsize='xx-large')

        plt.tight_layout()
        plt.savefig(f"{model_dir}/days_since_trigger/{i}.pdf")
        plt.close()

        plot_legend(leaf_names, filename=f"{model_dir}/days_since_trigger/legend.pdf")

def plot_roc_curves(y_true, y_pred, labels, title=None, img_file=None):

    chance = np.arange(-0.001,1.01,0.001)
    if y_pred.shape[1] <10:
        plt.figure(figsize=(12,12))
    else:
        plt.figure(figsize=(12,16))
    plt.plot(chance, chance, '--', color='black')

    color_arr=[cm(1.*i/y_true.shape[1]) for i in range(y_true.shape[1])]

    n_classes = y_true.shape[1]   
    fpr_all = np.zeros((n_classes, len(chance)))
    tpr_all = np.zeros((n_classes, len(chance)))
    macro_auc = 0

    for i, label in enumerate(labels):

        score = roc_auc_score(y_true[:, i], y_pred[:, i])
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])

        macro_auc += score
        fpr_all[i, :] = chance
        tpr_all[i, :] = np.interp(chance, fpr, tpr)

        plt.plot(fpr, tpr, label=f"{label} (AUC = {score:.2f})", color=color_arr[i])

    macro_auc = macro_auc/y_true.shape[1]
    fpr_macro = np.mean(fpr_all, axis=0)
    tpr_macro = np.mean(tpr_all, axis=0)

    plt.plot(fpr_macro, tpr_macro, linestyle=':', linewidth=4 , label=f"Macro avg (AUC = {macro_auc:.2f})", color='red')

    plt.xlabel('False Positive Rate', fontsize=40)
    plt.ylabel('True Positive Rate', fontsize=40)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2, fontsize = 20)
    plt.title(title, fontsize=40)
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

    fig.tight_layout()
        
    anim = animation.FuncAnimation(fig, animate, frames=len(images), interval=500)

    if gif_file:
        # Save the animation as a GIF
        anim.save(gif_file)

def make_training_history_plot(model_dir):

    window_size = 10

    df = pd.read_csv(f"{model_dir}/loss_history.csv")

    avg_train_losses = df['Avg_train_loss']
    avg_val_losses = df['Avg_val_loss']

    rolling_train = []
    rolling_val = []
    s = []

    for i in range(len(avg_train_losses) - window_size):

        rolling_train.append(np.mean(avg_train_losses[i:i+window_size]))
        rolling_val.append(np.mean(avg_val_losses[i:i+window_size]))
        s.append(i) #s.append(i + window_size)

    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(5, 7), layout="constrained")

    axs[0].plot(list(range(len(avg_train_losses))), np.log(avg_train_losses), label='Train Loss', color='C0', alpha=0.5)
    axs[0].plot(s, np.log(rolling_train), label='Rolling Avg Train Loss', color='C1')

    axs[0].set_ylabel("Mean log loss", fontsize='x-large')
    axs[0].legend()
    axs[0].set_xticks([])

    axs[1].plot(list(range(len(avg_val_losses))), np.log(avg_val_losses), label='Validation Loss', color='C0', alpha=0.5)
    axs[1].plot(s, np.log(rolling_val), label='Rolling Avg Validation Loss', color='C1')

    axs[1].set_xlabel("Epoch", fontsize='x-large')
    axs[1].set_ylabel("Mean log loss", fontsize='x-large')
    axs[1].legend()

    axs[0].set_ylim(-3.4, -1)
    axs[1].set_ylim(-3.4, -1)

    plt.savefig(f"{model_dir}/training_history.pdf")
    plt.close()

def make_z_plots(a_classes, redshifts, X_static, model_dir):

    unique_classes = np.unique(a_classes)

    class_z = {}
    class_z_ddf = {}
    class_z_non_ddf = {}

    max_z = 0
    n = 0

    # Get flags for DDFs
    f1 = np.array([X_static[i]['MW_plane_flag'] for i in range(len(X_static))])
    f2 = np.array([X_static[i]['ELAIS_S1_flag'] for i in range(len(X_static))])
    f3 = np.array([X_static[i]['XMM-LSS_flag'] for i in range(len(X_static))])
    f4 = np.array([X_static[i]['Extended_Chandra_Deep_Field-South_flag'] for i in range(len(X_static))])
    f5 = np.array([X_static[i]['COSMOS_flag'] for i in range(len(X_static))])

    ddf_flag = f1 | f2 | f3 | f4 | f5

    for i, c in enumerate(unique_classes):


        idx = list(np.where(np.array(a_classes) == c)[0])

        z = []
        z_ddf = []
        z_non_ddf = []
        for i in idx:
            if redshifts[i] != -9:
                z.append(redshifts[i])

                if ddf_flag[i] == 1:
                    z_ddf.append(redshifts[i])
                else:
                    z_non_ddf.append(redshifts[i])

                if redshifts[i] > max_z:
                    max_z = redshifts[i]
        
        if len(z) > 10:
            class_z_non_ddf[c] = z_non_ddf
            class_z_ddf[c] = z_ddf
            class_z[c] = z
            n += 1

    fig, axs = plt.subplots(ncols=1, nrows=n, figsize=(5, n*1))

    i = 0
    bins = np.arange(0, max_z + 0.5, 0.5)
    for key in class_z:

        z = class_z[key]
        z_ddf = class_z_ddf[key]
        z_non_ddf = class_z_non_ddf[key]

        ax = axs[i]
        ax.hist(np.array(z), bins=bins, color='black', label=key, fill=False, alpha=1, linewidth=1, histtype='step')
        ax.hist(np.array(z_non_ddf), bins=bins, color='#008080', label=key, fill=True, alpha=0.8, histtype='stepfilled')
        ax.hist(np.array(z_ddf), bins=bins, color='#FF6645', label=key, fill=True, alpha=0.8,  histtype='stepfilled')

        ax.set_xlim(0, max_z)
        #ax.set_ylim(0, 4)
        ax.annotate(f"{key} | Count: {len(z)}", xy=(0.6,0.85),xycoords='axes fraction',fontsize='large')
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_yscale("log")
        #ax.set_yticks([])

        if i == int(n/2):
            ax.set_ylabel('Count', fontsize='x-large')
        else:
            ax.set_ylabel('')

        i += 1

        if i != n:
            ax.set_xticks([])
        else:
            ax.set_xlabel('Redshift', fontsize='x-large')

    plt.tight_layout()
    plt.savefig(f"{model_dir}/z_dist.pdf")