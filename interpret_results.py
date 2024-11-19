import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from functools import reduce

from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from taxonomy import source_node_label
from vizualizations import plot_confusion_matrix, plot_roc_curves

def get_indices_where(arr, target):
    
    to_return = []
    for i, obs in enumerate(arr):
        if obs == target:
            to_return.append(i)
            
    return to_return

def get_conditional_probabilites(y_pred, tree):
    
    # Create a new arrays to store pseudo (conditional) probabilities.
    pseudo_conditional_probabilities = np.copy(y_pred)
    pseudo_probabilities = np.copy(y_pred)

    level_order_nodes = list(nx.bfs_tree(tree, source=source_node_label).nodes())
    parents = [list(tree.predecessors(node)) for node in level_order_nodes]
    for idx in range(len(parents)):

        # Make sure the graph is a tree.
        assert len(parents[idx]) == 0 or len(parents[idx]) == 1, 'Number of parents for each node should be 0 (for root) or 1.'
        
        if len(parents[idx]) == 0:
            parents[idx] = ''
        else:
            parents[idx] = parents[idx][0]

    # Finding unique parents for masking
    unique_parents = list(set(parents))
    unique_parents.sort()

    # Create masks for applying soft max and calculating loss values.
    masks = []
    for parent in unique_parents:
        masks.append(get_indices_where(parents, parent))
    
    # Get the masked softmaxes
    for mask in masks:
        pseudo_probabilities[:, mask] = np.exp(y_pred[:, mask]) / np.sum(np.exp(y_pred[:, mask]), axis=1, keepdims=True)
        

    for node in level_order_nodes:
        
        # Find path from node to 
        path = list(nx.shortest_path(tree, source_node_label, node))
        
        # Get the index of the node for which we are calculating the pseudo probability
        node_index = level_order_nodes.index(node)
        
        # Indices of all the classes in the path from source to the node for which we are calculating the pseudo probability
        path_indices = [level_order_nodes.index(u) for u in path]
        
        #print(node, path, node_index, path_indices, pseudo_probabilities[:, path_indices])
        
        # Multiply the pseudo probabilites of all the classes in the path so that we get the conditional pseudo probabilites
        pseudo_conditional_probabilities[:, node_index] = np.prod(pseudo_probabilities[:, path_indices], axis = 1)
        
    return pseudo_probabilities, pseudo_conditional_probabilities

def save_leaf_cf_and_rocs(y_true, y_pred, tree, model_dir, plot_title):
    
    # Find the indexes of the leaf nodes i.e. nodes with out degree = 0.
    level_order_nodes = list(nx.bfs_tree(tree, source=source_node_label).nodes())
    n_children = [tree.out_degree(node) for node in level_order_nodes]
    idx = get_indices_where(n_children, 0)

    leaf_labels = np.array(level_order_nodes)[idx]
    
    y_pred_label = [leaf_labels[i] for i in np.argmax(y_pred[:, idx], axis=1)]
    y_true_label = [leaf_labels[i] for i in np.argmax(y_true[:, idx], axis=1)]

    # Make the dirs to store results
    os.makedirs(f"{model_dir}/gif/leaf_cf", exist_ok=True)
    os.makedirs(f"{model_dir}/gif/leaf_roc", exist_ok=True)
    os.makedirs(f"{model_dir}/gif/leaf_csv", exist_ok=True)

    csv_plot_file = f"{model_dir}/gif/leaf_csv/{plot_title}.csv"
    
    plot_confusion_matrix(y_true_label, y_pred_label, leaf_labels, plot_title, f"{model_dir}/gif/leaf_cf/{plot_title}.png")
    plt.close()

    plot_confusion_matrix(y_true_label, y_pred_label, leaf_labels, plot_title, f"{model_dir}/gif/leaf_cf/{plot_title}.pdf")
    plt.close()
    
    plot_roc_curves(y_true[:, idx], y_pred[:, idx], leaf_labels, plot_title, f"{model_dir}/gif/leaf_roc/{plot_title}.png")
    plt.close()

    plot_roc_curves(y_true[:, idx], y_pred[:, idx], leaf_labels, plot_title, f"{model_dir}/gif/leaf_roc/{plot_title}.pdf")
    plt.close()

    report = classification_report(y_true_label, y_pred_label)
    print(report)
    report = classification_report(y_true_label, y_pred_label, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(csv_plot_file)
    print('===========')

def save_all_cf_and_rocs(y_true, y_pred, tree, model_dir, plot_title):
    
    def get_path_length(tree, source, target):
        return len(nx.shortest_path(tree, source=source, target=target)) - 1
    
    # Find the parents
    level_order_nodes = list(nx.bfs_tree(tree, source=source_node_label).nodes())
    depths = [get_path_length(tree, source_node_label, node) for node in level_order_nodes]
    
    # Finding unique depths for masking
    unique_depths = list(set(depths))
    unique_depths.sort()

    # Create masks for classification
    masks = []
    for depth in unique_depths:
        masks.append(get_indices_where(depths, depth))
    
    # Get the masked softmaxes
    for mask, depth in zip(masks, unique_depths):
        
        # Every alert 
        if depth != 0 and depth != 3:
        
            true_labels = []
            pred_labels = []
            
            mask_classes = [level_order_nodes[m] for m in mask]
            for i in range(y_true.shape[0]):

                # Find the true label
                true_class_idx = np.argmax(y_true[i, mask])
                true_labels.append(mask_classes[true_class_idx])

                # Find the predicted label
                predicted_class_idx = np.argmax(y_pred[i, mask])
                pred_labels.append(mask_classes[predicted_class_idx])

            # Create the dirs to save plots
            os.makedirs(f"{model_dir}/gif/level_{depth}_cf", exist_ok=True)
            os.makedirs(f"{model_dir}/gif/level_{depth}_roc", exist_ok=True)
            os.makedirs(f"{model_dir}/gif/level_{depth}_csv", exist_ok=True)

            csv_plot_file = f"{model_dir}/gif/level_{depth}_csv/{plot_title}.csv"

            plot_confusion_matrix(true_labels, pred_labels, mask_classes, plot_title, f"{model_dir}/gif/level_{depth}_cf/{plot_title}.png")
            plt.close()

            plot_confusion_matrix(true_labels, pred_labels, mask_classes, plot_title, f"{model_dir}/gif/level_{depth}_cf/{plot_title}.pdf")
            plt.close()

            plot_roc_curves(y_true[:, mask], y_pred[:, mask], mask_classes, plot_title, f"{model_dir}/gif/level_{depth}_roc/{plot_title}.png")
            plt.close()

            plot_roc_curves(y_true[:, mask], y_pred[:, mask], mask_classes, plot_title, f"{model_dir}/gif/level_{depth}_roc/{plot_title}.pdf")
            plt.close()
            
            report = classification_report(true_labels, pred_labels)
            print(report)
            report = classification_report(true_labels, pred_labels, output_dict=True)
            pd.DataFrame(report).transpose().to_csv(csv_plot_file)
            print('===========')

def save_all_phase_vs_accuracy_plot(model_dir, days = 2 ** np.array(range(11)), levels = ["level_1", "level_2", "leaf"]):

    plt.style.use(['default'])

    # Making the f1 plot
    for level in levels:

        f1 = []
        for d in days:

            df_alpha1 = pd.read_csv(f'{model_dir}/gif/{level}_csv/Trigger + {d} days.csv')
            f1.append(df_alpha1['f1-score'].to_numpy()[-2])

        plt.plot(days, f1, label=level, marker = 'o')

    plt.xlabel("Days from first detection", fontsize='xx-large')
    plt.ylabel("Macro avg F1 score", fontsize='xx-large')

    plt.grid()
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.xscale('log')
    plt.ylim(0.5, 1.01)
    plt.xticks(days, days)
    plt.savefig(f"{model_dir}/f1-performance.pdf")
    plt.savefig(f"{model_dir}/f1-performance.jpg")
    plt.close()

    for level in levels:

        precision = []
        for d in days:

            df_alpha1 = pd.read_csv(f'{model_dir}/gif/{level}_csv/Trigger + {d} days.csv')
            precision.append(df_alpha1['precision'].to_numpy()[-2])

        plt.plot(days, precision, label=level, marker = 'o')

    plt.xlabel("Days from first detection", fontsize='xx-large')
    plt.ylabel("Macro avg precision", fontsize='xx-large')

    plt.grid()
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.xscale('log')
    plt.ylim(0.5, 1.01)
    plt.xticks(days, days)
    plt.savefig(f"{model_dir}/precision-performance.pdf")
    plt.savefig(f"{model_dir}/precision-performance.jpg")
    plt.close()

    for level in levels:

        recall = []
        for d in days:

            df_alpha1 = pd.read_csv(f'{model_dir}/gif/{level}_csv/Trigger + {d} days.csv')
            recall.append(df_alpha1['recall'].to_numpy()[-2])

        plt.plot(days, recall, label=level, marker = 'o')

    plt.xlabel("Days from first detection", fontsize='xx-large')
    plt.ylabel("Macro avg recall", fontsize='xx-large')

    plt.grid()
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.xscale('log')
    plt.ylim(0.5, 1.01)
    plt.xticks(days, days)
    plt.savefig(f"{model_dir}/recall-performance.pdf")
    plt.savefig(f"{model_dir}/recall-performance.jpg")
    plt.close()

def save_class_wise_phase_vs_accuracy_plot(model_dir, days = 2 ** np.array(range(11)), levels = ["level_1", "level_2", "leaf"]):

    plt.style.use(['default'])

    cm = plt.get_cmap('gist_rainbow')

    # Making the f1 plot
    for level in levels:

        class_wise_f1s = {}
        for d in days:

            df_alpha1 = pd.read_csv(f'{model_dir}/gif/{level}_csv/Trigger + {d} days.csv')

            classes = df_alpha1.iloc[:-3,0].to_numpy()
            f1 = df_alpha1['f1-score'].to_numpy()[:-3]


            for i, c in enumerate(classes):
                
                if c not in class_wise_f1s:
                    class_wise_f1s[c] = []
                
                class_wise_f1s[c].append(f1[i])

        for i, c in enumerate(classes):

            if i >= 10:
                linestyle='dotted'
                plt.plot(days, class_wise_f1s[c], label=c, marker = '*', alpha=0.5, linestyle=linestyle)
            else:
                linestyle='dashed'
                plt.plot(days, class_wise_f1s[c], label=c, marker = '.', alpha=0.5, linestyle=linestyle)

        plt.xscale('log')

        if len(classes) > 10:
            plt.ylim(-0.1, 1.01)
            plt.legend(ncol=5, fontsize=7, loc='lower right')
        else:
            plt.legend()
        plt.xticks(days, days)

        #plt.ylim(0, 1.1)


        plt.xlabel("Days from first detection", fontsize='xx-large')
        plt.ylabel("F1 score", fontsize='xx-large')
        plt.tight_layout()

        plt.savefig(f"{model_dir}/per_class_{level}_F1.pdf")
        plt.close()


def merge_performance_tables(model_dir, days=[2,8,64,1024]):

    levels = ['level_1','level_2','leaf']

    for level in levels:

        data_frames = []

        for d in days:

            df = pd.read_csv(f"{model_dir}/gif/{level}_csv/Trigger + {d} days.csv", index_col=0)
            df.drop(columns=["support"], inplace=True)
            df.index.name = 'Class'
            df.rename(columns={'precision': '$p_{' + f"{d}d" + '}$'}, inplace=True)
            df.rename(columns={'recall': '$r_{' + f"{d}d" + '}$'}, inplace=True)
            df.rename(columns={'f1-score': '$f1_{' + f"{d}d" + '}$'}, inplace=True)
            data_frames.append(df)

        df_merged = reduce(lambda  left,right: pd.merge(left,right, how='left',on='Class', sort=False), data_frames)
        print(df_merged.to_latex(float_format="%.2f"))