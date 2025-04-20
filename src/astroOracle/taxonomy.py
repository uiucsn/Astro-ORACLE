import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from tensorflow import keras

source_node_label = 'Alert'

# Elasticc class to Astrophysical class
class_map = {
                'SNII-NMF': 'SNII', 
                'SNIc-Templates': 'SNIb/c', 
                'CART': 'CART', 
                'EB': 'EB', 
                'SNIc+HostXT_V19': 'SNIb/c', 
                'd-Sct': 'Delta Scuti', 
                'SNIb-Templates': 'SNIb/c', 
                'SNIIb+HostXT_V19': 'SNII', 
                'SNIcBL+HostXT_V19': 'SNIb/c', 
                'CLAGN': 'AGN', 
                'PISN': 'PISN', 
                'Cepheid': 'Cepheid', 
                'TDE': 'TDE', 
                'SNIa-91bg': 'SNI91bg', 
                'SLSN-I+host': 'SLSN', 
                'SNIIn-MOSFIT': 'SNII', 
                'SNII+HostXT_V19': 'SNII', 
                'SLSN-I_no_host': 'SLSN', 
                'SNII-Templates': 'SNII', 
                'SNIax': 'SNIax', 
                'SNIa-SALT3': 'SNIa', 
                'KN_K17': 'KN', 
                'SNIIn+HostXT_V19': 'SNII', 
                'dwarf-nova': 'Dwarf Novae', 
                'uLens-Binary': 'uLens', 
                'RRL': 'RR Lyrae', 
                'Mdwarf-flare': 'M-dwarf Flare', 
                'ILOT': 'ILOT', 
                'KN_B19': 'KN', 
                'uLens-Single-GenLens': 'uLens', 
                'SNIb+HostXT_V19': 'SNIb/c', 
                'uLens-Single_PyLIMA': 'uLens'
            }

def get_taxonomy_tree():

    # Graph to store taxonomy
    tree = nx.DiGraph(directed=True)

    tree.add_node('Alert', color='red')

    # Level 1
    level_1_nodes = ['Transient', 'Variable']
    tree.add_nodes_from(level_1_nodes)
    tree.add_edges_from([('Alert', level_1_node) for level_1_node in level_1_nodes])

    # Level 2a nodes for Transients
    level_2a_nodes = ['SN', 'Fast', 'Long']
    tree.add_nodes_from(level_2a_nodes)
    tree.add_edges_from([('Transient', level_2a_node) for level_2a_node in level_2a_nodes])

    # Level 2b nodes for Transients
    level_2b_nodes = ['Periodic', 'AGN']
    tree.add_nodes_from(level_2b_nodes)
    tree.add_edges_from([('Variable', level_2b_node) for level_2b_node in level_2b_nodes])

    # Level 3a nodes for SN Transients
    level_3a_nodes = ['SNIa', 'SNIb/c', 'SNIax', 'SNI91bg', 'SNII']
    tree.add_nodes_from(level_3a_nodes)
    tree.add_edges_from([('SN', level_3a_node) for level_3a_node in level_3a_nodes])

    # Level 3b nodes for Fast events Transients
    level_3b_nodes = ['KN', 'Dwarf Novae', 'uLens', 'M-dwarf Flare']
    tree.add_nodes_from(level_3b_nodes)
    tree.add_edges_from([('Fast', level_3b_node) for level_3b_node in level_3b_nodes])

    # Level 3c nodes for Long events Transients
    level_3c_nodes = ['SLSN', 'TDE', 'ILOT', 'CART', 'PISN']
    tree.add_nodes_from(level_3c_nodes)
    tree.add_edges_from([('Long', level_3c_node) for level_3c_node in level_3c_nodes])

    # Level 3d nodes for periodic stellar events
    level_3d_nodes = ['Cepheid', 'RR Lyrae', 'Delta Scuti', 'EB'] 
    tree.add_nodes_from(level_3d_nodes)
    tree.add_edges_from([('Periodic', level_3d_node) for level_3d_node in level_3d_nodes])

    return tree

def get_prediction_probs(y_pred):

    tree = get_taxonomy_tree()

    # Create a new array to store pseudo conditional probabilities.
    pseudo_probabilities = np.copy(y_pred)

    level_order_nodes = nx.bfs_tree(tree, source=source_node_label).nodes()
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
        masks.append(np.array(parents) == parent)
    
    for mask in masks:
        pseudo_probabilities[:, mask] = np.exp(y_pred[:, mask]) / np.sum(np.exp(y_pred[:, mask]))

    # Add weights to edges based on the probabilities.
    level_order_nodes = list(level_order_nodes)
    for i in range(len(level_order_nodes)):

        node = level_order_nodes[i]
        parent = parents[i]
        weight = pseudo_probabilities[0][i]

        if parent != '':
            tree[parent][node]['weight'] = weight
    
    return pseudo_probabilities, tree

def get_most_likely_path(tree, path, source=source_node_label):

    # Get all the children to the node
    successors = list(tree.successors(source))

    # If you are at the leaf node, exit and return the full path
    if len(successors) == 0:
        return path

    # Else, recursively take the most likely step at each node in your decision tree
    weights = []
    children = []

    # Loop through all children and find the path that has highest probability
    for node in successors:

        w = tree[source][node]['weight']
        weights.append(w)
        children.append(node)

    # Find the highest probability step and add to the list
    idx = np.argmax(weights)
    next_node = children[idx]
    path.append(next_node)

    # Recurse and return
    return get_most_likely_path(tree, path, next_node)

def get_highest_prob_path(tree, source=source_node_label):

    leaf_nodes = [x for x in tree.nodes() if tree.out_degree(x)==0 and tree.in_degree(x)==1]
    leaf_probs = []

    for leaf in leaf_nodes:
        path = nx.shortest_path(tree, source, leaf)
        weight_prod = np.prod([tree.get_edge_data(u, v)['weight'] for u, v in zip(path[:-1], path[1:])])
        leaf_probs.append(weight_prod)
    
    idx = np.argmax(leaf_probs)
    return leaf_probs, nx.shortest_path(tree, source, leaf_nodes[idx])

def plot_pred_vs_truth(true, pred, X_ts, X_static, tree):

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))

    pos = graphviz_layout(tree, prog='dot')

    # Plot the ground truth and predicted probs
    nx.draw_networkx(tree, ax=axes[0][0], with_labels=True, font_weight='bold', arrows=True, node_color=true, font_size = 8, pos=pos, cmap='Wistia')
    nx.draw_networkx(tree, ax=axes[0][1], with_labels=True, font_weight='bold', arrows=True, node_color=pred, font_size = 8, pos=pos, cmap='Wistia')

    pos = graphviz_layout(tree, prog='dot')
    nx.draw_networkx(tree, ax=axes[1][0], pos = pos, font_weight='bold', font_size = 8, node_color='white')
    labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in tree.edges(data=True)}
    nx.draw_networkx_edge_labels(tree, ax=axes[1][0], pos = pos, edge_labels = labels)

    time = X_ts[0, :, 0] * 100 # Scaled time
    detection_flag = X_ts[0, :, 1] 
    cal_flux = X_ts[0, :, 2] # Scaled flux 
    cal_flux_err = X_ts[0, :, 3]
    c = X_ts[0, :, 4]

    fmts = np.where((detection_flag) == 1, '*', '.')

    # Plot flux time series
    for i in range(len(time)):
        axes[1][1].errorbar(x=time[i], y=cal_flux[i], yerr=cal_flux_err[i], fmt=fmts[i], markersize = '10')

    axes[1][1].set_xlabel('Time since first observation')
    axes[1][1].set_ylabel('Calibrate Flux')

    # patches = [mpatches.Patch(color=f"C{n}", label=band, linewidth=1) for band, n in zip(lsst_bands, range(4,10))]
    # axes[1][1].legend(handles=patches)

    plt.tight_layout()
    plt.show()  

def get_astrophysical_class(elasticc_class):
    
    tree = get_taxonomy_tree()
    leaf_nodes = [x for x in tree.nodes() if tree.out_degree(x)==0 and tree.in_degree(x)==1]
    assert sorted(leaf_nodes) == sorted(list(set(class_map.values())))

    return class_map[elasticc_class]

def get_classification_labels(astrophysical_class):

    tree = get_taxonomy_tree()
    leaf_nodes = [x for x in tree.nodes() if tree.out_degree(x)==0 and tree.in_degree(x)==1]

    assert astrophysical_class in sorted(list(set(class_map.values()))), f'astrophysical_class {astrophysical_class} was not one of elasticc classes'
    assert astrophysical_class in leaf_nodes, f'astrophysical_class {astrophysical_class} was not one of the leaf nodes in the taxonomy'


    # Do a level order traversal of the tree to get an ordering of the nodes
    level_order_nodes = nx.bfs_tree(tree, source=source_node_label).nodes()

    # Find the path from alert to the astrophysical_class
    path = nx.shortest_path(tree, source=source_node_label, target=astrophysical_class)

    labels = np.zeros(len(level_order_nodes))
    for node in path:
        idx = list(level_order_nodes).index(node)
        labels[idx] = 1
    
    return level_order_nodes, labels

def plot_colored_tree(labels):

    # Get the tree and labels
    tree = get_taxonomy_tree()

    pos = graphviz_layout(tree, prog='dot')
    nx.draw_networkx(tree, with_labels=True, font_weight='bold', arrows=True, node_color=labels, font_size = 8, pos=pos, cmap='Wistia')

    plt.show()

if __name__=='__main__':

    tree = get_taxonomy_tree()

    elasticc_class = 'KN_B19'
    astrophysical_class = get_astrophysical_class(elasticc_class)
    nx.shortest_path(tree, source='Alert', target=astrophysical_class)

    print(get_classification_labels(astrophysical_class))

    pos = graphviz_layout(tree, prog='dot')
    nx.draw_networkx(tree, with_labels=True, font_weight='bold', arrows=True, node_color='white', font_size = 13, pos=pos)
    plt.tight_layout()

    plt.title('Taxonomy for hierarchical classification.', fontsize=15)

    plt.show()

    _, labels = get_classification_labels(astrophysical_class)
    plot_colored_tree(labels)