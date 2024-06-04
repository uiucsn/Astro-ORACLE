import numpy as np
import networkx as nx

from networkx.drawing.nx_agraph import write_dot, graphviz_layout

from taxonomy import source_node_label

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

def get_all_confusion_matrices(y_true, y_pred, tree):
    
    # Get all the probabilites
    pseudo_probabilities, pseudo_conditional_probabilities = get_conditional_probabilites(y_pred, tree)
    
    
    # Find the parents
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

    # Create masks for classification
    masks = []
    for parent in unique_parents:
        masks.append(get_indices_where(parents, parent))
    
    
    # Get the masked softmaxes
    for mask in masks:
        
        true_labels = []
        pred_labels = []
        
        
        for i in range(y_true.shape[0]):
            
            # Find the true label of the object for this mask
            if np.sum(y_true[i, mask]) == 1:
                 # This object belongs to some class in this mask
                true_class_idx = mask[np.argmax(y_true[i, mask])]
                true_labels.append(level_order_nodes[true_class_idx])
            
            
            elif np.sum(y_true[i, mask]) == 0:
                # This object does not belong to some class in this mask
                true_labels.append('Other')
                
            else:
                # This means I fucked up
                assert False, 'This should not have happened. I would offer to help but clearly you should not trust me since you reached an unreachable state'
        
            
            # Find the predicted label
            predicted_class_idx = mask[np.argmax(y_pred[i, mask])]
            pred_labels.append(level_order_nodes[predicted_class_idx])
            
        mask_classes = level_order_nodes[mask] + ['Other']
        plot_confusion_matrix(true_labels, pred_labels, mask_classes)