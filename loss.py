import numpy as np
import networkx as nx
import tensorflow as tf

from tensorflow import keras

from taxonomy import source_node_label, get_taxonomy_tree
from LSTM_model import get_LSTM_Classifier
from dataloader import ts_length

class WHXE_Loss:

    # Implementation of Weighted Hierarchical Cross Entropy loss function by Villar et. al. 2023 (https://arxiv.org/abs/2312.02266)

    def __init__(self, tree, leaf_labels, alpha = 0.5) -> None:

        # Free parameter that determines how much weight is given to different levels in the hierarchy
        self.alpha = alpha

        # The leaf labels of all the objects in the data set, used to compute the class weights
        self.leaf_labels = leaf_labels

        # The taxonomy tree is used for computing soft max at different levels and computing the WHXE loss
        self.tree = tree

        # Do a level order traversal of the tree to get an ordering of the nodes
        self.level_order_nodes = nx.bfs_tree(self.tree, source=source_node_label).nodes()

        # Compute and store parents for each node, ordered by level order traversal of the tree
        self.compute_parents()

        # Compute and store the masks used for soft max
        self.create_masks()

        # Compute and store the path lengths from the root to each node in the tree, ordered by level order traversal of the tree
        self.compute_path_lengths()

        # Compute the class weights, ordered by level order traversal of the tree
        self.compute_class_weights()

    def compute_class_weights(self):

        # Count the total number of events in the data set
        N_all = len(self.leaf_labels)

        # Count the number of labels in the taxonomy
        N_labels = len(self.level_order_nodes)

        # Maintain a dictionary to count the number of occurrences of each node in the tree
        counts_dict = {}
        for node in self.level_order_nodes:
            counts_dict[node] = 0

        # Iterate through all the leaf labels in the data set
        for labels in self.leaf_labels:

            # Compute the path down the taxonomy for each element in the data set
            path = nx.shortest_path(self.tree, source=source_node_label, target=labels)

            # Iterate the counter for the node whenever the node appears in the path.
            for step in path:
                counts_dict[step] += 1

        # Array to store the number of occurrences of each node in the taxonomy for the given data set, ordered by level order traversal of the tree
        N_c = []
        for node in self.level_order_nodes:
            N_c.append(counts_dict[node])

        # Compute the final weights, ordered by level order traversal of the tree
        self.class_weights = N_all / (N_labels * np.array(N_c))

    def compute_parents(self):

        # Find the parent for each node in the tree, ordered using level order traversal
        self.parents = [list(self.tree.predecessors(node)) for node in self.level_order_nodes]
        for idx in range(len(self.parents)):

            # Make sure the graph is a tree.
            assert len(self.parents[idx]) == 0 or len(self.parents[idx]) == 1, 'Number of parents for each node should be 0 (for root) or 1.'
            
            if len(self.parents[idx]) == 0:
                self.parents[idx] = ''
            else:
                self.parents[idx] = self.parents[idx][0]
    
    def create_masks(self):

        # Finding unique parents for masking
        unique_parents = list(set(self.parents))
        unique_parents.sort()

        # Create masks for applying soft max and calculating loss values.
        self.masks = []
        for parent in unique_parents:
            self.masks.append(np.array(self.parents) == parent)

    def compute_path_lengths(self):
        
        # Compute the shortest paths from the root node to each of the other nodes in the tree.
        self.path_lengths = []

        for node in self.level_order_nodes:
            self.path_lengths.append(len(nx.shortest_path(self.tree, source_node_label, node)) - 1)

        self.path_lengths = np.array(self.path_lengths)

        # Compute the secondary weight term, which emphasizes different levels of the tree. See paper for more details.
        self.lambda_term = np.exp(-self.alpha * self.path_lengths)


    def compute_loss(self, y_pred, target_probabilities, epsilon=1e-10):

        total = 0

        # Apply soft max to each set of siblings
        for mask in self.masks:

            logits = tf.boolean_mask(y_pred, mask, axis=1) + epsilon
            masked_soft_maxes = keras.activations.softmax(logits, axis = 1)

            log_p = tf.math.log(masked_soft_maxes)
            result0 = tf.math.subtract(1.0, tf.boolean_mask(target_probabilities, mask, axis=1))
            result1 = tf.math.multiply(log_p, result0)
            result2 = tf.math.multiply(result1, self.class_weights[mask])
            result3 = tf.math.multiply(result2, self.lambda_term[mask])
            result4 = tf.math.reduce_sum(result3, axis=1)
            result5 = tf.math.reduce_mean(result4, axis=0)
    
            total -= result5
            
        return total
    



if __name__=='__main__':
    
    tree = get_taxonomy_tree()
    loss = WHXE_Loss(tree, list(tree.nodes))

    ts_dim = 5
    static_dim = 15
    latent_size = 10
    output_dim = len(list(tree.nodes))

    batch_size = 4

    model = get_LSTM_Classifier(ts_dim, static_dim, output_dim, latent_size, "categorical_crossentropy")

    input_ts = np.random.randn(batch_size, ts_length, ts_dim)
    input_static = np.random.randn(batch_size, static_dim)

    outputs = model.predict([input_ts, input_static])
    print(loss.compute_loss(outputs, outputs))


