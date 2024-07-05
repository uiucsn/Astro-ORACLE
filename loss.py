import numpy as np
import networkx as nx
import tensorflow as tf

from tensorflow import keras

from taxonomy import source_node_label, get_taxonomy_tree
from RNN_model import get_RNN_model
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
            self.masks.append(np.where(np.array(self.parents) == parent,1,0))

    def compute_path_lengths(self):
        
        # Compute the shortest paths from the root node to each of the other nodes in the tree.
        self.path_lengths = []

        for node in self.level_order_nodes:
            self.path_lengths.append(len(nx.shortest_path(self.tree, source_node_label, node)) - 1)

        self.path_lengths = np.array(self.path_lengths)

        # Compute the secondary weight term, which emphasizes different levels of the tree. See paper for more details.
        self.lambda_term = np.exp(-self.alpha * self.path_lengths)

    @keras.saving.register_keras_serializable(package="my_package", name="custom_fn")
    def compute_loss(self, target_probabilities, y_pred, epsilon=1e-10):

        # Go through set of siblings
        for mask in self.masks:
            
            # Get the e^logits
            exps = tf.math.exp(y_pred)

            # Multiply (dot product) the e^logits with the mask to maintain just the e^logits values that belong to this mask. All other values will be zeros.
            masked_exps = tf.math.multiply(exps, mask)

            # Find the sum of the e^logits values that belong to the mask. Do this for each element in the batch separately. Add a small value to avoid numerical problems with floating point numbers.
            masked_sums = tf.math.reduce_sum(masked_exps, axis=1, keepdims=True) + epsilon

            # Compute the softmax by dividing the e^logits with the sume (e^logits)
            softmax = masked_exps/masked_sums

            # (1 - mask) * y_pred gets the logits for all the values not in this mask and zeros out the values in the mask. Add those back so that we can repeat the process for other masks.
            y_pred =  softmax + ((1 - mask) * y_pred)
        
        # At this point we have the masked softmaxes i.e. the pseudo probabilities. We can take the log of these values
        y_pred = tf.math.log(y_pred)

        # Weight them by the level at which the corresponding node appears in the hierarchy
        y_pred = y_pred * self.lambda_term

        # Weight them by the class weight after using the target_probabilities as indicators. Then sum them up for each batch
        v1 = tf.math.reduce_sum(self.class_weights * (y_pred * target_probabilities), axis=1)

        # Finally, find the mean over all batches. Since we are taking logs of numbers <1 (the pseudo probabilities), we have to multiply by -1 to get a +ve loss value.
        v2 = -1 * tf.math.reduce_mean(v1)
            
        return v2
    
class HXE_Loss:

    # Implementation of Weighted Hierarchical Cross Entropy loss function by Villar et. al. 2023 (https://arxiv.org/abs/2312.02266)

    def __init__(self, tree, alpha = 0.5) -> None:

        # Free parameter that determines how much weight is given to different levels in the hierarchy
        self.alpha = alpha

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
            self.masks.append(np.where(np.array(self.parents) == parent,1,0))

    def compute_path_lengths(self):
        
        # Compute the shortest paths from the root node to each of the other nodes in the tree.
        self.path_lengths = []

        for node in self.level_order_nodes:
            self.path_lengths.append(len(nx.shortest_path(self.tree, source_node_label, node)) - 1)

        self.path_lengths = np.array(self.path_lengths)

        # Compute the secondary weight term, which emphasizes different levels of the tree. See paper for more details.
        self.lambda_term = np.exp(-self.alpha * self.path_lengths)

    @keras.saving.register_keras_serializable(package="my_package", name="custom_fn")
    def compute_loss(self, target_probabilities, y_pred, epsilon=1e-10):

        # Go through set of siblings
        for mask in self.masks:
            
            # Get the e^logits
            exps = tf.math.exp(y_pred)

            # Multiply (dot product) the e^logits with the mask to maintain just the e^logits values that belong to this mask. All other values will be zeros.
            masked_exps = tf.math.multiply(exps, mask)

            # Find the sum of the e^logits values that belong to the mask. Do this for each element in the batch separately. Add a small value to avoid numerical problems with floating point numbers.
            masked_sums = tf.math.reduce_sum(masked_exps, axis=1, keepdims=True) + epsilon

            # Compute the softmax by dividing the e^logits with the sume (e^logits)
            softmax = masked_exps/masked_sums

            # (1 - mask) * y_pred gets the logits for all the values not in this mask and zeros out the values in the mask. Add those back so that we can repeat the process for other masks.
            y_pred =  softmax + ((1 - mask) * y_pred)
        
        # At this point we have the masked softmaxes i.e. the pseudo probabilities. We can take the log of these values
        y_pred = tf.math.log(y_pred)

        # Weight them by the level at which the corresponding node appears in the hierarchy
        y_pred = y_pred * self.lambda_term

        # Use the target_probabilities as indicators. Then sum them up for each batch
        v1 = tf.math.reduce_sum(y_pred * target_probabilities, axis=1)

        # Finally, find the mean over all batches. Since we are taking logs of numbers <1 (the pseudo probabilities), we have to multiply by -1 to get a +ve loss value.
        v2 = -1 * tf.math.reduce_mean(v1)
            
        return v2
    
class PAWHXE_Loss:

    # Implementation of a Phase Aware Weighted Hierarchical Cross Entropy loss function by yours truly et al. (2024, or 2025 if I get lazy.....sigh)
    def __init__(self, tree, leaf_labels, alpha = 0.5, beta=0.1) -> None:

        # Free parameter that determines how much weight is given to different levels in the hierarchy
        self.alpha = alpha

        # Free parameter that determines how quickly the alpha parameter decays as a function of % of the light curve observed
        self.beta = beta

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
            self.masks.append(np.where(np.array(self.parents) == parent,1,0))

    def compute_path_lengths(self):
        
        # Compute the shortest paths from the root node to each of the other nodes in the tree.
        self.path_lengths = []

        for node in self.level_order_nodes:
            self.path_lengths.append(len(nx.shortest_path(self.tree, source_node_label, node)) - 1)

        self.path_lengths = np.array(self.path_lengths).astype(np.float32)


    @keras.saving.register_keras_serializable(package="my_package", name="custom_fn")
    def compute_loss(self, target_probabilities, y_pred, fractions, epsilon=1e-10):

        # Go through set of siblings
        for mask in self.masks:
            
            # Get the e^logits
            exps = tf.math.exp(y_pred)

            # Multiply (dot product) the e^logits with the mask to maintain just the e^logits values that belong to this mask. All other values will be zeros.
            masked_exps = tf.math.multiply(exps, mask)

            # Find the sum of the e^logits values that belong to the mask. Do this for each element in the batch separately. Add a small value to avoid numerical problems with floating point numbers.
            masked_sums = tf.math.reduce_sum(masked_exps, axis=1, keepdims=True) + epsilon

            # Compute the softmax by dividing the e^logits with the sume (e^logits)
            softmax = masked_exps/masked_sums

            # (1 - mask) * y_pred gets the logits for all the values not in this mask and zeros out the values in the mask. Add those back so that we can repeat the process for other masks.
            y_pred =  softmax + ((1 - mask) * y_pred)
        
        # At this point we have the masked softmaxes i.e. the pseudo probabilities. We can take the log of these values
        y_pred = tf.math.log(y_pred)

        # Compute the secondary weight term, which emphasizes different levels of the tree at the appropriate phase. See paper for more details.
        decay_term = tf.math.pow(self.beta, fractions)
        decayed_alpha = tf.math.multiply(self.alpha, decay_term)
        lambda_term = tf.math.exp(tf.math.multiply(tf.expand_dims(-decayed_alpha, axis=1), tf.expand_dims(self.path_lengths, axis=0)))

        # Weight them by the level at which the corresponding node appears in the hierarchy
        y_pred = y_pred * lambda_term

        # Weight them by the class weight after using the target_probabilities as indicators. Then sum them up for each batch
        v1 = tf.math.reduce_sum(self.class_weights * (y_pred * target_probabilities), axis=1)

        # Finally, find the mean over all batches. Since we are taking logs of numbers <1 (the pseudo probabilities), we have to multiply by -1 to get a +ve loss value.
        v2 = -1 * tf.math.reduce_mean(v1)
            
        return v2

if __name__=='__main__':
    
    tree = get_taxonomy_tree()
    
    ts_dim = 5
    static_dim = 15
    latent_size = 10
    output_dim = len(list(tree.nodes))

    batch_size = 4

    model = get_RNN_model(ts_dim, static_dim, output_dim, latent_size)

    input_ts = np.random.randn(batch_size, ts_length, ts_dim)
    input_static = np.random.randn(batch_size, static_dim)

    outputs = model.predict([input_ts, input_static])
    
    weighted_loss = WHXE_Loss(tree, list(tree.nodes))
    unweighted_loss = HXE_Loss(tree)

    
    print(weighted_loss.compute_loss(outputs, outputs))
    print(unweighted_loss.compute_loss(outputs, outputs))


