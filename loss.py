import torch 
import numpy as np
import networkx as nx
import torch.nn.functional as F

from taxonomy import source_node_label, get_taxonomy_tree
from LSTM_model import LSTMClassifier

class WHXE_Loss:

    def __init__(self, tree) -> None:
        
        # The taxonomy tree is used for computing soft max at different levels and computing the WHXE loss
        self.tree = tree

        # Do a level order traversal of the tree to get an ordering of the nodes
        self.level_order_nodes = nx.bfs_tree(self.tree, source=source_node_label).nodes()

        # Find the parent for each node in the tree, ordered using level order traversal
        self.parents = [list(self.tree.predecessors(node)) for node in self.level_order_nodes]
        for idx in range(len(self.parents)):

            # Make sure the graph is a tree.
            assert len(self.parents[idx]) == 0 or len(self.parents[idx]) == 1, 'Number of parents for each node should be 0 (for root) or 1.'
            
            if len(self.parents[idx]) == 0:
                self.parents[idx] = ''
            else:
                self.parents[idx] = self.parents[idx][0]

        # Finding unique parents for masking
        unique_parents = list(set(self.parents))
        unique_parents.sort()

        # Create masks for applying soft max and calculating loss values.
        self.masks = []
        for parent in unique_parents:
            self.masks.append(np.array(self.parents) == parent)
    
    def masked_softmax(self, y_pred):

        # Create a new array to store pseudo conditional probabilities.
        y_pred = y_pred.detach()
        pseud_probabilities = np.zeros(y_pred.shape)

        # Apply soft max to each set of siblings
        for mask in self.masks:
            pseud_probabilities[:, mask] = F.softmax(y_pred[:, mask], dim = 1)

        return pseud_probabilities
    
    def compute_loss(self, y_pred, target_probabilities):

        # Apply hierarchical soft max to get pseudo probability outputs using the data from the machine learning models.
        pred_probabilities = self.masked_softmax(y_pred)

        pass



if __name__=='__main__':

    tree = get_taxonomy_tree()
    loss = WHXE_Loss(tree)

    learning_rate = 0.001
    ts_input_dim = 5
    static_input_dim = 5
    lstm_hidden_dim = 64
    output_dim = len(loss.level_order_nodes)
    lstm_num_layers = 4
    batch_size = 4

    model = LSTMClassifier(ts_input_dim, static_input_dim, lstm_hidden_dim, output_dim, lstm_num_layers)

    ts_length = 5
    input_ts = torch.randn(batch_size, ts_length, ts_input_dim)
    input_static = torch.randn(batch_size, static_input_dim)

    outputs = model(input_ts, input_static)

    loss.masked_softmax(outputs)

