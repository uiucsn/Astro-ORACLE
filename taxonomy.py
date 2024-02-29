import networkx as nx
import matplotlib.pyplot as plt

# Graph to store taxonomy
tree = nx.Graph()

tree.add_node('Alert', color='red')

# Level 1
level_1_nodes = ['Transient', 'Stellar']
tree.add_nodes_from(level_1_nodes)
tree.add_edges_from([('Alert', level_1_node) for level_1_node in level_1_nodes])

# Level 2a nodes for Transients
level_2a_nodes = ['SN', 'Fast', 'Long']
tree.add_nodes_from(level_2a_nodes)
tree.add_edges_from([('Transient', level_2a_node) for level_2a_node in level_2a_nodes])

# Level 2b nodes for Transients
level_2b_nodes = ['Periodic', 'Non-Periodic']
tree.add_nodes_from(level_2b_nodes)
tree.add_edges_from([('Stellar', level_2b_node) for level_2b_node in level_2b_nodes])

# Level 3a nodes for SN Transients
level_3a_nodes = ['SNa', 'SNIb/c', 'SNIax', 'SNI91bg', 'SNII']
tree.add_nodes_from(level_3a_nodes)
tree.add_edges_from([('SN', level_3a_node) for level_3a_node in level_3a_nodes])

# Level 3b nodes for Fast events Transients
level_3b_nodes = ['KN', 'Dwarf Novae', 'uLens']
tree.add_nodes_from(level_3b_nodes)
tree.add_edges_from([('Fast', level_3b_node) for level_3b_node in level_3b_nodes])

# Level 3c nodes for Long events Transients
level_3c_nodes = ['SLSN', 'TDE', 'ILOT', 'CART', 'PISN']
tree.add_nodes_from(level_3c_nodes)
tree.add_edges_from([('Long', level_3c_node) for level_3c_node in level_3c_nodes])

# Level 3d nodes for periodic stellar events
level_3d_nodes = ['Cepheid', 'RR Lyrae', 'Delta Scuti', 'EB', 'LPV/Mira']
tree.add_nodes_from(level_3d_nodes)
tree.add_edges_from([('Periodic', level_3d_node) for level_3d_node in level_3d_nodes])

# Level 3e nodes for non periodic stellar events
level_3e_nodes = ['AGN',]
tree.add_nodes_from(level_3e_nodes)
tree.add_edges_from([('Non-Periodic', level_3e_node) for level_3e_node in level_3e_nodes])

def get_astrophysical_class(elasticc_class):

    # Elasticc class to Astrophysical class
    class_map = {
        'KN_B19': 'KN',
        'KN_K17': 'KN',
    }

    return class_map[elasticc_class]

if __name__=='__main__':

    elasticc_class = 'KN_B19'

    print(nx.shortest_path(tree, source='Alert', target=get_astrophysical_class(elasticc_class)))

    nx.draw_networkx(tree, with_labels=True, font_weight='bold', arrows=True, node_color='white')
    plt.show()