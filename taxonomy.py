import networkx as nx
import matplotlib.pyplot as plt

from networkx.drawing.nx_agraph import write_dot, graphviz_layout

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
    level_2b_nodes = ['Periodic', 'Non-Periodic']
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
    level_3d_nodes = ['Cepheid', 'RR Lyrae', 'Delta Scuti', 'EB', 'LPV/Mira']
    tree.add_nodes_from(level_3d_nodes)
    tree.add_edges_from([('Periodic', level_3d_node) for level_3d_node in level_3d_nodes])

    # Level 3e nodes for non periodic stellar events
    level_3e_nodes = ['AGN',]
    tree.add_nodes_from(level_3e_nodes)
    tree.add_edges_from([('Non-Periodic', level_3e_node) for level_3e_node in level_3e_nodes])

    return tree

def get_astrophysical_class(elasticc_class):

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
                    'SNIcBL+HostXT_V19': 'SNII', 
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

    return class_map[elasticc_class]

if __name__=='__main__':

    tree = get_taxonomy_tree()
    # elasticc_class = 'KN_B19'
    # print(nx.shortest_path(tree, source='Alert', target=get_astrophysical_class(elasticc_class)))

    pos = graphviz_layout(tree, prog='dot')
    nx.draw_networkx(tree, with_labels=True, font_weight='bold', arrows=True, node_color='white', font_size = 8, pos=pos)

    plt.title('Taxonomy for hierarchical classification.')
    plt.show()