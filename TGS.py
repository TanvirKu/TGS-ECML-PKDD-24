# -*- coding: utf-8 -*-
"""
Truss-based Graph Sparsification
"""

#!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
#!pip install dgl
# import dgl

import networkx as nx
import time
import random
import numpy as np
global avg_truss_calc_time, truss_change_operation_time
import torch
import torch_geometric.utils as pyg_utils
from CustomDataset import CustomDataset
from LGC import Dataset_Operation


import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--cutoff', type = int, default= 3, help = 'The cutoff threshold')
parser.add_argument('--dataset_name', type = str, default= 'PROTEINS', help = 'DD/PROTEINS/NCI1/NCI109/PTC/IMDB-BINARY/IMDB-MULTI/REDDIT-BINARY')
parser.add_argument('--epts', nargs='+', type= float, default= [3.00, 3.25, 3.50, 3.75, 4.00], help = 'A list of edge pruning threshold (ept) \
                    PROTEINS/DD: [3.00, 3.25, 3.5, 3.75, 4.00]; \
                    IMDB-BINARY/MULTI:[3, 4, 5, 6, 7]\
                    REDDIT-BINARY: [3.00, 3.50, 4.00]; \
                    NCI1/NCI109: [2.50, 3.00]; PTC:[2.50]')
parser.add_argument('--dense', type = bool, default= False, help = 'dense is only used for diffpool model')

class TrussOperation:

    def __init__(self, dataset_name, cutoff, dense):
        self.cutoff = cutoff # 3
        # self.ept = ept # 3.5
        self.dataset_name = dataset_name 
        self.dense = False 
    
    def set_ept(self, ept):
        self.ept = ept

    ###############################################################################
    ## Count Support of Each Edge
    def support_count(self, G):
    
        """
          Calculate the support count of each edge in graph G.
          
          The support count of an edge (u, v) is defined as the number of
          common neighbors of u and v, i.e., the number of vertices that are
          adjacent to both u and v. This function iterates through all edges
          of G, calculates the support count for each edge, and stores the
          results in a dictionary.
          
          Parameters:
          - G (NetworkX graph): The input graph.
          
          Returns:
          - dict: A dictionary where the keys are edges (u, v) and the values
                  are the support count of each edge.
        """
    
        support_dict = dict()
          
        for u, v in G.edges():
            N_u = G.neighbors(u)
            N_v = G.neighbors(v)
            support = set(N_u).intersection(set(N_v))
            support_dict[tuple(sorted([u,v]))] = len(support)
    
        return support_dict

    ###############################################################################
    ## Assign trussness score to edges
    def get_all_edges_trussness_in_graph(self, G):
    
        """
          Compute and assign the trussness score to each edge in graph G.
      
          This function iteratively finds the k-truss subgraph for increasing values
          of k, starting from k=2. For each value of k, it determines the edges that
          belong to the (k-1)-truss but not the k-truss. For these edges, it assigns
          the trussness score as k-1 and also calculates and assigns their support.
          The trussness score and support are stored in the 'weight' attribute of the
          edges in a copy of the original graph G.
      
          Parameters:
          - G (NetworkX graph): The input graph.
      
          Returns:
          - G_main (NetworkX graph): A copy of G where each edge has an assigned
                    'weight' attribute containing a dictionary with the trussness and
                    support.
          - trussness_dict (dict): A dictionary where the keys are trussness scores
                    and the values are lists of edges that have the corresponding trussness.
        """
    
        i = 2
        trussness_dict = dict()
        G_main = G.copy()
          
        support_dict = self.support_count(G_main)
          
          
        G1 = None
        total_edges = 0
        
        while (1):
            G = nx.k_truss(G, i)
          
            if i > 2:
                edges = set(list(G1.edges())) - set(list(G.edges())) # get_trussness_edges(edge_list1, edge_list2) #
                edges = list(edges)
                # edges = get_trussness_edges(list(G1.edges()), list(G.edges()))
          
                for u, v in edges:
                    ### sp is the support edge
                    sp = None
                    if (u, v) in support_dict:
                        sp = (u, v)
                    else:
                        sp = (v, u)
              
                    if G_main.has_edge(u, v):
                        G_main[u][v]['weight'] = {'trussness':i-1, 'support': support_dict[sp]}
              
                if len(edges) != 0:
                    trussness_dict[i-1] = edges
                total_edges += len(edges)
          
            # print(len(G.edges()), len(edges))
            if len(G.edges()) == 0 and len(edges) == 0:
                break
            G1 = G.copy()
            i += 1
    
        return G_main, trussness_dict

    #######################################################################################
    ### Calculate edges trussness after pruning
    def calculate_trussness_after_pruning(self, G, u, v):
    
        """
        This function calculates the trussness of the edges after pruning.
      
        Parameters:
            G (nx.Graph): The graph to calculate on.
            u, v (int): The nodes of the edge to remove.
      
        Returns:
            nx.Graph: The graph with recalculated trussness.
        """
    
        commNb = list(nx.common_neighbors(G, u, v))
        # print(f"Removing Edge{u, v}----------------------common_neighbors: {commNb}")
          
        u_edges = [tuple(sorted([u, i])) for i in commNb]
        v_edges = [tuple(sorted([v, i])) for i in commNb]
          
        support_change_edges = u_edges
        support_change_edges.extend(v_edges)
          
        # global_concered_edges = list()
          
        support_change_edges = set([tuple(sorted(item)) for item in support_change_edges])
        trussness_changed_edges = list()
        ### we observe only those edges whoose support has changed
        for u, v in support_change_edges:
            ### reduce the support by one of all edges having triangular relation with E(u, v)
            G[u][v]['weight']['support'] -= 1
            sp = G[u][v]['weight']['support']
            k = G[u][v]['weight']['trussness']
            
            # print(f'Support Change Edge:{u,v}\t sup :',sp,'\t truss :',tr)
            
            ### Check if the trussness of an edge greater than maximum truss value for a support or not
            ### if the max_truss value less than the current trussness , change the trussness value as
            ### the max_truss value
    
            if (sp) < (k-2):
                G[u][v]['weight']['trussness'] = (sp + 2)
          
                G.nodes[u]['nb_with_trussness'][v] = (sp + 2)
                G.nodes[v]['nb_with_trussness'][u] = (sp + 2)
          
                ### Calculate Average \ min
                G.nodes[u]['avg_nb_trussness'] = np.mean( list(G.nodes[u]['nb_with_trussness'].values())) ## np.mean
                G.nodes[v]['avg_nb_trussness'] = np.mean( list(G.nodes[v]['nb_with_trussness'].values())) ## np.mean
                
                trussness_changed_edges.append((u,v))
    
        affected_edges = set()
    
        for u, v in trussness_changed_edges:
            common_nbs = nx.common_neighbors(G, u, v)
            for w in common_nbs:
                affected_edges.add(tuple(sorted([u,w])))
                affected_edges.add(tuple(sorted([v,w])))
    
        filtered_affected_edges = affected_edges - set(trussness_changed_edges)
    
        for u, v in filtered_affected_edges:
    
            change_trussness = True
            common_nbs = nx.common_neighbors(G, u, v)
      
            current_trussness = G[u][v]['weight']['trussness']
            count = 0
    
            for w in common_nbs:
                
                sp_temp = min(G[u][w]['weight']['support'], G[v][w]['weight']['support'] )

                if sp_temp >= (current_trussness - 2):
                    count += 1
      
                if count == (current_trussness - 2) :
                    change_trussness = False
                    break
    
            if change_trussness:
                current_trussness -= 1
                G[u][v]['weight']['trussness'] =  current_trussness
      
                G.nodes[u]['nb_with_trussness'][v] = current_trussness
                G.nodes[v]['nb_with_trussness'][u] = current_trussness
      
                G.nodes[u]['avg_nb_trussness'] = np.mean( list(G.nodes[u]['nb_with_trussness'].values())) ### np.mean
                G.nodes[v]['avg_nb_trussness'] = np.mean( list(G.nodes[v]['nb_with_trussness'].values())) ### np.mean
                
        return G

    ### Getting Node attributes: neighbors trussness and average neighborhood trussness
    def get_node_arrtibutes(self, G):
      
        """
        This function calculate the attributes of each node.The attributes are the 
        neighbors of a node 'n' along with their edge trussness and the average of 
        those neighborhood edge trussness. 
        
        Input: A plain graph  without nodes's attributes 
        Output: Graph with nodes attribute 
        
        """
        neighborhood_trussness_dict = dict()
        for node in G.nodes():
            neighbors_trussvalue_dict = dict()
            neighbors = G[node]
            for nb in neighbors:
                neighbors_trussvalue_dict[nb] = G[node][nb]['weight']['trussness']
      
            # Check if neighbors_trussvalue_dict is empty
            avg_trussness = 0
            if not neighbors_trussvalue_dict:
                avg_trussness = 0  # or np.nan, depending on your preference
            else:
                avg_trussness = np.mean(list(neighbors_trussvalue_dict.values())) ## np.mean
      
            neighborhood_trussness_dict[node] = {'nb_with_trussness' : neighbors_trussvalue_dict,\
                                               'avg_nb_trussness' : avg_trussness }
        nx.set_node_attributes(G, neighborhood_trussness_dict)
        
        return G

    ########################### Truss based greedy sparsification  #############################
    def truss_based_sparsification(self, Gmain):
        """
        This function sparsifies the dense regions of a graph those contribute in 
        redundant message passing in GNNs. 
        
        Input: Original graph with edge trussness 
        Ouput: Sparsified graph 
        """
        Gcom = Gmain.copy()
        com_high_truss_edge_dict = {(u, v):Gmain[u][v]['weight']['trussness'] for u, v in Gcom.edges() \
                                   if Gmain[u][v]['weight']['trussness'] >= self.cutoff}
        
        ### Shuffle and sort the high truss edges 
        ### We shuffle the edges to ensure adjacent edges not come sequentially during pruning 
        ### We sort edges becaause it reduces recursion 
        com_high_truss_edge_dict = self.shuffle_dictonary(com_high_truss_edge_dict)
        com_high_truss_edge_dict = dict(sorted(com_high_truss_edge_dict.items(), key = lambda item: item[1], reverse= True))
      
        no_of_high_truss_edges = len(com_high_truss_edge_dict)
        # print('Number of High truss Edges: ', no_of_high_truss_edges)
        ### number of high truss edges
        n = int(no_of_high_truss_edges)
    
        ### pruned edges in graph
        p_e_comm = 0
        time1 = time.time()
        count = 0
      
        for i in range(1, n+1):
      
            idx = 0
        
            if len(list(com_high_truss_edge_dict.keys())) > 0:
                u, v = list(com_high_truss_edge_dict.keys())[idx]
            else:
                break
      
            trussness = 0
            if Gmain.has_edge(u, v):
                trussness = com_high_truss_edge_dict[(u, v)]###Gmain[u][v]['weight']['trussness']
      
            # Gmain, trussness_dict = get_all_edges_trussness_in_graph(Gmain)
            u_nb_trussness = Gmain.nodes[u]['avg_nb_trussness']
            v_nb_trussness = Gmain.nodes[v]['avg_nb_trussness']
                        
            min_nb_avg_trussness = min([u_nb_trussness, v_nb_trussness]) ### min

            if trussness > 0 and min_nb_avg_trussness >= self.ept:
                if Gmain.has_edge(u, v):
                    Gmain.remove_edge(u, v)
    
                count += 1
                del Gmain.nodes[u]['nb_with_trussness'][v]
                del Gmain.nodes[v]['nb_with_trussness'][u]
    
                Gmain.nodes[u]['avg_nb_trussness'] = np.mean( list(Gmain.nodes[u]['nb_with_trussness'].values())) ### np.mean
                Gmain.nodes[v]['avg_nb_trussness'] = np.mean( list(Gmain.nodes[v]['nb_with_trussness'].values())) ### np.mean
                
                Gmain = self.calculate_trussness_after_pruning(Gmain, u, v)

                if count % 500 == 0:
                    time2 = time.time()
                    print(f"After Pruning {count} edges, time: {time2-time1}")
                p_e_comm += 1
    
                ### Delete the edge from the dictionary
                del com_high_truss_edge_dict[(u, v)]
                  
                time2 = time.time()
        return Gmain

    def shuffle_dictonary(self, dictionary):
        items = list(dictionary.items())
        random.shuffle(items)
        return dict(items)

    def data_preprocessing_pyg(self, PyG_Graph):
        
        """
        This function converts the set of PyG graphs in Networkx graph. Then 
        TGS algorithm applied on those graphs. The sparsified networkx graphs 
        agin converted to PyG graphs 
        
        Input: PyG Graph Dataset 
        Output: Sparsified PyG Graph Dataset
        """
        change = 0
        if not self.dense:
            nxG = pyg_utils.to_networkx(PyG_Graph)
        else:
            numpy_adj_matrix = PyG_Graph.adj.cpu().numpy() 
            nxG = nx.from_numpy_array(numpy_adj_matrix)
        nxG = nx.Graph(nxG)
        nxG.remove_edges_from(nx.selfloop_edges(nxG))
        
        G_main, trussness_dict = self.get_all_edges_trussness_in_graph(nxG)
        G_main  = self.get_node_arrtibutes(G_main)
        
        ### Apply truss-based Graph Sparsification 
        tempG = self.truss_based_sparsification(G_main)
    
        if len(nxG.nodes()) != len(tempG.nodes()) or \
            len(nxG.edges()) != len(tempG.edges()):
            print('Original Graph: ', nxG)
            print('Manipulated Graph: ', tempG)
            change = len(nxG.edges()) - len(tempG.edges())
            #  print('Trussness dict: ', trussness_dict)
        # print(max_truss_Graph)
        for node in tempG.nodes(data=True):
            # List of keys to remove (to avoid modifying the dictionary while iterating over it)
            keys_to_remove = [key for key, value in node[1].items() if isinstance(value, dict)]
        
            # Removing the keys
            for key in keys_to_remove:
                del node[1][key]
                
        for u, v, d in tempG.edges(data=True):
            d['weight'] = float(1.0) if d['weight']['trussness'] > 0 else float(0.0)
        
        new_PyG_graph = pyg_utils.from_networkx(tempG)

        if self.dense:
            sparse_adj_matrix = nx.adjacency_matrix(tempG)
            dense_adj_maxtrix = torch.tensor(sparse_adj_matrix.toarray())
            new_PyG_graph.edge_index = None
            new_PyG_graph.adj = dense_adj_maxtrix.float()
            new_PyG_graph.num_nodes = PyG_Graph.num_nodes
            new_PyG_graph.mask = PyG_Graph.mask
            
        new_PyG_graph.weight = None 
        new_PyG_graph.avg_nb_trussness = None 
        new_PyG_graph.nb_with_trussness = None
        new_PyG_graph.x = PyG_Graph.x.float()
        new_PyG_graph.y = PyG_Graph.y
        
        return new_PyG_graph, change


    def create_custom_sparsified_dataset(self, dataset):
        
        """
        This functions converts a palin PyG graphs dataset to a sparsified PyG graphs' 
        dataset 
        
        Input: Original PyG graphs dataset 
        Output: Sparsified PyG graphs dataset 
        """
        modified_graphs = list()
        custom_dataset = None
        graph_change_count = 0
        total_pruned_edge = 0
        data_count = 0
        
        for graph in dataset:
            data_count += 1
            modified_pyg_graph, change = self.data_preprocessing_pyg(graph)
  
            if change > 0: 
                graph_change_count += 1
                total_pruned_edge += change 
            modified_graphs.append(modified_pyg_graph)
        
            if data_count % 50 == 0:
                print(f"****- Number of explored data: {data_count} -****")
                
        print(f"****# End of explored data: {data_count} #****")
        print(f"------#  Number of Changed Graphs: {graph_change_count} #------")
        custom_dataset = CustomDataset(modified_graphs)

        ### Create an instance of the custom dataset
        return custom_dataset, graph_change_count, total_pruned_edge
    
    def dataset_Processing_TGS(self, dataset):
        
        def save_custom_dataset(dataset, file_name):
            torch.save(dataset, file_name)
            
        def load_custom_dataset(file_name):
            dataset = torch.load(file_name)
            return dataset 
        
        def get_dataset_name(name):
            # path = os.getcwd()
            
            if self.dense:
                dataset_name = 'custom_'+ name +'_dense' + '_' + str(self.ept) +'_.pt'
            else:
                dataset_name = 'custom_'+ name + '_' + str(self.ept) +'_.pt'
            return dataset_name
        
        
        print(f" CREATE CUSTOM DATASET AFTER K-truss Pruning Wehre Edge Pruning Threshold (ept) : {self.ept}")
        custom_dataset, graph_change_count, total_pruned_edge = self.create_custom_sparsified_dataset(dataset)
        custom_dataset.num_classes = dataset.num_classes
        custom_dataset.num_features = dataset.num_features
        
        save_custom_dataset(custom_dataset, get_dataset_name(self.dataset_name))
        
        if graph_change_count > 0:
            edge_pruned_per_graph = total_pruned_edge / graph_change_count
        else:
            edge_pruned_per_graph = 0
        
        with open('Pruned_Info.txt', 'a') as pruned_file: 
            pruned_file.write('Dataset: '+ str(self.dataset_name) +'\n')
            pruned_file.write('Cutoff: '+str(cutoff)+', ept: '+ str(self.ept) +'\n')
            pruned_file.write('# of changed graphs: '+str(graph_change_count)+'\t total # of pruned edges: '+ str(total_pruned_edge)+'\n')
            pruned_file.write('% of edges pruned per graph: '+str(edge_pruned_per_graph)+"\n")
               
            print(f'END of Processing: ept {ept}')
        print("End of Code")
        
#################### ##############################
if __name__ == "__main__":

    args = parser.parse_args() 
    
    cutoff = args.cutoff
    epts = args.epts #[3, 3.25, 3.5, 3.75, 4]
    ### Variable dense is used here for DiffPool model's data manipulation
    dense = args.dense
    dataset_name = args.dataset_name
    
    dataset_Operation = Dataset_Operation(dataset_name = dataset_name)
    dataset = dataset_Operation.Operation()
    truss_op = TrussOperation(dataset_name, cutoff, dense)
    
    
    for ept in epts:
        truss_op.set_ept(ept=ept)
        truss_op.dataset_Processing_TGS(dataset)
    

