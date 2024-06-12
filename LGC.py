# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 11:54:18 2023

@author: thsou

LGC: Library-based Graph Conversation
"""

from torch_geometric.datasets import TUDataset
import dgl 
from dgl.data import GINDataset
import networkx as nx 
# import os
import os.path as osp
import numpy as np
import torch_geometric.transforms as T
import torch
import torch_geometric.utils as pyg_utils
#from torch.utils.data import Dataset
from torch_geometric.utils import degree
from torch_geometric.transforms import OneHotDegree
from CustomDataset import CustomDataset

####  ["DD", "PTC", "NCI1", "PROTEINS", "NCI109", "IMDB-BINARY", "IMDB-MULTI", "REDDIT-BINARY"] ###
###################################################################################################


class Dataset_Operation:
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        
    def from_dgl_to_pyg_both(self, dataset_dgl, dataset_pyg):
        
        """
        This function converts the dgl datasets to pyg dataset 
        Input: dgl dataset and pyg dataset 
        Output: PyG dataset 
        """
        
        modified_list = list() 
        feature_list = list() 
        all_clear = True
    
        data_count = 0
        for i in range(len(dataset_pyg)):
            
            data_count += 1
            num_nodes_dgl = dataset_dgl[i][0].num_nodes() ### num_nodes_dgl
            num_nodes_pyg = dataset_pyg[i].num_nodes ### num_edges_dgl 
            
            ### Checking each dgl and pyg graphs one by one 
            if num_nodes_dgl != num_nodes_pyg: 
                print("not the similar dataset so we cannot convert")
                all_clear = False
                break
            else: 
                nxG = pyg_utils.to_networkx(dataset_pyg[i])
                new_PyG_graph = pyg_utils.from_networkx(nxG)
                graph_attr_dgl = dataset_dgl[i][0].ndata['attr']
                
                if num_nodes_pyg == graph_attr_dgl.shape[0]:
                    new_PyG_graph.x = graph_attr_dgl
                    for node_features in new_PyG_graph.x:
                        feature_list.append(node_features)
                    new_PyG_graph.y = dataset_pyg[i].y
                else:
                    print("not the similar dataset so we cannot convert")
                    all_clear = False
                    break
        
                modified_list.append(new_PyG_graph)
            
            if data_count % 500 == 0:
                print(f"# of explored data: {data_count}")
        print(f"End of explored data: {data_count}")
        
        if all_clear: 
            custom_dataset = CustomDataset(modified_list, library= 'pyg')
            custom_dataset.y = torch.tensor([graph.y for graph in custom_dataset])
            custom_dataset.x = torch.stack(feature_list)
            custom_dataset.num_classes = len(set([item.item() for item in custom_dataset.y])) 
            custom_dataset.num_features = custom_dataset[0].x.shape[1] 
                
        return custom_dataset
    
    def to_pyg_only_from_dgl(self, dataset_dgl):
        
        """
        This method is specified for the PTC dataset 
        
        """
        modified_list = list() 
        feature_list = list() 
        
        for graph, label in dataset_dgl:
            nxG = dgl.to_networkx(graph)
            new_PyG_graph = pyg_utils.from_networkx(nxG)
            graph_attr_dgl = graph.ndata['attr']
            
            new_PyG_graph.x = graph_attr_dgl
            for node_features in new_PyG_graph.x:
                feature_list.append(node_features)
            new_PyG_graph.y = label
            modified_list.append(new_PyG_graph)
            # print(new_PyG_graph)
            
        custom_dataset = CustomDataset(modified_list, library= 'pyg')
        custom_dataset.y = torch.tensor([graph.y for graph in custom_dataset])
        custom_dataset.x = torch.stack(feature_list)
        custom_dataset.num_classes = len(set([item.item() for item in custom_dataset.y])) 
        custom_dataset.num_features = custom_dataset[0].x.shape[1] 
        
        return custom_dataset
    
    def to_dgl_only_from_pyg(self, dataset_pyg):
        
        """
        This method is used for converting PyG datasets to DGL datasets 
        """
        modified_list = list() 
        labels = list()
        
        for graph in dataset_pyg:
            nxG = pyg_utils.to_networkx(graph)
            new_DGL_graph = dgl.from_networkx(nxG)
            new_DGL_graph.ndata['attr'] = graph.x
            modified_list.append(new_DGL_graph)
            labels.append(graph.y)
        custom_dataset = CustomDataset(modified_list, labels = labels, library= 'dgl')
        custom_dataset.num_classes = len(set([label for graph, label in custom_dataset]))
        custom_dataset.num_features = custom_dataset[0][0].ndata['attr'].shape[1]
        
        return custom_dataset
    
 
    def to_One_HOT(self, dataset_pyg, dense = False, gl_max_node = 150, dataset_name = None):
        
        """
        This method is to get the graphs features of IMDB (BINARY and MULTI) and REDDIT-BINARY datasets
        In PyG library these social network datasets features are unavailable. Then we apply 1-hot vector 
        to attain graphs' feature. Here, The highest dimension of a graph is 150. If any graph dataset's
        feature pass over 150 then we do not use one hot vector for them. In case of REDDIT-BINARY datasets 
        its Memory consuming to use 1-hot vector ( that is over 400 dimension). In that context, we use simple 
        ones of 16 dimensions. That works well in each model. 
        """
        
        modified_list = list()
        feature_list = list() 
        
        overall_max_degree = 0 
        if not dense:
            for graph in dataset_pyg:
                # calculate the degree of each node 
                deg = degree(graph.edge_index[0], dtype = torch.int16)
                max_deg_of_graph = deg.max().item() 
                
                if max_deg_of_graph > overall_max_degree:
                    overall_max_degree = max_deg_of_graph
            
            print('Overall max degree:', overall_max_degree)
            
            if dataset_name == 'REDDIT-BINARY':
                pass
            else:
                transform = OneHotDegree(max_degree=overall_max_degree, cat=False)
             
            for graph in dataset_pyg:
                #print(graph)
                if dataset_name == "REDDIT-BINARY":
                    graph.x = torch.ones(graph.num_nodes, 16)
                else:
                    graph = transform(graph)
                modified_list.append(graph)
                
                for node_features in graph.x:
                    feature_list.append(node_features)
        else:
            graphs_max_node = 0 
            
            for graph in dataset_pyg:
                if graph.num_nodes > graphs_max_node:
                    graphs_max_node = graph.num_nodes
            
            for graph in dataset_pyg:
                ### gl_max_node -> global_max_node
                features = torch.eye(gl_max_node) # torch.ones(gl_max_node, gl_max_node)
                features = features[:graphs_max_node].float()
                graph.x = features.T 
                
                #print(graph, features)
                modified_list.append(graph)
                
                for node_features in graph.x:
                    feature_list.append(node_features)
    
        custom_dataset = CustomDataset(modified_list, library= 'pyg')
        custom_dataset.y = torch.tensor([graph.y for graph in custom_dataset])
        custom_dataset.x = torch.stack(feature_list)
        custom_dataset.num_classes = len(set([item.item() for item in custom_dataset.y])) 
        custom_dataset.num_features = custom_dataset[0].x.shape[1] 
        
        return custom_dataset

    def Operation(self):
        
        NCI109_DD_to_dgl = False 
        IR_with_one_hot = True 
        dense = False 
        ept = ''
        
        pyg_dataset_name = self.dataset_name 
        if self.dataset_name in ['IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY']:
            dgl_segment = self.dataset_name.split('-')
            dgl_dataset_name = dgl_segment[0] + dgl_segment[1]
        else:
            dgl_dataset_name = self.dataset_name 
        if not dense:
            if dgl_dataset_name == "PTC":
                dataset_dgl = GINDataset( dgl_dataset_name, self_loop= True, degree_as_nlabel= False)
                custom_dataset = self.to_pyg_only_from_dgl(dataset_dgl)
        # =============================================================================
        #         print("--------------------------- 1 Hot Encoding ---------------------------------")
        #         custom_dataset = to_One_HOT_COLLAB(custom_dataset)
        # =============================================================================
            elif pyg_dataset_name == "NCI109" or pyg_dataset_name == "DD":
                
                if NCI109_DD_to_dgl:
                    ept = 'dgl_'
                    dataset_pyg = TUDataset(root='data/TUDataset', name= pyg_dataset_name)
                    custom_dataset = self.to_dgl_only_from_pyg(dataset_pyg)
                else:
                    dataset_pyg = TUDataset(root='data/TUDataset', name= pyg_dataset_name)
                    print(len(dataset_pyg))
                    import time
                    # time.sleep(1)
                    custom_dataset = self.to_One_HOT(dataset_pyg) 
            else:
                if not IR_with_one_hot:
                    dataset_dgl = GINDataset( dgl_dataset_name, self_loop= True, degree_as_nlabel= False)
                    dataset_pyg = TUDataset(root='data/TUDataset', name= pyg_dataset_name)
                    custom_dataset = self.from_dgl_to_pyg_both(dataset_dgl, dataset_pyg)
                else:
                    dataset_pyg = TUDataset(root='data/TUDataset', name= pyg_dataset_name)
                    print(len(dataset_pyg))
                    import time
                    time.sleep(1)
                    
                    if pyg_dataset_name == "REDDIT-BINARY":
                        custom_dataset = self.to_One_HOT(dataset_pyg, dataset_name = pyg_dataset_name)
                    else:
                        custom_dataset = self.to_One_HOT(dataset_pyg)
                    
        else:
            max_nodes = 150
            ept = ept + 'dense_' 
            
            if dgl_dataset_name == 'PTC':
                
                # edge_index_list = list() 
                dataset_dgl = GINDataset( dgl_dataset_name, self_loop= True, degree_as_nlabel= False)
                custom_dataset = self.to_pyg_only_from_dgl(dataset_dgl)
                # transform = T.ToDense(max_nodes)
                # pre_filter=lambda data: data.num_nodes <= max_nodes 
                
                for graph in custom_dataset:
                    nxG = pyg_utils.to_networkx(graph)
                    adj_sparse = np.zeros([max_nodes, max_nodes])
                    nxG_sparse = nx.from_numpy_array(adj_sparse)
                    nxG = nx.compose(nxG, nxG_sparse)
                    sparse_adj_matrix = nx.adjacency_matrix(nxG)
                    dense_adj_maxtrix = torch.tensor(sparse_adj_matrix.toarray())
                    graph.adj = dense_adj_maxtrix.float() 
                    mask = torch.rand(max_nodes) > 0.5
                    graph.mask = mask
                    graph.edge_index = None
                    graph.id = None
                    graph.x = None
                    #graph = pre_filter(graph)
                    
                custom_dataset = self.to_One_HOT(custom_dataset, dense=dense)
    
            else:
                dataset_name = pyg_dataset_name 
                path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                                str(dataset_name) + '_dense')
        
                dataset_pyg = TUDataset(
                    path,
                    name= dataset_name,
                    transform=T.ToDense(max_nodes),
                    pre_filter=lambda data: data.num_nodes <= max_nodes,
                )
                custom_dataset = self.to_One_HOT(dataset_pyg, dense=dense)
                
    ### "MUTAG", "PTC", "NCI1", "PROTEINS", "IMDBBINARY", "IMDBMULTI", "REDDITBINARY", "COLLAB"
        for item in custom_dataset:
            print(item)
       
        ept = ept + 'Original'
    
        def save_custom_dataset(dataset, file_name):
            torch.save(dataset, file_name)
            
        def load_custom_dataset(file_name):
            dataset = torch.load(file_name)
            return dataset 
        
        def get_dataset_name(name):
            # path = os.getcwd()
            dataset_name = 'custom_'+ name + '_' + str(ept) +'_.pt'
            return dataset_name
        
        if dgl_dataset_name == "PTC":
            save_custom_dataset(custom_dataset, get_dataset_name(dgl_dataset_name))
        else:
            save_custom_dataset(custom_dataset, get_dataset_name(pyg_dataset_name))
        print('END of Code')
        
        return custom_dataset


if __name__ == "__main__":
    dataset_name = 'IMDB-BINARY'
    dataset_Operation = Dataset_Operation(dataset_name=dataset_name)
    custom_dataset = dataset_Operation.Operation()
