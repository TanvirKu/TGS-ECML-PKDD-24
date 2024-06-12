# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:11:39 2024

@author: thsou
"""

import random
global avg_truss_calc_time, truss_change_operation_time
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, graphs, labels = None, library = 'pyg'):
        self.graphs = graphs
        self.labels = labels
        self.library = library

        if self.library.lower() == 'dgl':
            # Assuming all graphs have the same feature dimensionality
            # and that node features are stored in ndata['feat']
            if len(graphs) > 0 and 'attr' in graphs[0].ndata:
                self.dim_nfeats = graphs[0].ndata['attr'].shape[1]
            else:
                self.dim_nfeats = 0  # Or appropriate default value
    
            # Calculate the number of unique graph categories
            label_list = set([label.item() for label in self.labels])
            self.gclasses = len(label_list)

        elif self.library.lower() == 'pyg':
            if len(graphs) > 0:
                self.dim_nfeats = graphs[0].x.shape[1]
            else:
                self.dim_nfeats = 0

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        if self.labels != None:
            return self.graphs[idx], self.labels[idx]
        return self.graphs[idx]
    
    def shuffle(self, dataset):
    
        # Shuffling graphs and labels together to maintain correspondence
        shuffled_graphs = list(dataset)
        random.shuffle(shuffled_graphs)
        custom_dataset = CustomDataset(shuffled_graphs, None, self.library)

        custom_dataset.num_classes = len(set([item.item() for item in custom_dataset.y])) 
        custom_dataset.num_features = custom_dataset[0].x.shape[1] 
        return custom_dataset
    