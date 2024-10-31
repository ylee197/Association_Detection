import math

import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_networkx
from torch_geometric.utils.convert import from_networkx
from torch_scatter import scatter_add

def set_uniform_train_val_test_split(
    seed: int,
    data: Data) -> Data:
    train_ratio = 0.70
    #train_ratio = 1
    val_ratio = 0.29
    #val_ratio = 0.01
    
    #rnd_state = np.random.RandomState(seed)
    num_nodes = data.y.shape[0]
    
    # without -1
    #labeled_nodes = torch.where(data.y != -1)[0]
    #unlabeled_nodes = torch.where(data.y == -1)[0]
    '''
    num_labeled_nodes = labeled_nodes.shape[0]
    #num_unlabeled_nodes = unlabeled_nodes.shape[0]
    num_train = math.floor(num_labeled_nodes * train_ratio)
    num_val = math.floor(num_labeled_nodes * val_ratio)
    #num_train = 200
    
    idxs = list(range(num_labeled_nodes))
    #un_idxs = list(range(num_unlabeled_nodes))
    rnd_state.shuffle(idxs)
    
    #train_idx = labeled_nodes[idxs]
    train_idx = idxs[:num_train]
    val_idx = idxs[num_train:num_train+num_val]
    test_idx1 = idxs[num_train:]
    #test_idx2 = unlabeled_nodes[un_idxs]
    
    train_idx = labeled_nodes[train_idx]
    val_idx = labeled_nodes[val_idx]
    #test_idx = labeled_nodes[test_idx]
    test_idx = labeled_nodes[test_idx1]
    #test_idx2 = unlabeled_nodes[un_idxs]
    
    #test_idx = torch.cat((test_idx1,test_idx2),0)
    '''
    '''
    ## Including -1 data
    labeled_nodes = torch.where(data.y != -1)[0]
    unlabeled_nodes = torch.where(data.y == -1)[0]
    
    #labeled_nodes = torch.where(labeled_nodes <= 400)[0]
    #unlabeled_nodes = torch.where(unlabeled_nodes <= 400)[0]
    
    num_labeled_nodes = labeled_nodes.shape[0]
    num_unlabeled_nodes = unlabeled_nodes.shape[0]
    num_train = math.floor(num_labeled_nodes * train_ratio)
    num_val = math.floor(num_labeled_nodes * val_ratio)
    #num_train = 400
    #num_val = 300
    idxs = list(range(num_labeled_nodes))
    un_idxs = list(range(num_unlabeled_nodes))
    #rnd_state.shuffle(idxs)
    
    #train_idx = labeled_nodes[idxs]
    train_idx = idxs[:num_train]
    val_idx = idxs[num_train:num_train+num_val]
    test_idx1 = idxs[num_train:]
    #test_idx2 = un_idxs[:400]
    
    train_idx = labeled_nodes[train_idx]
    val_idx = labeled_nodes[val_idx]
    #test_idx = labeled_nodes[test_idx]
    test_idx1 = labeled_nodes[test_idx1]
    test_idx2 = unlabeled_nodes[un_idxs]
    
    test_idx = torch.cat((test_idx1,test_idx2),0)
    #unlabeled_nodes = torch.where(data.y == -1)[0]
    #num_unlabeled_nodes = unlabeled_nodes.shape[0]
    
    #un_idxs = list(range(num_unlabeled_nodes))
    #test_un_idx = unlabeled_nodes[un_idxs]
    '''
    
    train_idx1, test_idx1, val_idx1 = stratified( data.y, 1, train_ratio, val_ratio, seed)
    train_idx2, test_idx2, val_idx2 = stratified( data.y, 2, train_ratio, val_ratio, seed)
    
    
    unlabeled_nodes = torch.where(data.y == -1)[0]
    #num_unlabeled_nodes = unlabeled_nodes.shape[0]
    
    #num_train = math.floor(num_labeled_nodes * train_ratio)
    #num_val = math.floor(num_labeled_nodes * val_ratio)
    #print('num of training set : ', num_train)
    #print('num of validation set : ', num_val)
    
    #idxs = list(range(num_labeled_nodes))
    
    #rnd_state.shuffle(idxs)
    
    train_idx = torch.cat((train_idx1,train_idx2),0)
    val_idx = torch.cat((val_idx1,val_idx2),0)
    test_idx = torch.cat((test_idx1,test_idx2,unlabeled_nodes),0)
    
    '''
    test_idx2 = torch.cat((test2_idx1,test2_idx2),0)
    print('num of training set : ', train_idx.shape[0])
    #print('num of validation set : ', val_idx.shape[0])
    print('num of test set : ', test_idx2.shape[0])
    '''
    data.train_mask = get_mask(train_idx, num_nodes)
    data.val_mask = get_mask(val_idx, num_nodes)
    #data.test_mask = get_mask(test_un_idx, num_nodes)
    data.test_mask = get_mask(test_idx, num_nodes)
    #data.test_idx = test_un_idx
    data.test_idx = test_idx
    print('num of training set : ', train_idx.shape[0])
    print('num of validation set : ', val_idx.shape[0])
    #print(train_idx)
    #print('num of test set1 : ', test_idx1.shape[0])
    print('num of test set : ', test_idx.shape[0])
    return data

def stratified(y, label, train_ratio, val_ratio, seed):
    rnd_state = np.random.RandomState(seed)
    labeled_nodes = torch.where(y == label)[0]
    num_labeled_nodes = labeled_nodes.shape[0]
    
    num_train = math.floor(num_labeled_nodes * train_ratio)
    num_val = math.floor(num_labeled_nodes * val_ratio)
    
    idxs = list(range(num_labeled_nodes))
    rnd_state.shuffle(idxs)
    
    train_idx = idxs[:num_train]
    val_idx = idxs[num_train:num_train+num_val]
    test_idx = idxs[num_train:]
    train_idx = labeled_nodes[train_idx]
    val_idx = labeled_nodes[val_idx]
    test_idx = labeled_nodes[test_idx]
    
    return train_idx, test_idx, val_idx


def get_mask(idx, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask

def get_symmetrically_normalized_adjacency(edge_index, num_nodes):
    edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return DAD
'''
def row_normalize(edge_index, edge_weight, n_nodes):
    row_sum = get_adj_row_sum(edge_index, edge_weight, n_nodes)
    row_idx = edge_index[0]
    return edge_weight / row_sum[row_idx]
    
def get_adj_row_sum(edge_index, edge_weight, n_nodes):
    """
    Get weighted out degree for nodes. This is equivalent to computing the sum of the rows of the weighted adjacency matrix.
    """
    row = edge_index[0]
    return scatter_add(edge_weight, row, dim=0, dim_size=n_nodes)
'''