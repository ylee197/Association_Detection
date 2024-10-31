import os
import requests
import types
import json
import csv
import pickle
import pandas as pd

import numpy as np
from sklearn.preprocessing import label_binarize
import scipy.io

import torch
from torch_geometric.data import Data
#from torch_geometric.datasets import Planetoid, Amazon, Coauthor, DeezerEurope, Actor, MixHopSyntheticDataset
import torch_geometric.transforms as transforms
from torch_geometric.utils import to_undirected, add_remaining_self_loops
#from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

#from data_utils import keep_only_largest_connected_component
NUM_data = 400
def get_dataset(n_user, n_domain, data_path, label):
    DATA_PATH = data_path
    FILE_PATH = f'{DATA_PATH}/data/predicted_user_id.csv'
    print(DATA_PATH)
    
    if not os.path.exists(f'{DATA_PATH}'):
        print(" The file path does not exist. Please double checking the file path.\n")
    #elif os.path.exsts(FILE_PATH):
    elif label == 'domain':
        dataset = load_dataset(DATA_PATH,FILE_PATH, n_user, n_domain) 
    else:
        dataset = load_dataset(DATA_PATH, None, n_user, n_domain)
        
    # Make graph undirected so that we have edges for both directions and add self loops
    dataset.data.edge_index = to_undirected(dataset.data.edge_index)
    dataset.data.edge_index, _=add_remaining_self_loops(dataset.data.edge_index, num_nodes=dataset.data.x.shape[0])
    
    return dataset

def load_dataset(DATA_PATH, FILE_PATH, n_user, n_domain):
    if FILE_PATH is None:
        A, label, features, missing = load_data(DATA_PATH, n_user, n_domain)
    else:
        A, label, features, missing = load_data_domain(DATA_PATH, FILE_PATH, n_user, n_domain)
    
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features, dtype=torch.float)
    
    data = Data(
        x=node_feat,
        edge_index=edge_index,
        y=torch.tensor(label),
        missing=torch.tensor(missing),
     )
    dataset = types.SimpleNamespace()
    dataset.data = data
    dataset.num_classes = data.y.max().item() + 1
    
    return dataset

def load_data(DATA_PATH, n_user, n_domain):
    n_domain = n_domain + 1
    label = []
    node_ids = []
    src = []
    targ = []
    uniq_ids = set()
    
    with open(f'{DATA_PATH}/data/target.csv', 'r') as f: # labeled user. TARGET
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            node_id = int(row[0]) #labeling domain
            if node_id not in uniq_ids:
                uniq_ids.add(node_id)
                label.append(int(float(row[2]))) # user codes
                node_ids.append(int(row[0]))
                
    node_ids = np.array(node_ids, dtype=np.int)
   
    with open(f'{DATA_PATH}/data/graph_user.csv', 'r') as f: # GRAPH
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            src.append(int(row[0]))
            targ.append(int(row[1]))
    
    #with open(f'{DATA_PATH}/Ukraine/known_feature_user.json', 'r') as f: # features
    with open(f'{DATA_PATH}/data/feature.json', 'r') as f: # features
        j = json.load(f)
        
    src = np.array(src)
    targ = np.array(targ)
    label = np.array(label)
    
    n = label.shape[0]
    print(len(src))
    print(len(targ))
    print(n)

    A = scipy.sparse.csr_matrix((np.ones(len(src)), (np.array(src), np.array(targ))), shape = (n, n)) # Node-Node graph
    
    features = np.zeros((n,(n_domain))) # domain
    for node, feats in j.items():
        if int(node) >= n:
            continue
        if len(feats) == 0:
            features[int(node), :] = np.full((n_domain), -1, dtype=int)
        else:
            features[int(node), np.array(feats, dtype=int)] = 1
    
    missing = features
    missing = features != -1
    
    print('missing rate: ' +  str(np.count_nonzero(missing == False)/(n*n_domain)))
    print("feature -1 rate : " + str(np.count_nonzero(features == -1)/(n*n_domain)))

    return A, label, features, missing

def load_data_domain(DATA_PATH, FILE_PATH, n_user, n_domain):
    #n_domain = n_domain + 1
    label = []
    node_ids = []
    src = []
    targ = []
    uniq_ids = set()
    
    with open(f'{DATA_PATH}/data/original_domain_id.csv', 'r') as f: # labeled user. TARGET
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            node_id = int(row[0]) #labeling domain
            if node_id < (n_domain + 1):
                if node_id not in uniq_ids:
                    uniq_ids.add(node_id)
                    label.append(int(float(row[3]))) # domain codes
                    node_ids.append(int(row[0]))
    ## predicted user label
    df_user = pd.read_csv(f'{FILE_PATH}')
    df_user.columns = ['node_index','users','predict']
    node_ids = np.array(node_ids, dtype=np.int)
   
    with open(f'{DATA_PATH}/data/feature.json', 'r') as f: # features
        j = json.load(f)
        
    label = np.array(label)
    
    n = label.shape[0]
    
    features = np.zeros((n,(n_domain))) # domain
    for node, feats in j.items():
        if int(node) >= n:
            continue
        features[int(node), np.array(feats, dtype=int)] = 1
        
    ## domain connection graph
    M = pd.DataFrame(features)
    
    df_user_index = pd.DataFrame(np.random.randint(0, n_user, size = (n_user, 1)))
    df_user_index = df_user_index.reset_index(drop = False)
    df_user_index = df_user_index.drop(df_user_index.columns[1], axis=1)
    
    df_merge = df_user_index.merge(df_user, left_on = 'index', right_on = 'node_index', how = 'left')
    df_merge = df_merge.drop(columns = ['node_index'])
    df_merge = df_merge.fillna(0)
    
    df_D_edge = domain_graph(M, df_merge, n_domain)
    
    #print(df_D_edge)
    
    src = np.array(df_D_edge['from'].tolist())
    targ = np.array(df_D_edge['to'].tolist())
    
    M_T = M.T
    #M_T = M_T.head(800)
    
    M_T['sum'] = M_T.sum(axis=1)
    for index, row in M_T.iterrows():
        if row['sum'] == 0:
            M_T.loc[index,:] = -1
    M_T = M_T.drop(columns = ['sum'])
    
    domain_features = M_T.to_numpy()
    
    print(domain_features.shape)
    A = scipy.sparse.csr_matrix((np.ones(len(src)), (np.array(src), np.array(targ))), shape = (n, n)) # domain-domain graph
   
    print(len(src))
    print(len(targ))
    print(n)
    
    missing = domain_features
    missing = domain_features != -1
    
    print('missing rate: ' +  str(np.count_nonzero(missing == False)/(n*n_user)))
    print("feature -1 rate : " + str(np.count_nonzero(domain_features == -1)/(n*n_user)))
   
    return A, label, domain_features, missing

def edge_weight(M, df, label, n_domain):
    
    n = df[df['predict'] == label].shape[0]
  
    l_filter = [True if x == label else False for x in df['predict'].tolist()]
    F = [l_filter] * (n_domain +1)
    filter_M = pd.DataFrame(F)
   
    M_masked = M.mask(filter_M == False, 0)
    M_masked_T = M_masked.T
    
    D_edge = M_masked.dot(M_masked_T)
    for index, row in D_edge.iterrows():
        D_edge.loc[index, index] = 0
    D_edge = D_edge/n
    return D_edge


def domain_graph(M, df, n_domain):
    # df.collumns : node_index, predict
    # 1 - Ukraine, 2 - Russia
    
    M_T = M.T
    D_edge1 = edge_weight(M_T, df, 1, n_domain)
    D_edge2 = edge_weight(M_T, df, 2, n_domain)
    D_edge = D_edge1.copy()
    
    for i in range(D_edge.shape[1]):
        D_edge.loc[:,i] = np.where(D_edge1.loc[:,i] > D_edge2.loc[:,i], D_edge1.loc[:,i], D_edge2.loc[:,i])
    
    D_edge_weight = M_T.dot(M)
    for index, row in D_edge_weight.iterrows():
        D_edge_weight.loc[index, index] = 0
    
    for i in range(D_edge_weight.shape[1]):
        D_edge_weight.loc[:,i] = np.where(D_edge_weight.loc[:,i] > 0 , D_edge.loc[:,i], 0)
    avg = D_edge_weight[D_edge_weight > 0].stack().mean(axis = 0)
    std = D_edge_weight[D_edge_weight > 0].stack().std(axis = 0)
    print("AVG : {:.5f}".format(avg))
    print("Standard dividation : {:.5f}".format(std))

    D_edge_weight = D_edge_weight[D_edge_weight > avg].stack()
    
    D_edge_weight = D_edge_weight.reset_index(drop = False)
    D_edge_weight.columns = ['from','to','weight']
    return D_edge_weight


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    