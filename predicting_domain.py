import torch
from torch import Tensor
from torch_scatter import scatter_add
import pandas as pd
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

import numpy as np
import matplotlib.pyplot as plt
import sklearn
NUM_data = 800
'''
def predicting_users(F, Y, data_path):
    file_path = 'data/'
    
    path = os.path.join(data_path, file_path)
    #print(F_T)
    user_id = pd.read_csv(f'{path}/user_id.csv')
    #user_index = pd.read_csv(f'{path}/user_id.csv')
    #user_id = pd.read_csv(f'{path}input/known_users.csv')
    #user_id = user_id[['users','code']]
    #domain_id = pd.read_csv(f'{path}input/known_domain.csv')
    
    gb_Y = Y.groupby(['node_index']).agg(lambda x:x.value_counts().index[0])
    
    gb_Y = gb_Y.reset_index(drop = False)
    gb_Y.columns = ['node_index','predict']
   
    #user_id = user_index[['index','count','users']].merge(user_id, left_on = 'users', right_on = 'users', how = 'left')
    #user_id['code'] = user_id['code'].fillna(-1)
    #user400 = user_id.head(1600)
    
    user_id['code'] = user_id['code'].fillna(-1)
    #user_id.loc[user_id['index'] > 1600, 'code']  = -1
    user400 = user_id.copy()
    merge = user400.merge(gb_Y, left_on = 'index', right_on = 'node_index', how = 'left')
    print(merge.head(6400))
    print(merge)
    #merge[['node_index','predict','users']].to_csv(f'{path}/predicted_user_id.csv', index = False)
    merge_cal = merge.dropna(subset = ['predict'])
    merge_cal = merge_cal[merge_cal['code'] != -1]
    merge_cal = merge_cal.dropna(subset = ['code'])
    #print('predict user code accuracy : ', merge_cal[merge_cal['predict'] == merge_cal['code']].shape[0]/merge_cal.shape[0])
    #known_user = gb_Y.loc[0,'node_index']
    #l_known_user = gb_Y['node_index'].tolist()
    
    for index, row in merge.iterrows():
        if row['code'] == -1:
        #if pd.isna(row['predict']) == False:
            merge.loc[index,'code'] = row['predict']
    merge = merge[['index','users','code']]
    #user400.loc[gb_Y['node_index'].tolist(), 'code'] = gb_Y['predict'].tolist()
    merge.to_csv(f'{path}/predicted_user_id.csv', index = False)
    #print(gb_Y)
    print(merge[merge['code'] == 1].shape)
    print(merge[merge['code'] == 2].shape)
    user400 = merge.copy()
    
    return user400
'''

def predicting_users(F, Y, data_path):
    file_path = 'data/'
    
    path = os.path.join(data_path, file_path)
    #print(F_T)
    user_index = pd.read_csv(f'{path}/user_id.csv')
    user_id = user_index.copy()
    #domain_id = pd.read_csv(f'{path}/original_domain_id.csv')
    #user_id = pd.read_csv(f'{path}input/known_users.csv')
    #user_id = pd.read_csv(f'{path}code_by_cluster.csv')
    user_id = user_id[['users','code']]
    #domain_id = pd.read_csv(f'{path}input/known_domain.csv')
    #F_T_weight = torch.ones((F_T.size(1), ), device=F_T.device)
    #row, col = F_T.shape[0], F_T.shape[1]
    #F_T_sum = F_T.sum(dim = 0, keepdim = False)
    #print(F_T_sum)
    
    gb_Y = Y.groupby(['node_index']).agg(lambda x:x.value_counts().index[0])
    
    gb_Y = gb_Y.reset_index(drop = False)
    gb_Y.columns = ['node_index','predict']
   
    #user_id = user_index[['index','count','users']].merge(user_id, left_on = 'users', right_on = 'users', how = 'left')
    #user_id['code'] = user_id['code'].fillna(-1)
    #user400 = user_id.head(NUM_data)
    
    #user_id['code'] = user_id['code'].fillna(-1)
    user400 = user_id.copy()
    merge = user400.merge(gb_Y, left_on = 'index', right_on = 'node_index', how = 'left')

    #merge[['node_index','predict','users']].to_csv(f'{path}/predicted_user_id.csv', index = False)
    merge_cal = merge.dropna(subset = ['predict'])
    merge_cal = merge_cal[(merge_cal['index'] >= (NUM_data/2))&(merge_cal['index'] < NUM_data)]
    merge_cal = merge_cal[merge_cal['code'] != -1]
    merge_cal = merge_cal.dropna(subset = ['code'])
    print('predict user code accuracy : ', merge_cal[merge_cal['predict'] == merge_cal['code']].shape[0]/merge_cal.shape[0])
    #known_user = gb_Y.loc[0,'node_index']
    #l_known_user = gb_Y['node_index'].tolist()
    
    for index, row in merge.iterrows():
        if row['code'] == -1:
        #if pd.isna(row['predict']) == False:
            merge.loc[index,'code'] = row['predict']
    merge = merge[['index','users','code']]
    #user400.loc[gb_Y['node_index'].tolist(), 'code'] = gb_Y['predict'].tolist()
    merge.to_csv(f'{path}/predicted_user_id.csv', index = False)
    #print(gb_Y)
    print(merge[merge['code'] == 1].shape)
    print(merge[merge['code'] == 2].shape)
    user400 = merge.copy()
    
    return user400

def predicting_domain(F, Y, data_path):
    file_path = 'data/'
    
    path = os.path.join(data_path, file_path)
    #print(F_T)
    #user_id = pd.read_csv(f'{path}/user_id.csv')
    #domain_index = pd.read_csv(f'{path}/original_domain_id.csv')
    user_id = pd.read_csv(f'{path}input/known_users.csv')
    #domain_id = pd.read_csv(f'{path}input/known_domain.csv')
    #print(domain_id)
    #print(domain_id.drop_duplicates(subset = 'domain')
    #F_T_weight = torch.ones((F_T.size(1), ), device=F_T.device)
    #row, col = F_T.shape[0], F_T.shape[1]
    #F_T_sum = F_T.sum(dim = 0, keepdim = False)
    #print(F_T_sum)
    '''
    gb_Y = Y.groupby('node_index')['predict'].mean()
    gb_Y = gb_Y.reset_index(drop = False)
    gb_Y.columns = ['node_index','avg']
   
    gb_Y.loc[gb_Y['avg'] <= 1.5, 'code'] = 1
    gb_Y.loc[(gb_Y['avg'] > 1.5) & (gb_Y['avg'] <=2), 'code'] = 2
    '''
    gb_Y = Y.groupby(['node_index']).agg(lambda x:x.value_counts().index[0])
    gb_Y = gb_Y.reset_index(drop = False)
    gb_Y.columns = ['node_index','predict']

    domain_id = domain_index[['index','domain']].merge(domain_id, left_on = 'domain', right_on = 'domain', how = 'left')
    #domain400 = domain_id.head(NUM_data)
    domain400 = domain_id.copy()
    merge = domain400.merge(gb_Y, left_on = 'index', right_on = 'node_index', how = 'left')
    #merge[['node_index','predict','users']].to_csv(f'{path}/predicted_user_id.csv', index = False)
    merge_cal = merge.dropna(subset = ['predict'])
    merge_cal = merge_cal[(merge_cal['index'] >= (NUM_data/2))&(merge_cal['index'] < NUM_data)]
    merge_cal = merge_cal[merge_cal['code'] != -1]
    merge_cal = merge_cal.dropna(subset = ['code'])
    
    print('predict domain code accuracy : ', merge_cal[merge_cal['predict'] == merge_cal['code']].shape[0]/merge_cal.shape[0])
    #known_user = gb_Y.loc[0,'node_index']
    #l_known_user = gb_Y['node_index'].tolist()
    
    for index, row in merge.iterrows():
        if row['code'] == -1:
        #if pd.isna(row['predict']) == False:
            merge.loc[index,'code'] = row['predict']
    merge = merge[['index','domain','code']]
    #print(merge)
    #sys.exit()
    #user400.loc[gb_Y['node_index'].tolist(), 'code'] = gb_Y['predict'].tolist()
    merge.to_csv(f'{path}/predicted_domain_id_'+str(NUM_data)+'.csv', index = False)
    #print(gb_Y)
    #print(merge)
    print(merge[merge['code'] == 1].shape)
    print(merge[merge['code'] == 2].shape)
    domain400 = merge.copy()
    '''
     ## predicting domains
    df = pd.DataFrame(F.cpu().numpy())
    user400 = user_id.head(800)
    df_T = df.T
    domain_label = domain400['code']
    df_data = df_T.copy()
    df_data['code'] = user400['code']
    #print(df_data)
   
    df_data = df_data[df_data['code'] > 0]
    df_data = df_data.reset_index(drop = True)
    
    X = df_data.copy()
    X = X.drop(columns = 'code')
    Y = df_data['code']
    
    df_T = df.T.astype(int)
    df_sum = df_T.sum(axis = 1)
    df_sum = df_sum.reset_index(drop = False)
    df_sum.columns = ['index','sum']
   
    domain_label = domain400['code']
    df_user_label = df_T.dot(domain_label)
    df_user_label = df_user_label.reset_index(drop = False)
    df_user_label.columns = ['index','sum']
    df_user_label['d'] = df_sum['sum']
    df_user_label['label'] = df_user_label['sum']/df_user_label['d']
    
    df_user_label.loc[df_user_label['label'] < 1.5, 'code'] = 1
    df_user_label.loc[df_user_label['label'] >= 1.5, 'code'] = 2
    
    user400 = user_id.head(400)
    df_user_label['index'] = df_user_label['index'].astype('int')
    df_user_test = df_user_label.merge(user400, left_on = 'index', right_on = 'index', how = 'left')
    
    ## Filtering out unknown domains
    df_user_test = df_user_test[df_user_test['code_y'] > 0]
    df_user_test = df_user_test.dropna()
    print(df_user_test)
    print('predict user code accuracy : ', df_user_test[df_user_test['code_x']==df_user_test['code_y']].shape[0]/df_user_test.shape[0])
    '''
    return merge