import os
import pandas as pd
import json
import numpy as np

##Dataset for filled dataset##
def stance_labeling(df):
    df.columns = ['c1','c2']
    df = df.groupby('c1').agg({'c1':'size','c2':'mean'})
    df.columns = ['col1','col2']
    df = df.reset_index(drop = False)
    df.columns = ['id','count','stance_val']
    df = df[df['id'] != '[]']
    df = df.sort_values(by=['count'], ascending = False)
    df = df.reset_index(drop = True)
    
    df.loc[(df['stance_val'] > 0.0) & (df['stance_val'] < 1), 'stance'] = 0
    df.loc[df['stance_val'] <= 0, 'stance'] = 1
    df.loc[df['stance_val'] >= 1, 'stance'] = 2
    return df
def str2list(l):
    res = l.strip('][').split(', ')
    l_tag = []
    for tag in res:
        l_tag.append(tag[1:-1])
    return l_tag

def creating_dataset(data_path:str):
    # clean_text, hashtag, text(hash+text), bigrams, trigrams, domain
    label = 'clean_text'
    file_path = 'data/'
    
    path = os.path.join(data_path, file_path)
    df_tweet = pd.read_csv(<File Path>)
    l = []
    for index, row in df_tweet.iterrows():
        l_tag = str2list(row.hashtags)
        l_bi = str2list(row.bigrams)
        l_tri = str2list(row.trigrams)
        
        item = {'tweet_id':row.tweet_id, 'username':row.username, 'original_user':row.original_user, 
                'clean_text': row.clean_text, 'tweet_stance':row.tweet_stance, 'user_stance':row.user_stance, 
                'original_user_stance':row.original_user_stance,'urls':row.urls, 'hashtags':l_tag, 
                'shortDate':row.shortDate, 'user_created':row.user_created, 'bigrams':l_bi,
                'trigrams':l_tri,'domain':row.host}
        l.append(item)
    df_tweet = pd.DataFrame(l)
    
    df_tweet['hash_text'] = [','.join(map(str, l)) for l in df_tweet['hashtags']]
    df_tweet['hash_text'] = df_tweet['hash_text'].str.replace('#','')
    df_tweet['text'] = df_tweet['clean_text'].astype(str) +' '+ df_tweet['hash_text']
    
    df_user = df_tweet[['username','user_stance']]
    df_user.columns = ['user','stance']
    df_original_user = df_tweet[['original_user','original_user_stance']]
    df_original_user.columns = ['user','stance']
    
    ### user ID
    df_user_id = pd.concat([df_user, df_original_user])
    df_user_id = stance_labeling(df_user_id)
    
    ### Domain ID
    #df_domain_id = df_tweet[['domain','tweet_stance']]
    #df_domain_id = stance_labeling(df_domain_id)
    df_domain = df_tweet[['domain']]
    df_domain = df_domain.groupby('domain', as_index = False).size()
    #df_domain = df_domain.reset_index(drop = False)
    df_domain.columns = ['id','count']
    df_domain = df_domain[df_domain['id'] != '']
    df_domain = df_domain.sort_values(by=['count'], ascending = False)
    df_domain = df_domain.reset_index(drop = True)
    
    ### Tweet Text
    df_text = df_tweet[['clean_text','tweet_stance']]
    df_text = stance_labeling(df_text)
    df_text = df_text.reset_index(drop = False)
    df_text = df_text[df_text['count'] > 1]
    
    ### Bigrams
    df_bigram = df_tweet.explode('bigrams')
    df_bigram = df_bigram.groupby('bigrams').size()
    df_bigram = df_bigram.reset_index(drop = False)
    df_bigram.columns = ['id','count']
    df_bigram = df_bigram[df_bigram['id'] != '']
    df_bigram = df_bigram.sort_values(by=['count'], ascending = False)
    df_bigram = df_bigram.reset_index(drop = True)
    
    ### Trigrams
    df_trigram = df_tweet.explode('trigrams')
    df_trigram = df_trigram.groupby('trigrams').size()
    df_trigram = df_trigram.reset_index(drop = False)
    df_trigram.columns = ['id','count']
    df_trigram = df_trigram[df_trigram['id'] != '']
    df_trigram = df_trigram.sort_values(by=['count'], ascending = False)
    df_trigram = df_trigram.reset_index(drop = True)
    
    ### Hashtag
    df_hashtag = df_tweet.explode('hashtags')
    df_hashtag = df_hashtag.groupby('hashtags').size()
    df_hashtag = df_hashtag.reset_index(drop = False)
    df_hashtag.columns = ['id','count']
    df_hashtag = df_hashtag[df_hashtag['id'] != '']
    df_hashtag = df_hashtag.dropna()
    df_hashtag = df_hashtag.sort_values(by=['count'], ascending = False)
    df_hashtag = df_hashtag.reset_index(drop = True)
    
    ### Tweet Hashtag + Text
    df_hash_text = df_tweet[['text','tweet_stance']]
    df_hash_text = stance_labeling(df_hash_text)
    df_hash_text = df_hash_text.reset_index(drop = False)
    print('hash_text : ',df_hash_text.shape)
    df_hash_text = df_hash_text[df_hash_text['count'] > 1]
    print(df_hash_text.shape)
    
    df_user_id.loc[df_user_id['stance'] == 0, 'stance'] = -1
    df_user_id['stance'] = df_user_id['stance'].astype(int)
    
    user_id = df_user_id[['id','stance']]
    user_id = user_id.reset_index(drop = False)
    user_id.columns = ['index','users','code']
    user_id.to_csv(f'{path}/user_id.csv', index = False)
    print('user_id : ', user_id.shape)
    
    print('hashtag : ', df_hashtag.shape)
    hashtag = df_hashtag[df_hashtag['count'] > 10]
    print(hashtag.shape)
    hashtag = hashtag[['id']]
    hashtag = hashtag.reset_index(drop = False)
    hashtag.columns = ['index','id']
    
    print('bigram : ', df_bigram.shape)
    bigram = df_bigram[df_bigram['count'] > 10]
    print(bigram.shape)
    bigram = bigram[['id']]
    bigram = bigram.reset_index(drop = False)
    bigram.columns = ['index','id']
    
    print('trigram :', df_trigram.shape)
    trigram = df_trigram[df_trigram['count'] > 1]
    print(trigram.shape)
    trigram = trigram[['id']]
    trigram = trigram.reset_index(drop = False)
    trigram.columns = ['index','id']
    
    print('domain : ', df_domain.shape)
    domain = df_domain[['id']]
    domain = domain.reset_index(drop = False)
    domain.columns = ['index' ,'id']
    
    if label == 'clean_text':
        dict_item = pd.Series(df_text['index'].values, index = df_text['id']).to_dict()
    elif label == 'hashtags':
        dict_item = pd.Series(hashtag['index'].values, index = hashtag['id']).to_dict()
    elif label == 'text':
        dict_item = pd.Series(df_hash_text['index'].values, index = df_hash_text['id']).to_dict()
    elif label == 'bigrams':
        dict_item = pd.Series(bigram['index'].values, index = bigram['id']).to_dict()
    elif label == 'trigrams':
        dict_item = pd.Series(trigram['index'].values, index = trigram['id']).to_dict()
    elif label == 'domain':
        dict_item = pd.Series(domain['index'].values, index = domain['id']).to_dict()
  
    dict_user = pd.Series(user_id['index'].values, index = user_id['users']).to_dict()
    
    ## Graph file
    df_node = user_id[['index','users','code']]
    
    df_known_graph = df_tweet[(df_tweet['username'].isin(user_id['users'])) & (df_tweet['original_user'].isin(user_id['users']))]
    
    df_graph = df_known_graph[['username','original_user']]
    df_graph = df_graph.replace({'username':dict_user, 'original_user':dict_user})
    
    gb_graph = df_graph.groupby(['username','original_user']).size()
    gb_graph = gb_graph.reset_index(drop = False)
    gb_graph.columns = ['from','to','count']
    gb_graph = gb_graph.sort_values(by = 'count', ascending = False)
    gb_graph = gb_graph.dropna()
    gb_graph = gb_graph.astype(int)
    
    gb_graph.to_csv(f'{path}/graph_user.csv', index = False)
    
    ## feature matrix
    df_item = df_tweet[['tweet_id','username','original_user',label]]
    df_item = df_item.dropna()
    
    l_target = []
    
    for index, row in df_node.iterrows():
        df_summary1 = df_item[df_item['username'] == row['users']]
        df_summary2 = df_item[df_item['original_user'] == row['users']]
        df_summary = pd.concat([df_summary1, df_summary2])
        
        item_list = []
        if df_summary.shape[0] > 0:
            item_list = df_summary[label].apply(pd.Series).stack().unique()
        item = {'id':row['index'], 'users':row['users'], 'item_list': item_list}
        l_target.append(item)
    df_target = pd.DataFrame(l_target)
    
    ## Target.csv
    df_target_code = df_node[['index','users','code']]
    df_target_code = df_target_code.fillna(-1)
    df_target_code.to_csv(f'{path}/target.csv', index = False)
    
    ## Features
    l_code = []
    for index, row in df_target.iterrows():
        #print(row)
        l = row['item_list']
        l_code_list = []
        
        if len(l) > 0:
            l_code_list = list((pd.Series(l)).map(dict_item))
            l_code_list = [i for i in l_code_list if str(i) != 'nan']
            #l_code_list2 = [i for i in l_code_list if i < NUM_data]
        #print(l_code_list)
        item = {'id':row['id'], 'users':row['users'],'item_list':row['item_list'], 'item_code':l_code_list}
        l_code.append(item)
    df_code = pd.DataFrame(l_code)
    dict_features = pd.Series(df_code['item_code'].values, index = df_code['id']).to_dict()
    with open(f'{path}/feature.json', 'w') as fp:
        json.dump(dict_features, fp)
    print(dict_features)
    
    if label == 'clean_text':
        return user_id.shape[0], df_text.shape[0]
    elif label == 'hashtags':
        return user_id.shape[0], hashtag.shape[0]
    elif label == 'text':
        return user_id.shape[0], df_hash_text.shape[0]
    elif label == 'bigrams':
        return user_id.shape[0], bigram.shape[0]
    elif label == 'trigrams':
        return user_id.shape[0], trigram.shape[0]
    elif label == 'domain':
        return user_id.shape[0], domain.shape[0]
