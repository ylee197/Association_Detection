import pandas as pd
import numpy as np
import os
from tqdm import tqdm

#df = pd.read_pickle('/home/ylee197/YJ/bot_detection/2022/0520/output/0520tweet_hashtag.pkl')
#df = df.rename(columns = {'expanded_url':'original_url','tid':'tweet_id','retweeted_id':'parent_tweet_id'})
df = pd.read_pickle('/home/ylee197/YJ/bot_detection/2022/0220/output/Ru_war_retweet_0220.pkl')
df_bot = pd.read_csv('/home/ylee197/YJ/bot_detection/2022/0520/Bot/data/total_bot_list.csv')
'''
df1 = pd.read_pickle('/home/ylee197/YJ/bot_detection/2022/0220/output/Ru_war_retweet_0220.pkl')
df2 = pd.read_pickle('/home/ylee197/YJ/bot_detection/2022/0510_Uk/data/Ukraine_may_URL.pkl')
df2 = df2.rename(columns = {'screen_name':'username','unwound_url':'original_url','tid':'tweet_id','host':'domain'})
df3 = pd.read_pickle('/home/ylee197/YJ/bot_detection/2022/0520/output/0520tweet_hashtag.pkl')
df3 = df3.rename(columns = {'expanded_url':'original_url','tid':'tweet_id','retweeted_id':'parent_tweet_id'})

df = pd.concat([df1, df3])
df = df.explode('original_url')
df['domain'] = df['original_url'].str.replace('https://','')
df['domain'] = df['domain'].str.replace('http://','')
df['domain'] = df['domain'].str.replace('www.','')
df['domain'] = df['domain'].str.split('/').str[0]
df = pd.concat([df1, df])
df = df.drop_duplicates()

df = df[df['retweet_to'].notnull()]
'''
#df = df[df['timestamp'] > '2022-01-17 21:17:56']
#print(df)
#print(df['timestamp'].max())
#print(df['timestamp'].min())
#print(df.columns)
#df_color = pd.read_csv('data/predicted_user_id_800_.csv')
df_color = pd.read_csv('/home/ylee197/YJ/bot_detection/2022/RWC_0806/input/predicted_user_id.csv')
df_color.loc[df_color['code'] == 1, 'code'] = 'U'
df_color.loc[df_color['code'] == 2, 'code'] = 'R'
l_user = df_color['users'].unique()
l_bot = df_bot['user'].unique()

#df1 = df1.rename(columns = {'screen_name':'username'})
df_color.loc[((df_color['users'].isin(l_bot)) & (df_color['code'] == 'U')), 'code'] = 'U_B'
df_color.loc[((df_color['users'].isin(l_bot)) & (df_color['code'] == 'R')), 'code'] = 'R_B'

#df1 = df1[['username','retweet_to','timestamp']]
#df2 = df2[['username','retweet_to','timestamp']]
#df_shares = df1.append(df2)

df_shares = df[['username','retweet_to','timestamp']]
df_shares = df_shares.rename(columns = {'username':'ID', 'timestamp':'time', 'retweet_to':'retweet_to'})

df_ID = df_shares[['ID','retweet_to','time']]
df_ID = df_ID[df_ID['ID'].isin(l_user)]
df_ID = df_ID[df_ID['retweet_to'].isin(l_user)]
#df_ID = df_shares[['ID','domain','time']]
df_ID.columns = ['Source','Target','time']
#l_bot = df_ID['Source'].tolist()

## For node timestamp
df_ID = df_ID[~pd.isna(df_ID['Target'])]
#df_ID.to_csv('output/080521_Total_tweet.csv', index = False)

# Removing the data that Targets overlap with Sources.
#l_source = df_ID['Source'].unique().tolist()

#df_ID = df_ID[~df_ID['Target'].isin(l_source)]
'''
## Edge list (For Pr.Corman)
df_original_edge = df_ID.copy()
df_original_edge.columns = ['retweet_from','retweet_to','time']
df_original_edge = df_original_edge.drop(columns = 'time')
gb_original_edge = df_original_edge.groupby(['retweet_from','retweet_to']).size()
gb_original_edge = gb_original_edge.reset_index(drop = False)
gb_original_edge.columns = ['bot','amplifier','count']
gb_original_edge = gb_original_edge.sort_values(by = 'count', ascending = False)
gb_original_edge = gb_original_edge.reset_index(drop = True)
print(gb_original_edge)
gb_original_edge.to_csv('output/0220_Ru_Uk_retweet_total_connection.csv',index = False)

'''
df_target = df_ID.copy()
df_target = df_target.drop(columns = ['Source'])
df_target = df_target.drop_duplicates()

df_source = df_ID.copy()
df_source = df_source.drop(columns = ['Target'])

df_target = df_target.rename(columns = {'Target':'id'})
#df_target['type'] = 'target'
#print(len(l_source))
#print(df_target)
df_node = df_ID.copy()
df_node = df_node.drop(columns = ['Target'])
df_node = df_node.drop_duplicates()
df_node = df_node.rename(columns = {'Source':'id'})
#df_node['type'] = 'source'
df_node = df_node.append(df_target)

gb_node = df_node.groupby('id')
df_node['min_time'] = gb_node['time'].transform('min')
df_node['max_time'] = gb_node['time'].transform('max')
#df_node = df_node.merge(df_bot, left_on = 'id', right_on = 'id', how = 'left')
df_node = df_node.drop(columns = ['time'])
df_node = df_node.merge(df_color[['users','code']], left_on = 'id', right_on = 'users', how = 'left')
#df_node = df_node.rename(columns = {'code':'type'})

#df_node.loc[df_node['id'].isin(l_bot), 'type'] = 'bot'
#df_node.loc[df_node['type'].isnull(), 'type'] = 'user'
#df_node = df_node.rename(columns = {'fast/slow':'type'})
#df_node = df_node.replace({np.nan:None})
#df_node.loc[((df_node['type'].isnull())&(df_node['id'].isin(l_amp))), 'type'] = 'Amp'
#df_node.loc[df_node['type'].isnull(), 'type'] = 'Other'
#print(df_node)
df_node = df_node.drop_duplicates()
df_node = df_node.reset_index(drop = True)
df_node = df_node.reset_index(drop = False)
#df_node = df_node.rename(columns = {'level_0':'index'})
df_node = df_node.drop(columns = ['users'])
'''
df_hosttype = pd.DataFrame()
df_hosttype = df_host.copy()
df_hosttype = df_hosttype.drop(columns =['Source'])
df_hosttype = df_hosttype.rename(columns = {'Target':'id'})
df_hosttype['type'] = 'domain'
gb_hosttype = df_hosttype.groupby('id')
df_hosttype['min_time'] = gb_hosttype['time'].transform('min')
df_hosttype['max_time'] = gb_hosttype['time'].transform('max')
df_hosttype = df_hosttype.drop(columns = ['time'])
df_hosttype = df_hosttype.drop_duplicates()
print(df_hosttype)

## For node timestamp
df_ID = pd.DataFrame()
df_ID = df_host.copy()
df_ID = df_ID.drop(columns = ['Target'])
df_ID = df_ID.rename(columns = {'Source':'id'})
gb_ID = df_ID.groupby('id')
df_ID['min_time'] = gb_ID['time'].transform('min')
df_ID['max_time'] = gb_ID['time'].transform('max')
df_ID = df_ID.drop(columns = ['time'])
df_ID = df_ID.drop_duplicates()
df_ID = df_ID.rename(columns = {'id':'l_id'})
df_ID = df_ID.merge(df_IDtype, left_on = 'l_id', right_on = 'id')
df_ID = df_ID.drop(columns = ['l_id'])
print(df_ID)

df_node = df_ID.append(df_hosttype)
df_node = df_node.drop_duplicates()
df_node = df_node.reset_index(drop = True)
df_node = df_node.reset_index(drop = False)
print(df_node)
'''
df_node.columns = ['id','label','min_time','max_time','type']
print(df_node)
df_node.to_csv('data/0220_Ru_Uk_bot_node.csv', index = False)


df_node = df_node[['id','label','min_time','max_time']]
df_edge = df_ID.copy()
df_edge = df_edge.merge(df_node, left_on = 'Source', right_on = 'label')
print(df_edge.columns)
df_edge = df_edge.drop(columns = ['Source','label','min_time','max_time'])
df_edge = df_edge.rename(columns = {'id':'Source'})
df_edge = df_edge.merge(df_node, left_on = 'Target', right_on = 'label')
df_edge = df_edge.drop(columns = ['Target','label','min_time','max_time'])
df_edge = df_edge.rename(columns = {'id':'Target'})
df_edge['time2'] = df_edge['time']
df_edge = df_edge.drop(columns = ['time'])
df_edge = df_edge.rename(columns = {'time2':'time'})
print(df_edge)
print(df_edge.columns)

df_edge.to_csv('data/0220_Ru_Uk_bot_edge.csv', index = False)

