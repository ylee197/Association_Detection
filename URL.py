import pandas as pd
import numpy as np
#import pycurl
#import urllib.request as urllib2
import urllib
import re
import requests
import socket
import http.client as httplib
import time

from tqdm import tqdm
from multiprocessing import Pool
import math

class URL:
    def __init__(self, df):
        self.df = df
        self.df = self.get_url(self.df)
        
    def intervals(self, group_size, num):
        part_duration = num / group_size
        return [(math.floor(i * part_duration), math.floor((i + 1) * part_duration)) for i in range(group_size)]

    def work(self, partition_range):
        START = partition_range[0]
        END = partition_range[1]
        df_split = self.df.iloc[START:END, :]
        data_list = []
        id_count = df_split.shape[0]
        i = 0

        with tqdm(total = df_split.shape[0]) as pbar:
            l_fullURL = []
            for index, row in df_split.iterrows():
                urls = row.urls[1:-1]
                urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', row.urls)
                tweet_id = row.tweet_id
                username = row.user
                original_user = row.original_user
                clean_text = row.clean_text
                tweet_stance = row.tweet_stance
                user_stance = row.user_stance
                original_user_stance = row.original_user_stance
                #urls = urls
                hashtags = row.hashtags
                shortDate = row.shortDate
                user_created = row.user_created
                bigrams = row.bigrams
                trigrams = row.trigrams
                l = []
                #i = 0
                for url in urls:
                    if len(url) > 0:
                        url = url[:-2]
                        try:
                            fp = urllib.request.urlopen(url, timeout = 10)
                            #fp.add_header('Referer', 'http://asu.edu')
                            actual_url = fp.geturl()
                            #print(url)
                            #print(actual_url)
                            l.append(actual_url)
                            '''
                            conn = pycurl.Curl()
                            conn.setopt(pycurl.URL, url)
                            conn.setopt(pycurl.FOLLOWLOCATION,1)
                            conn.setopt(pycurl.CUSTOMREQUEST, 'HEAD')
                            conn.setopt(pycurl.NOBODY, True)
                            conn.perform()
                            actual_url = conn.getinfo(pycurl.EFFECTIVE_URL)
                            '''
                            #l.append(actual_url)
                            #print(conn.getinfo(pycurl.EFFECTIVE_URL))

                        except ConnectionResetError:
                            print('ConectionResetError')
                            time.sleep(30)
                            continue
                        except urllib.error.HTTPError as e:
                            print (e.code)
                            continue
                        except urllib.error.URLError as e:
                            print ('URL_Error')
                            continue
                        except socket.timeout as e:
                            print('Connection timeout')
                            continue
                        except httplib.HTTPException as e:
                            print("HTTPException")
                            continue
                        else:
                            continue
                            #print('Access successful.')
                       #if (i % 20000 == 0):
                        #    time.sleep(60)
                       # i += 1
                item = {'tweet_id': tweet_id, 'username': username, 'original_user':original_user, 'clean_text':clean_text, 
                        'tweet_stance':tweet_stance, 'user_stance':user_stance, 'original_user_stance':original_user_stance, 
                        'urls':l,'hashtags':hashtags, 'shortDate':shortDate, 'user_created':user_created, 
                        'bigrams':bigrams, 'trigrams':trigrams}
                pbar.update()
                l_fullURL.append(item)
                df_split = pd.DataFrame(l_fullURL)
        if df_split.shape[0] != 0:
            return df_split
        else:
            return None

    def get_url(self, df):
        df['shortDate'] = pd.to_datetime(df['shortDate'], errors = 'coerce', format = '%Y-%m-%d %H:%M:%S')
        df['shortDate'] = df['shortDate'].astype('datetime64[ns]')
        
        df['user_created'] = pd.to_datetime(df['user_created'], errors = 'coerce', format = '%Y-%m-%d %H:%M:%S')
        df['user_created'] = df['user_created'].astype('datetime64[ns]')

        num_rows = df.shape[0]
        df_data = pd.DataFrame()
        group_size = 10
        partitions = self.intervals(group_size, num_rows)
        print(partitions)

        partitioned_data = []
        with Pool(processes = group_size) as proc:
            with tqdm(total=group_size) as pbar:
                for i, page in enumerate(proc.imap_unordered(self.work, partitions)):
                    partitioned_data.append(page)
                    pbar.update()
            df_data = df_data.append(partitioned_data)
            if df_data.shape[0] == 0:
                print('there are not enough shares!')
        #print(df_data.urls)
        df_data = df_data.explode('urls')
        #df_data = df_data[df_data['urls'].notnull()]
        
        df_data['host'] = df_data['urls'].str.replace('https://','')
        df_data['host'] = df_data['host'].str.replace('http://','')
        df_data['host'] = df_data['host'].str.replace('www.','')
        df_data['host'] = df_data['host'].str.split('/').str[0]
        df_data = df_data.reset_index(drop = True)
        print(df_data)
        df_data.to_csv('/home/ylee197/Covid/FP_gcn/data/output/full_URL_tweet.csv')
        return df_data

        #df_data.to_pickle('output/Ru_war_retweet_0220.pkl')

if __name__ == "__main__":
    df = pd.read_csv('/home/ylee197/Covid/FP_gcn/data/input/preprocessed_tweet_0303.csv')
    URL(df)
    