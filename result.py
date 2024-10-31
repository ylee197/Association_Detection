import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def result():
    df_target = pd.read_csv('data/target.csv', dtype = str)
    df_pred = pd.read_csv('data/predict_output.csv', dtype = str)
    df_top = df_pred.groupby(['node_index'], sort=False)['predict'].max()
    df_merge = df_target.merge(df_top, left_on = 'index', right_on = 'node_index', how = 'left')
    print(df_merge.head(10))

    for index, row in df_merge.iterrows():
        if row['code'] == '-1':
            df_merge.loc[index, 'code'] = row['predict']
    df_result = df_merge[['index','users','code']]
    df_merge[['index','users','code']].to_csv('data/FP_result.csv', index = False)

    df_result.groupby('code').size()
    df_target.groupby('code').size()
    
    ## Testing accuracy ##
    df_target = df_target[df_target['code'] != '-1']
    df_test = df_merge[df_merge['index'].isin(df_target['index'].tolist())]
    df_test = df_test.dropna()
    y_true = df_target[df_target['index'].isin(df_test['index'].tolist())]['code']
    print(confusion_matrix(y_true, df_test['code']))
    print('accuracy : '+str(accuracy_score(y_true,df_test['code'])))
    