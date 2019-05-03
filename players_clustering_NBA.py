#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 01:55:08 2019

@author: ruiqianyang
"""

import requests  
import json  
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
import time

players = []  
stats=[]

def find_stats(player_id):
    
    print (player_id)
    url = 'https://stats.nba.com/stats/playercareerstats?PerMode=PerGame&PlayerID='+player_id
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.106 Safari/537.36'}

    #Create Dict based on JSON response
    response = requests.get(url,headers=headers)
    shots = response.json()['resultSets'][0]['rowSet']
    data = json.loads(response.text)

    headers = data['resultSets'][0]['headers']
    shot_data = data['resultSets'][0]['rowSet']
    #choose latest statstics with iloc[-1]
    if len(shot_data)>0:
        df = pd.DataFrame(shot_data,columns=headers).iloc [-1]

#just choose a few continuous feature related to being aggressive/defensive
        df=df[["PLAYER_ID","PLAYER_AGE","FGM","FTM","OREB","DREB","AST","PF","PTS"]]
        stats.append(df)
    
    
# if intersted in analyze mean of statstics for each player through different years,
# we can grab all data and calculate mean statistics for each player and genberate dataframe as follows:
#    fgm=df["FGM"].mean(axis=0)
#    ftm=df["FTM"].mean(axis=0)
#    oreb=df["OREB"].mean(axis=0)
#    dreb=df["DREB"].mean(axis=0)
#    pf=df["PF"].mean(axis=0)
#    df_mean={"PLAYER_ID":None,"FGM":None,"FTM":None,"OREB":None,"DREB":None,"PF":None}
#    df_mean["PLAYER_ID"]=player_id
#    df_mean["FGM"]=fgm
#    df_mean["FTM"]=ftm
#    df_mean["OREB"]=oreb
#    df_mean["DREB"]=dreb
#    df_mean["PF"]=pf
    
    
  
#grab list of players:
url_p='https://stats.nba.com/stats/commonallplayers?LeagueId=00&Season=2016-17&IsOnlyCurrentSeason=0'    
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}

response_p = requests.get(url_p, headers=headers)
shots_p = response_p.json()['resultSets'][0]['rowSet']
data_p = json.loads(response_p.text)

#get players ID
headers_p = data_p['resultSets'][0]['headers']
shot_data_p = data_p['resultSets'][0]['rowSet']
df_p = pd.DataFrame(shot_data_p,columns=headers_p)   
players=df_p[['PERSON_ID','DISPLAY_FIRST_LAST']][:500]     #subset of players size=500  

time.sleep(1) 

flg=0
#Generate stats data for clustering:
for i in players.iloc[:,0]:
#    time.sleep(5) 
    flg+=1
    find_stats(str(i))
headers_stat=["PLAYER_ID","PLAYER_AGE","FGM","FTM","OREB","DREB","AST","PF","PTS"]
stats_df=pd.DataFrame(stats,columns=headers_stat)
stats_df=stats_df.dropna(axis=0, how='any')#remove NaN
stats_df.info()

## Check which number of clusters works best
k_list = [2,3,4,5,8,10]

best={}   
## Run clustering with different k and check the metrics
def compare_k_means(k_list,data):
    
    for k in k_list:
        clusterer = KMeans(n_clusters=k,n_jobs=4)
        clusterer.fit(data)
        print("Silhouette Coefficient for k == %s: %s" % (
        k, round(metrics.silhouette_score(data, clusterer.labels_), 4)))
## Choose Silhouette to optimze k.The higher Silhouette Coefficient(up to 1) the better
        best[k]=round(metrics.silhouette_score(data, clusterer.labels_), 4)
        
        

compare_k_means(k_list,stats_df)
import operator
#choose best k, where silhouette score is the maximum
best_k=max(best.items(), key=operator.itemgetter(1))[0]

#if time is enough I would like the ELBOW method to get the optimal value of K 


clusterer_final = KMeans(n_clusters=best_k,random_state=0,n_jobs=4)
#kmeans_model is the final model 
kmeans_model = clusterer_final.fit_predict(stats_df)

#add Pred column to dataframe
stats_df['Pred']=kmeans_model 
stats_df=pd.merge(stats_df, players, how='left', left_on='PLAYER_ID', right_on='PERSON_ID')

stats_df['first_name']=stats_df['DISPLAY_FIRST_LAST'].apply(lambda x:x.split(' ')[0])
stats_df['last_name']=stats_df['DISPLAY_FIRST_LAST'].apply(lambda x:x.split(' ')[1])
stats_df=stats_df.drop(["DISPLAY_FIRST_LAST","PERSON_ID","FGM","FTM","OREB","DREB","AST","PF","PTS"], axis=1)

stats_df
#Print final player clusters
output={}
for i in range(best_k):
    cls=[]
    for row,v in stats_df[stats_df['Pred']==i].iterrows():
        item={}
        item['player_id']=v[0]
        item['first_name']=v[3]
        item['last_name']=v[4]
        cls.append(item)        
    output[i]=cls
    
print(output)

with open('Q5_subset_size_500_output.txt', 'w') as f:
    print(output, file=f)
f.close()


