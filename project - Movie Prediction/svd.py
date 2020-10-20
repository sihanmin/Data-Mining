#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 23:19:38 2019

@author: mintmsh
"""


import pandas as pd
import numpy as np
import sys
import random as rd
import time
import math

from time import perf_counter as pc
import scipy.sparse as sps
from sklearn.decomposition import NMF

def computeMSE(predict_y, y):
    return math.sqrt(np.sum((predict_y - y)**2)/predict_y.shape[0])

def split(data):
    y=data['rating']
    x=data.drop(['rating'], axis=1)
    return x,y

if __name__ == '__main__':
    
    #test=pd.read_csv("val_ratings.csv")
    #gs = pd.read_csv("genome-scores.csv")
    '''
    train = pd.read_csv("train_ratings.csv")
    
    user_num = int(max(train['userId']))
    
    g = train.groupby('movieId')
    count = g.count()
    
    count['movieId'] = count.index
    count = count[['movieId','userId']]
    count.columns = ['movieId','count']
    
    new_id = pd.Series(list(range(count.shape[0])))
    
    '''

    '''
    movie_num = count.shape[0]
    
    lm = count.copy()
    lm = lm.set_index([pd.Index(list(range(movie_num)))])
    
    lm['newId'] = lm.index
    
    ml = lm.copy()
    ml = ml.set_index(['movieId'])
    
    ml = ml[['newId']]
    lm = lm[['movieId']]
    '''

    #movieId2 = train['movieId'].copy()
    '''
    for i in range(4520000,4590000):
        movieId2[i] = train.iloc[i]['movieId']
        
        if movieId2.iloc[i] in ml.index:
            movieId2.iloc[i] = int(ml.loc[movieId2.iloc[i]])
        else:
            print("##################")
            print(i)
            print(movieId2.iloc[i])
            
        
        if i % 10000 == 0:
            print(i)
            

    #movieId2.to_csv('movieId2.csv',index=False)
    

  
       # nnz_i, nnz_j, nnz_val = np.random.choice(140000, size=45000), \
    #                        np.random.choice(13000, size=45000), \
     #                       np.random.random(size=45000)

    #userId = np.array(train["userId"].tolist())
    #movieId = np.array(movieId2.tolist())
    #rating = np.array(train["rating"].tolist())
    #user_num = int(max(train['userId'])) + 1
    #movie_num = int(ml.shape[0])
    
    #X =  sps.csr_matrix((rating, (userId, movieId)), shape=(user_num, movie_num))
    '''
    print('X-shape: ', X.shape, ' X nnzs: ', X.nnz)
    print('type(X): ', type(X))
    # <class 'scipy.sparse.csr.csr_matrix'> #                         
    
    """ NMF """
    model = NMF(n_components=30, init='random', tol=1.0e-50, max_iter=20000, verbose=True)
    
    #model = NMF(n_components=50, init='custom', tol=0.0001, max_iter=1000, verbose=True)
    
    start_time = pc()
    W = model.fit_transform(X)
    #W_start = np.nan_to_num(W)
    #W = model.fit_transform(X, W=W_start)
    end_time = pc()
    
    #pd.DataFrame(W).to_csv("W.csv")
    H = model.components_
    H = H.transpose()
    #pd.DataFrame(H).to_csv("H.csv")
    
    print('Used (secs): ', end_time - start_time)
    print(model.reconstruction_err_)
    print(model.n_iter_)

    
    
    
    #H = H.transpose()
    
    #val = pd.read_csv("val_ratings.csv")
    #val_x, val_y=split(val)
    
    
    #val_pred_y = [W[int(val_x.iloc[i]['userId'])].dot(H[int(ml.loc[int(val_x.iloc[i]['movieId'])])]) for i in range(len(val_x))]
    '''
    movie_mean = H.mean(axis = 0)
    
    val_pred_y = np.zeros(len(val_x))
    '''
    '''
    start = 127000
    end = start + 100
    for i in range(start, end):
        user = W[int(val_x.iloc[i]['userId'])]
        movie_i = int(val_x.iloc[i]['movieId'])
        if (movie_i in ml.index):
            movie = H[int(ml.loc[movie_i])]
        else:
            movie = movie_mean
        
        user = user * 5 / sum(user)
        movie = movie * 5.0 / sum(movie)
        
        temp[i-start] = user.dot(movie)
        
        if i % 10000 == 0:
            print(i)
     '''
    '''
    missing = 0
    for i in range(len(val_x)):
        if (movie_i not in ml.index):
            miss += 1
    '''
    '''
    print (computeMSE(val_pred_y, val_y))
    #val_pred_y.to_csv('val_MF50_y.csv',index=False)
    '''