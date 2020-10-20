#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 03:15:08 2019

@author: mintmsh
"""

import pandas as pd
import numpy as np
import sys
import random as rd
import time
import math

def computeMSE(predict_y, y):
    return math.sqrt(np.sum((predict_y - y)**2)/predict_y.shape[0])

def filter_1st(row):
    if row["userId"] <= 28000:
        return True
    return False

def split(data):
    y=data['rating']
    x=data.drop(['rating'], axis=1)
    return x,y

def little(test):
    return test[test.apply(filter_1st, axis=1)]


def read():
    linear = pd.read_csv("total_user_beta1.csv")
    linear2 = pd.read_csv("total_user_beta2.csv")
    linear3 = pd.read_csv("total_user_beta3.csv")
    linear4 = pd.read_csv("total_user_beta4.csv")
    linear5 = pd.read_csv("total_user_beta5.csv")
    linear = linear.append(linear2)
    linear = linear.append(linear3)
    linear = linear.append(linear4)
    linear = linear.append(linear5)

    return linear
    
    
def start_row(gs):
    g = gs.groupby('movieId')
    count = g.count()
    count['movieId'] = count.index
    count = count[['movieId','tagId']]
    count.columns = ['movieId','start']
    
    for i in range(len(count)):
        count.iloc[i]['start'] = i * 1128   
    return count 

def rounding(y):
    y2 = np.zeros(test_y.shape[0])
    for i in range(y.shape[0]):
        y2[i] = round(y[i] * 2.0) / 2
        
    return y2
    


if __name__ == '__main__':

    # Given beta for each user, test validation set
    linear = read()
    
    val = pd.read_csv("val_ratings.csv")
    gs = pd.read_csv("genome-scores.csv")
    #val = val.iloc[:1000]
    
    Id = pd.Series(range(1,linear.shape[0] + 1))
    
    linear['userId'] = Id
    linear = linear.set_index(['userId'])
    
    start = start_row(gs)
    
    movie_list = start["movieId"].tolist()
    c_test = val[val.movieId.isin(movie_list)]
    
    
    val_clean_x, val_clean_y=split(c_test)
    
    y = np.zeros(c_test.shape[0])

    for i in range(y.shape[0]):
        s = start.loc[c_test.iloc[i]['movieId'], 'start']
        a = gs.iloc[s:s+1128]['relevance']
        line = int(c_test.iloc[i]['userId']) - 1
        v = linear.iloc[line]
        bias = v[0]
        v = v[1:]
        y[i] = np.dot(a.T,v.T) + bias
        
        if y[i] > 5:
            y[i] = 5
        elif y[i] < 0.5:
            y[i] = 0.5


    print ("Clean MSE: ", computeMSE(y, c_test['rating']))
    c_test = c_test.assign(rating = y)
    
    
    val_x, r = split(val)
    val = pd.merge(val_x, c_test, on=['userId', 'movieId'], how='left')
    
    
    val_y = val['rating']
    empty = np.where(val_y.isna())[0]
    
    
    movie_avg = pd.read_csv("movie_ave.csv", index_col=0)
    user_avg = pd.read_csv("user_ave.csv", index_col=0)
    
    
    y_user = val_y.copy()
    y_movie = val_y.copy()
    y_avg = val_y.copy()

    for i in range(len(empty)):
        il = empty[i]
        user_l = int(val.iloc[il]['userId'])
        movie_l = int(val.iloc[il]['movieId'])
        user = 2.5
        movie = 2.5
        
        if user_l in user_avg.index:
            user = float(user_avg.loc[user_l])
        if movie_l in movie_avg.index:
            movie = float(movie_avg.loc[movie_l])
        
        y_user.iloc[il] = user
        y_movie.iloc[il] = movie
        y_avg.iloc[il] = (user + movie) / 2
        
    
    print ("User-based Linear Regression MSE: ", computeMSE(y_user, r))

    y_user.to_csv('val_linear_y_user.csv',index=False)
    #y_avg.to_csv('val_linear_y_avg.csv',index=False)

###########################   test set prediction   ##############################
    '''
    linear = read()
    test = pd.read_csv("test_ratings.csv")
    gs = pd.read_csv("genome-scores.csv")
    
    start = start_row(gs)
    movie_list = start["movieId"].tolist()
    '''
    test = pd.read_csv("test_ratings.csv")
    #test = test.iloc[:1000]
    c_test = test[test.movieId.isin(movie_list)]
    
    y = np.zeros(c_test.shape[0])
    
    for i in range(y.shape[0]):
        s = start.loc[c_test.iloc[i]['movieId'], 'start']
        a = gs.iloc[s:s+1128]['relevance']
        line = int(c_test.iloc[i]['userId']) - 1
        v = linear.iloc[line]
        bias = v[0]
        v = v[1:]
        y[i] = np.dot(a.T,v.T) + bias
        
        if y[i] > 5:
            y[i] = 5
        elif y[i] < 0.5:
            y[i] = 0.5
    
    clean_test = c_test.assign(rating = y)
    all_test = pd.merge(test, clean_test, on=['Id', 'userId', 'movieId'], how='left')
    
    all_y = all_test['rating']
    empty = np.where(all_y.isna())[0]
    

    y_user = all_y.copy()
    y_movie = all_y.copy()
    y_avg = all_y.copy()
    
    for i in range(len(empty)):
        il = empty[i]
        user_l = int(test.iloc[il]['userId'])
        movie_l = int(test.iloc[il]['movieId'])
        user = 2.5
        movie = 2.5
        
        if user_l in user_avg.index:
            user = float(user_avg.loc[user_l])
        if movie_l in movie_avg.index:
            movie = float(movie_avg.loc[movie_l])
        
        y_user[il] = user
        y_movie[il] = movie
        y_avg[il] = (user + movie) / 2

    
    

    submission = pd.read_csv("kaggle_sample_submission.csv")
    submission = submission.assign(rating = y_user)
    
    submission.to_csv('linear_submisssion2.csv',index =False)
    
    all_y.to_csv('test_linear_y2.csv',index=False)
    #pd.DataFrame(empty).to_csv('test_empty_index.csv',index=False)

    