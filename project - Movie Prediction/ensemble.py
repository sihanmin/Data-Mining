#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 02:25:10 2019

@author: mintmsh
"""
import pandas as pd
import numpy as np
import sys
import random as rd
import time
import math
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def computeMSE(predict_y, y):
    return math.sqrt(np.sum((predict_y - y)**2)/predict_y.shape[0])

def split(data):
    y=data['rating']
    x=data.drop(['rating'], axis=1)
    return x,y

def rate(row, b1, b2):
    return row['lin_u']*b1 + row['svd']*b2

def predict(row, b1, b2, bias):
    v = row['lin_u']*b1 + row['svd']*b2 + bias
    if v > 5:
        return 5
    elif v < 0.5:
        return 0.5
    return v

def seperate_train(train, l, m):
    index = np.random.choice(l,m, False) # randomly choose 90% as training set
    mask = np.ones(l, dtype=bool)
    mask[index] = False
    
    t = train[train.index.isin(index)]
    v = train[mask]
    
    tx, ty = split(t)
    vx, vy = split(v)
    return tx, ty, vx, vy
    

def holdout(n, train, B=True):
    l = len(val_y)
    m = int(l * 9 / 10)
    beta0 = []
    beta1 = []
    bias = []
    rmse = []
    
    for i in range(n):
        # Generate a uniform random sample 
        # from np.arange(l) of size m without replacement
        tx, ty, vx, vy = seperate_train(train, l, m)
        
        reg = LinearRegression().fit(tx, ty)
        beta = reg.coef_
        b = reg.intercept_
        
        py = reg.predict(vx)
        
        error = computeMSE(py, vy)
        
        print ("Trail ", i , " : ")
        print (beta, b)
        print ("RMSE : ", error)
        beta0.insert(0, beta[0])
        if B == True:
            beta1.insert(0, beta[1])
        bias.insert(0, b)
        rmse.insert(0, error)
        #accuracies.append(accuracy)
    return beta0, beta1, bias, rmse
        

if __name__ == '__main__':
    
    val = pd.read_csv("val_ratings.csv")
    _, val_y = split(val)
    val_lin_y = pd.read_csv("val_linear_y_user.csv")
    #val_mfknn_y = pd.read_csv("val_mfknn_y.csv")
    val_svd_y = pd.read_csv("val_svd_y.csv")

    #train_x = val_lin_y.join(val_mfknn_y, how='outer').join(val_svd_y, how='outer')
    train = val_lin_y.join(val_svd_y, how='outer').join(val_y, how='outer')
    

    beta0, beta1, bias, rmse = holdout(1000, train)
    
    beingsaved = plt.figure()
    plt.title("Histogram of W1 (1000 Trails)")
    plt.hist(beta0,bins= 25)
    plt.show()
    beingsaved.savefig("W1.jpg", format='jpg', dpi=900)
    
    beingsaved = plt.figure()
    plt.title("Histogram of W2 (1000 Trails)")
    plt.hist(beta1,bins= 25)
    plt.show()
    beingsaved.savefig("W2.jpg", format='jpg', dpi=900)
    
    Beta0 = pd.Series(beta0).mean(axis=0)
    Beta1 = pd.Series(beta1).mean(axis=0)


    train['beta'] = train.apply(rate, args=(Beta0, Beta1), axis=1)
    
    
    _, _, bias, _ = holdout(1000, train[['beta', 'rating']], B = False)
    Bias = pd.Series(bias).mean(axis=0)
    
    beingsaved = plt.figure()
    plt.title("Histogram of Bias (1000 Trails)")
    plt.hist(bias,bins= 25)
    plt.show()
    beingsaved.savefig("Bias.jpg", format='jpg', dpi=900)

    test_lin_y = pd.read_csv("test_linear_y2.csv")
    test_svd_y = pd.read_csv("test_svd_y.csv")
    #test_svd = pd.read_csv("svd27.csv")
    #test_svd_y = test_svd["svd"]
    #test_mfknn_y = pd.read_csv("test_mfknn_y.csv")
    
    #pred_x = test_lin_y.join(test_mfknn_y, how='outer').join(test_svd_y, how='outer')
    pred_x = test_lin_y.join(test_svd_y, how='outer')
    pred_y = pred_x.apply(predict, args=(Beta0, Beta1, Bias), axis=1)
    pred_y = pd.DataFrame(pred_y, columns=['rating'])
  
    
    
    submission = pd.read_csv("kaggle_sample_submission.csv")
    submission = submission.assign(rating = pred_y)
    submission.to_csv('lin_svd_submission2.csv',index=False)
    #test_dh_y['rating'].to_csv('test_dh_y.csv',index=False)