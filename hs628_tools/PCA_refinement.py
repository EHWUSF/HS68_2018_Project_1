#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 14:43:54 2018

@author: BlauBear
"""

import numpy as np
import pandas as pd
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

#find out which variables are correlated
#X is the independent variables, Y is the dependent variables, names is the variable names
def correlated_variables (X, Y, names):
    rf = RandomForestRegressor(n_estimators=20, max_depth=4)
    scores = []
    for i in range(X.shape[1]):
        score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                              cv=ShuffleSplit(len(X), 3, .3))
        scores.append((round(np.mean(score), 3), names[i]))
    return (sorted(scores, reverse=True))

#performing scaling
#df is the dataframe to be scaled
#names is a list of the variables which need scaling
#the dataframe is returned

def specified_scaling (df, names):
    standardscaling = StandardScaler()
    df[names] = standardscaling.fit_transform(df[names])
    return df
    
#performing PCA on selected variables
#df is the dataframe which has the variables which need PCA performed
#names is a list of the variables which need PCA
#the dataframe is returned
def specified_PCA (df, names, num_components):
    pca = PCA(n_components= num_components)
    principalComponents = pca.fit_transform(df[names])
    principalDf = pd.DataFrame(data = principalComponents)
    finalDf = pd.concat([principalDf, df[-[names]]], axis = 1)
    return finalDf


##main##
