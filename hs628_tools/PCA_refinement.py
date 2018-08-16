#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 14:43:54 2018

@author: BlauBear
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#find out which variables are correlated
#X is the independent variables, Y is the dependent variables, names is the variable names
def correlated_variables (X, Y, threshold, method):
    """This function finds out which variables are correlated
        Args: 
            X: numpy array containing independent variables
            Y: numpy array dependent variables
            threshold: correlation coefficient where we decide if the variables are correlated or not
            method: {‘pearson’, ‘kendall’, ‘spearman’}
                pearson : standard correlation coefficient
                kendall : Kendall Tau correlation coefficient
                spearman : Spearman rank correlation
        Returned: 
            corr_variables: index of feature columns with correlation greater than threshold
    """
    matrix = np.concatenate((X,Y), axis = 1)
    corr_matrix = np.absolute(np.corrcoef(matrix))
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    corr_variables = [column for column in upper.columns if any(upper[column] > threshold)]
    return corr_variables

def correlation_plot(X, Y):
    """This function helps to visualize the correlations so we can make better decisions
        Args: 
            X: numpy array containing independent variables
            Y: numpy array dependent variables
        Returned: 
            plot showing correlations
    """
    cm =np.corrcoef(np.concatenate((X,Y), axis = 1))
    plt.imshow(cm,interpolation='nearest')
    plt.colorbar()
    plt.show()

def specified_scaling (X, selected_fts):
    """ This function scales specific variables in the dataframe
         Args: 
            X: numpy array with independent variables
            selected_fts: index of feature columns, can be the list returned from correlated_variables()
        Returned: 
            df: Dataframe with the specificed variables scaled
    """
    standardscaling = StandardScaler()
    for i in selected_fts:
        X[:,i] = standardscaling.fit_transform(X[:,i])
    return X
    
def specified_PCA (X, components):
    """ This function performs PCA on selected variables
        Args: 
            X: numpy array which has the variables which need PCA performed
            components: list of index of the variables which need PCA, can be the list returned from correlated_variables()
            
        Returned: 
            finalX: numpy array with PCA performed on selected variables
    """
    num_components = len(components)
    principalComponents = []
    for i in components: 
        principalComponents = np.concatenate ((principalComponents, (X[:,i]).T),axis = 1)
    pca = PCA(n_components = num_components)
    principalComponents = pca.fit_transform(principalComponents)
    for i in np.size(X,1)-1: 
        if i not in components: 
            principalComponents = np.concatenate ((principalComponents, (X[:,i]).T),axis = 1)
    return principalComponents

##main##
