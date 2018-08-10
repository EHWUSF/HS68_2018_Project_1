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

def specified_scaling (df, names):
    """ This function scales specific variables in the dataframe
         Args: 
            df: dataframe to be scaled
            names: list of the variables which need scaling, names can be the list returned from correlated_variables()
        Returned: 
            df: Dataframe with the specificed variables scaled
    """
    standardscaling = StandardScaler()
    df[names] = standardscaling.fit_transform(df[names])
    return df
    
def specified_PCA (df, names, num_components):
    """ This function performs PCA on selected variables
        Args: 
            df: dataframe which has the variables which need PCA performed
            names: list of the variables which need PCA
        Returned: 
            finalDf: dataframe with PCA performed on selected variables
    """
    pca = PCA(n_components= num_components)
    principalComponents = pca.fit_transform(df[names])
    principalDf = pd.DataFrame(data = principalComponents)
    finalDf = pd.concat([principalDf, df[-[names]]], axis = 1)
    return finalDf

##main##
