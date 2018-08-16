# Module designed to facilitate visualization of descriptive summary statistics in Python.
# coding: utf-8

get_ipython().magic(u'matplotlib inline')

import matplotlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def main():
    printIntro()
    df1, col, col_cat = getInputs()
    listgrp=coltuple(df1,col,col_cat)# helper function
    colgroupbyclass=group_by(listgrp,1) # helper function
    print(summarystat(colgroupbyclass))

def printIntro():
               Print("The purpose of this program is to visualize some descriptive statistics grouped by column")

def getInputs() :
        """
           Build a tuple from the column and class label
            -----------
            Parameters:
            Prompt User for name categorical column
            User input for name of target column
            Prompt user for data frame
            -----------
            Returns:
            Variables representing categorical column,target column and dataframe
        """
               #Returns the three input parameters
               col_cat=(input("what is col_cat name ?"))
               col=float(input("what is the col name ?"))
               df1 = pd.read_csv("C:/Users/Nitie/Desktop/health_informatics/MSHI_CLASSES/summer_2018/650/project data/parkinsons_train_set.csv")
               return col_cat,col,df1

#Function 1: Build a tuple of the column and class label
"""
           Build a tuple from the column and class label
            -----------
            Parameters:
            Defined variable assigned to the input data frame
            The column to be analysed and grouped
            The categorical variable whose levels are used for the grouping
            -----------
            Returns:
                A tuple of column elements grouped by the levels in the categorical variable
        """
def coltuple(df,col,typpe):
    tuplle=[]
    for i in range(len(df[col])):
        for k in range(len(df[typpe])):
            if i==k:
                #newtuple=col[i],typpe[k]
                #newtuple = ['col', 'typpe'].apply(tuple, axis=1)
                newtuple = list(zip(df[col], df[typpe]))
                #tuplle=tuplle+newtuple
        print (newtuple)
    return newtuple


#Test the tuple generator
#df1 = pd.read_csv("C:/Users/Nitie/Desktop/health_informatics/MSHI_CLASSES/summer_2018/650/project data/parkinsons_train_set.csv")
#listgrp=coltuple(df1,'HNR','status')
# Function 2:Build the groupby function
    """
          Build the groupby function -(merge into  dictionary with value and key) 
          The target column element is the value and the key is the categorical level e.g 0 or 1
            -----------
            Parameters:
            Tuple made up of target column (value) & categorical column levels(key)
            The key index in tuple
            -----------
            Returns:
                A merged dictionary comprising of elements grouped by the keys.
        """

def group_by(tpls,idx=1,merge=True):
    d = dict()
    for tpl in tpls:
        k = tpl[idx]
        v = d.get(k,tuple()) + (tpl[:idx]+tpl[idx+1:] if merge else (tpl[:idx]+tpl[idx+1:],))
        d.update({k:v})
    return d

# Test the groupby function
#colgroupbyclass=group_by(listgrp)
# Function 3:Build probability density function normalized histogram for the groups
    """
          Build probability density function normalized for the groups
            -----------
            Parameters:
            Dictionary made of the value (target column) & key (categorical variable levels)
            -----------
            Returns:
            Specified descriptive statistics (avg,median,stdev) grouped by categorical column levels
            Plot of probability density function normalized for the groups
        """
def summarystat(listd):
    for i in range(len(listd)):
        avg=np.mean(listd[i])
        median=np.median(listd[i])
        stdev=np.std(listd[i])
        plt.hist(listd[i], normed=True, bins=20)
        plt.title('The "Normal" Distribution with Mean & St. Devs.')
        plt.xlabel("Variables"); plt.ylabel("Frequency")
        plt.grid(True)
        plt.rc('grid', linestyle="dashed", color='grey')
        plt.show()
        print(i,"Mean = {0}".format(avg),"Median = {0}".format(median),"stdev = {0}".format(stdev))
    return avg,median,stdev

# Call the groupby function and apply the arguments
#summarystat(colgroupbyclass)
#summarystat(d)

if __name__ == "__main__": main()


           



    
