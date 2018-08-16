"""
This package aids the user to display descriptive statistics of 
specified column grouped by the class label
Input: Column name ,class label name
Returns: A graphical plot of specified descriptive statistics

"""
import matplotlib
import re
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


class vizsumstat:
        def __init__(self, csvPath, colname, col_catname):
                self.df = pd.read_csv(csvPath)
                self.colname = colname
                self.col_catname = col_catname
                #print(self.colname)
                self.tuple = self.coltuple(self.df, self.colname, self.col_catname)
                self.colgroupbyclass=self.group_by(self.tuple)

                
        def group_by(self,tpls,idx=1,merge=True):
                """
                This helper function builds a dictionary
                input: Tuple of column element , class label values on same row index
                Returns Dictionary of key(class element) and value(column element)
                """
                d = dict()
                for tpl in tpls:
                        k = tpl[idx]
                        v = d.get(k,tuple()) + (tpl[:idx]+tpl[idx+1:] if merge else (tpl[:idx]+tpl[idx+1:],))
                        d.update({k:v})
                return d

        def coltuple(self,df,col,typpe):
                """
                Helper function builds a tuple of elements
                Input: Elements of column to be described,class label categorical values
                Returns : A tuple of combined elements 
                """
                #tuplle=[]
                for i in range(len(df[col])):
                        for k in range(len(df[typpe])):
                                if i==k:
                                        #newtuple=col[i],typpe[k]
                                        #newtuple = ['col', 'typpe'].apply(tuple, axis=1)
                                        newtuple = list(zip(df[col], df[typpe]))
                                        #tuplle=tuplle+newtuple
                                #print (newtuple)
                        return newtuple
                
        def summarystat(self):
                """
                Function generates specified descriptive summary statistics for the value elements grouped 
                by the key corresponding categorical key
                Input: Elements stored in the dictionary
                Returns: Summary statistics plot and calculated numeric values
                """
                for i in range(len(self.colgroupbyclass)):
                       avg=np.mean(self.colgroupbyclass[i])
                       median=np.median(  self.colgroupbyclass[i])
                       stdev=np.std(  self.colgroupbyclass[i])
                       plt.hist(  self.colgroupbyclass[i], density=True, bins=20)
                       plt.title('Descriptive statistics: \n\t')
                       plt.xlabel("Described variable"); plt.ylabel("Frequency")
                       plt.grid(True)
                       plt.rc('grid', linestyle="dashed", color='grey')
                       plt.tight_layout()
                       plt.show()
                       #print(avg,median,stdev)
                       print(i,"Mean = {0}".format(avg),"Median = {0}".format(median),"stdev = {0}".format(stdev))
                return avg,median,stdev

def main():
        myfile=input("Input path to your file:     e.g C:/Desktop/650/project data/parkinsons_train_set.csv : _")
        f=open(myfile,'r')
        col=input("input col name")
        col_catname=(input("inputcategorical col name: "))
        #path  = "C:/Users/Nitie/Desktop/health_informatics/MSHI_CLASSES/summer_2018/650/project data/parkinsons_train_set.csv"
        viz=vizsumstat(f,col,col_catname)
        viz.summarystat()
        f.close()

        
if __name__ == "__main__": main()





