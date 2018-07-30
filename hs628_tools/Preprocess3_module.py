import matplotlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


class vizsumstat:

        def __init__ (self,csvPath,colname,col_catname):

                self.df = pd.read_csv(csvPath)
                self.colname = colname
                self.col_catname = col_catname
                #print(self.colname)
                self.tuple = self.coltuple(self.df, self.colname, self.col_catname)
                self.colgroupbyclass=self.group_by(self.tuple)

                
        def group_by(self,tpls,idx=1,merge=True):
                d = dict()
                for tpl in tpls:
                        k = tpl[idx]
                        v = d.get(k,tuple()) + (tpl[:idx]+tpl[idx+1:] if merge else (tpl[:idx]+tpl[idx+1:],))
                        d.update({k:v})
                return d

        def coltuple(self,df,col,typpe):
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
                for i in range(len(self.colgroupbyclass)):
                       avg=np.mean(self.colgroupbyclass[i])
                       median=np.median(  self.colgroupbyclass[i])
                       stdev=np.std(  self.colgroupbyclass[i])
                       plt.hist(  self.colgroupbyclass[i], normed=True, bins=20)
                       plt.title('The "Normal" Distribution with Mean & St. Devs.')
                       plt.xlabel("Variables"); plt.ylabel("Frequency")
                       plt.grid(True)
                       plt.rc('grid', linestyle="dashed", color='grey')
                       plt.show()
                       #print(avg,median,stdev)
                       print(i,"Mean = {0}".format(avg),"Median = {0}".format(median),"stdev = {0}".format(stdev))
                return avg,median,stdev

def main():
        col=(input("input col name"))
        col_catname=(input("inputcategorical col name: "))
        path  = "C:/Users/Nitie/Desktop/health_informatics/MSHI_CLASSES/summer_2018/650/project data/parkinsons_train_set.csv"
        viz=vizsumstat(path,col,col_catname)
        viz.summarystat()

        
if __name__ == "__main__": main()
