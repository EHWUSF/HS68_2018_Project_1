
# coding: utf-8

# In[233]:


get_ipython().magic(u'matplotlib inline')


# In[234]:




import matplotlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
#pd.test()


# In[235]:


df1 = pd.read_csv("C:/Users/Nitie/Desktop/health_informatics/MSHI_CLASSES/summer_2018/650/project data/parkinsons_train_set.csv")

df1.head(10)


# In[236]:




plt.hist(df1["HNR"], normed=True, bins=20)
plt.ylabel('FREQ');
plt.grid()
plt.rc('grid', linestyle="dashed", color='grey')



# In[237]:


cat=df1["status"].unique()
cat


# In[238]:


df1[["HNR","status"]]


# In[239]:


df1["HNR_status"] =list(zip(df1.HNR, df1.status))
df1["HNR_status"].head(10)


# Build a tuple of the column and class label

# In[240]:



def coltuple(df,col,typpe):
    tuplle=[]
    for i in range(len(df[col])):
        for k in range(len(df[typpe])):
            if i==k:
                #newtuple=col[i],typpe[k]
                #newtuple = ['col', 'typpe'].apply(tuple, axis=1)
                newtuple = list(zip(df[col], df[typpe]))
                #tuplle=tuplle+newtuple
        print newtuple
    return newtuple
            


# In[241]:


listgrp=coltuple(df1,'HNR','status')


# Build the groupby function

# In[242]:


def group_by(tpls,idx=0,merge=True):
    d = dict()
    for tpl in tpls:
        k = tpl[idx]
        v = d.get(k,tuple()) + (tpl[:idx]+tpl[idx+1:] if merge else (tpl[:idx]+tpl[idx+1:],))
        d.update({k:v})
    return d



# In[243]:


colgroupbyclass=group_by(listgrp,1)


# In[244]:


colgroupbyclass[1][24]


# Build probability density function normalized histogram for the groups

# In[245]:


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
#avg4 = np.mean(colgroupbyclass[0])
#print("Numpy Average: np.mean(heights) = {0}".format(avg),"Numpy Average: np.median(heights) = {0}".format(median))


# Call the groupby function and apply the arguments

# In[246]:


summarystat(colgroupbyclass)

