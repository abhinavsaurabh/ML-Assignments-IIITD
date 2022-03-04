#!/usr/bin/env python
# coding: utf-8

# In[11]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from pandas import DataFrame


# In[12]:


data1 = loadmat('dataset_2.mat')
data2 = data1['samples']
mlabel = data1['labels'][0]
data3 = np.column_stack((data2,mlabel))
df = pd.DataFrame(data3)


# In[3]:


def split(df):

# Shuffle your dataset 
    shuffle_df = df.sample(frac=1)

# Define a size for your train set 
    train_size = int(0.7 * len(df))

# Split your dataset 
    train_set = shuffle_df[:train_size]
    test_set = shuffle_df[train_size:]
    
    return train_set,test_set  


# In[4]:


train_set, test_set = split(df)


# In[5]:


feature_cols = [0,1]
X_train = train_set[feature_cols] # Features
X_test = test_set[feature_cols]

feature1 =[2]
Y_train = train_set[feature1] # Target variable
Y_test = test_set[feature1]


# In[13]:


list1=[]
for i in range(15):
    clf = DecisionTreeClassifier(max_depth=i+1)
    clf = clf.fit(X_train,Y_train)
    y_testpred = clf.predict(X_test)
    y_testpred1=np.array(y_testpred)
    y_test1=np.array(Y_test)
 
    count=0
    for j in range(y_testpred1.size):
        if y_testpred1[j]==y_test1[j]:
            count+=1;
    count1=0        
    y_trainpred = clf.predict(X_train)        
    y_trainpred1=np.array(y_trainpred)
    y_train1=np.array(Y_train)
    for j in range(y_trainpred1.size):
        if y_trainpred1[j]==y_train1[j]:
            count1+=1;
    
    #print( acc)
    list1.append([i+1,count/y_testpred1.size,count1/y_trainpred1.size])


# In[14]:


list1


# In[15]:


df2 = DataFrame (list1,columns=['Depth','Test Accuracy','Train Accuracy'])


# In[16]:


df2


# In[ ]:




