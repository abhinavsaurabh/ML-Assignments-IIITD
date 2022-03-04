#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from pandas import DataFrame
from sklearn import preprocessing 
import statistics


# In[2]:


df = pd.read_csv('PRSA.csv')


# In[3]:


df = df.fillna(0)


# In[4]:


label_encoder = preprocessing.LabelEncoder() 
df['cbwd']= label_encoder.fit_transform(df['cbwd'])


# In[5]:


def split(df):

# Shuffle your dataset 
    shuffle_df = df.sample(frac=1)

# Define a size for your train set 
    train_size = int(0.8 * len(df))

# Split your dataset 
    train_set = shuffle_df[:train_size]
    test_set = shuffle_df[train_size:]
    
    return train_set,test_set


# In[6]:


train_set, test_set = split(df)


# In[7]:


def feature(train_set,test_set):
    feature_cols = ['year','day','hour','pm2.5','DEWP','TEMP','PRES','cbwd','Iws','Is','Ir']
    X_train = train_set[feature_cols] # Features
    X_test = test_set[feature_cols]

    feature1 =['month']
    Y_train = train_set[feature1] # Target variable
    Y_test = test_set[feature1]
    
    return X_train,X_test,Y_train,Y_test


# In[9]:


def split50(df):
    shuffle_df = df.sample(frac=1)
    train_size = int(0.5 * len(df))
    train_set = shuffle_df[:train_size]
    return train_set


# In[10]:


#def split20(df):
 #   shuffle_df = df.sample(frac=1)
 #   test_size = int(0.5 * len(df))
 #   test_set = shuffle_df[:test_size]
 #   return test_set


# In[50]:


testpred={}
for i in range(100):     
    clf = DecisionTreeClassifier(max_depth=3)
    #train_set, test_set = split(df)
    train_set50= split50(train_set)
    X_train,X_test,Y_train,Y_test = feature(train_set,test_set)
    X_train1,X_test1,Y_train1,Y_test1 = feature(train_set50,test_set)
    clf = clf.fit(X_train1,Y_train1)
    y_testpred = clf.predict(X_test)
    testpred[i]=(y_testpred)
    #y_trainpred = clf.predict(X_train)
    #trainpred[k]= (y_trainpred)
    #y_trainpred = clf.predict(X_train)
    #list1.append([i,metrics.accuracy_score(Y_test, y_testpred),metrics.accuracy_score(Y_train, y_trainpred)])
    


# In[51]:


pred1 = pd.DataFrame.from_dict(testpred,orient ='index')


# In[52]:


pred2=pred1.mode()


# In[53]:


pred2 = pred2.loc[0, : ]


# In[54]:


metrics.accuracy_score(Y_test, pred2)


# In[ ]:




