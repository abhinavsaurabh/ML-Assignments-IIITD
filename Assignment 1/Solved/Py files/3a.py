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


# In[2]:


df = pd.read_csv('PRSA.csv')


# In[3]:


df


# In[4]:


df['cbwd'].unique() 


# In[5]:


label_encoder = preprocessing.LabelEncoder() 
df['cbwd']= label_encoder.fit_transform(df['cbwd'])


# In[6]:


df['cbwd'].unique()


# In[7]:


df = df.fillna(0)


# In[8]:


def split(df):

# Shuffle your dataset 
    shuffle_df = df.sample(frac=1)

# Define a size for your train set 
    train_size = int(0.8 * len(df))

# Split your dataset 
    train_set = shuffle_df[:train_size]
    test_set = shuffle_df[train_size:]
    
    return train_set,test_set  


# In[9]:


train_set, test_set = split(df)


# In[10]:


feature_cols = ['year','day','hour','pm2.5','DEWP','TEMP','PRES','cbwd','Iws','Is','Ir']
X_train = train_set[feature_cols] # Features
X_test = test_set[feature_cols]

feature1 =['month']
Y_train = train_set[feature1] # Target variable
Y_test = test_set[feature1]


# In[16]:


clf = DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X_train,Y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))


# In[12]:


clf = DecisionTreeClassifier(criterion="gini")
clf = clf.fit(X_train,Y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))


# In[ ]:




