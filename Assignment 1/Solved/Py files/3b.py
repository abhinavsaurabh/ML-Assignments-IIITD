#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from pandas import DataFrame
from sklearn import preprocessing 


# In[3]:


df = pd.read_csv('PRSA.csv')


# In[4]:


df = df.fillna(0)


# In[5]:


df['cbwd'].unique() 


# In[6]:


label_encoder = preprocessing.LabelEncoder() 
df['cbwd']= label_encoder.fit_transform(df['cbwd'])


# In[7]:


df['cbwd'].unique() 


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


# In[11]:


depths= [2, 4, 8, 10, 15, 30]


# In[15]:


list1=[]
for i in depths:     
    clf = DecisionTreeClassifier(max_depth=i)
    clf = clf.fit(X_train,Y_train)
    y_testpred = clf.predict(X_test)
    y_trainpred = clf.predict(X_train)
    list1.append([i,metrics.accuracy_score(Y_test, y_testpred),metrics.accuracy_score(Y_train, y_trainpred)])


# In[16]:


df2 = DataFrame (list1,columns=['Depth','Test Accuracy','Train Accuracy'])


# In[17]:


df2


# In[21]:


plt.plot(df2['Depth'],df2['Test Accuracy'],label='Test Accuracy')
plt.plot(df2['Depth'],df2['Train Accuracy'],label='Train Accuracy')
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.legend()


# In[ ]:




