#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[4]:


df = pd.read_csv('iris.data')


# In[10]:


df


# In[18]:


x = df.iloc[:, [0, 1, 2, 3, 4]].values


# In[19]:


x


# In[5]:


feature_cols = ['sepal length', 'sepal width', 'petal length', 'petal width']
target = ['class']
X = df[feature_cols]
y = df[target]


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


# In[8]:


X_train


# In[9]:





# In[ ]:




