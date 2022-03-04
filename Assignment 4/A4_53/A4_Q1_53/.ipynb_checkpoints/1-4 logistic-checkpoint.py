#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import metrics


# In[2]:


df = pd.read_csv('iris.data')


# In[3]:


feature_cols = ['sepal length', 'sepal width','petal length', 'petal width']
target = ['class']
X = df[feature_cols]
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y = le.fit_transform(df['class'])


# In[4]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


# In[9]:


from sklearn.linear_model import LogisticRegression  


# In[12]:


clf= LogisticRegression(max_iter=10000,random_state=0)


# In[13]:


clf.fit(X_train,y_train)


# In[14]:


y_trainpred = clf.predict(X_train)


# In[15]:


print(metrics.accuracy_score(y_trainpred, y_train))


# In[16]:


y_testpred = clf.predict(X_test)


# In[17]:


print(metrics.accuracy_score(y_testpred, y_test))


# In[ ]:




