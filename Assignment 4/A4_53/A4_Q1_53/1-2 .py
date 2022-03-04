#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df = pd.read_csv('iris.data')


# In[ ]:





# In[4]:


feature_cols = ['sepal length', 'sepal width', 'petal length', 'petal width']
target = ['class']
X = df[feature_cols]
y = df[target]


# In[7]:


import matplotlib.pyplot as plt
rangeC = range(1,11)
wss = []
from sklearn.cluster import KMeans
for k in rangeC:
    clf = KMeans(k)
    clf.fit(X)
    wss.append(clf.inertia_)
    
plt.xlabel('# Clusters')
plt.ylabel('WSS')
plt.plot(rangeC, wss, marker = 'o')
plt.show()


# In[ ]:




