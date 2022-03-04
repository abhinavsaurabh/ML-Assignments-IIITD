#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('iris.data')


# In[3]:


feature_cols = ['sepal length', 'sepal width', 'petal length', 'petal width']
target = ['class']
X = df[feature_cols]
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y = le.fit_transform(df['class'])


# In[4]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_p = pca.fit_transform(X)


# In[6]:


import matplotlib.pyplot as plt
plt.scatter(X_p[y == 0,0], X_p[y == 0,1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X_p[y == 1, 0], X_p[y == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X_p[y == 2, 0], X_p[y == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')
plt.legend()


# In[10]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
scaled_data = scaler.transform(X)


# In[11]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(scaled_data)

X_p = pca.transform(scaled_data)


# In[12]:


import matplotlib.pyplot as plt
plt.scatter(X_p[y == 0,0], X_p[y == 0,1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X_p[y == 1, 0], X_p[y == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X_p[y == 2, 0], X_p[y == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')
plt.legend()


# In[ ]:




