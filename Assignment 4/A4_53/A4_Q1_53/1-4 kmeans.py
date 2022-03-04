#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn import metrics


# In[3]:


df = pd.read_csv('iris.data')


# In[ ]:


#'sepal length', 'sepal width'


# In[4]:


feature_cols = ['petal length', 'petal width']
target = ['class']
X = df[feature_cols]
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y = le.fit_transform(df['class'])


# In[28]:


df.shape


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state = 0)


# In[20]:


from sklearn.cluster import KMeans
clf =  KMeans(n_clusters=3,random_state=5,max_iter=250,n_init=10)
clf.fit(X_train)


# In[174]:


import pickle
filename = 'kmeans.sav'
pickle.dump(clf, open(filename, 'wb'))


# In[20]:


import pickle
filename = 'kmeans.sav'
clf = pickle.load(open(filename, 'rb'))


# In[21]:


y_trainpred = clf.predict(X_train)


# In[22]:


print(metrics.accuracy_score(y_trainpred, y_train))


# In[9]:


y_testpred = clf.predict(X_test)


# In[10]:


from sklearn import metrics
print(metrics.accuracy_score(y_testpred, y_test))


# In[ ]:




