#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('iris.data')


# In[9]:


feature_cols = ['sepal length', 'sepal width', 'petal length', 'petal width']
target = ['class']
X = df[feature_cols]
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y = le.fit_transform(df['class'])


# In[14]:


df['class'] = le.fit_transform(df['class'])


# In[15]:


df.corr()


# In[7]:


y


# In[10]:


X


# In[12]:


#from sklearn.cluster import KMeans
#clf =  KMeans(n_clusters=3,max_iter = 300, n_init = 10, random_state = 0)
#clf.fit(X_train,y_train)

X = X.to_numpy()


# In[17]:


import matplotlib.pyplot as plt

plt.scatter(X[y == 0,2], X[y == 0,3],s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X[y == 1, 2], X[y == 1, 3],s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[y == 2, 2], X[y == 2, 3],s = 100, c = 'green', label = 'Iris-virginica')
plt.legend()
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
#add x-axis and y-axis name
#pca graph for 2-dimension


# In[25]:


import seaborn as sns
sns.pairplot(df, hue="class", size=3, diag_kind="kde")

