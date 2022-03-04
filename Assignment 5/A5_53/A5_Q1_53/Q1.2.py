#!/usr/bin/env python
# coding: utf-8

# In[138]:


import pandas as pd

import numpy as np
from scipy.stats import mode

class KNN() :
    #error_rate.append(np.mean(pred_i != y_test))
    #error_rate.append(np.sum(y_test != pred_i) / len(y_test))
    #print(np.unique(pred_i, return_counts=True))
    #print(np.unique(y_test, return_counts=True))
    def __init__( self, K=1 ) :

        self.K = K

    def euclid( self, x, x_train ) :

        return np.sqrt( np.sum( np.square( x - x_train ) ) )

    def fit( self, x_train, y_train ) :

        self.m, self.n = x_train.shape

        self.x_train = x_train

        self.y_train = y_train


    def predict( self, x_test ) :

        self.x_test = x_test

        self.m_test, self.n = x_test.shape

        y_pred = np.zeros( self.m_test )

        for i in range( self.m_test ) :

            x = self.x_test[i]

            neigbr = np.zeros( self.K )

            neigbr = self.fdneigbr( x )

            y_pred[i] = mode( neigbr )[0][0]

        return y_pred


    def fdneigbr( self, x ) :


        euc_dists = np.zeros( self.m )

        for i in range( self.m ) :

            a = self.euclid( x, self.x_train[i] )

            euc_dists[i] = a

        inds = euc_dists.argsort()

        y_train_srtd = self.y_train[inds]

        return y_train_srtd[:self.K]


# In[139]:


import pandas as pd
df = pd.read_csv('sat.trn', sep='\s+',header=None)
feature_cols = list(df.columns)
feature_cols = feature_cols[0:36]
target = [36]
x_train = df[feature_cols]
y_train = df[target]


# In[140]:


df2 = pd.read_csv('sat.tst', sep='\s+',header=None)
feature_cols2 = list(df2.columns)
feature_cols2 = feature_cols2[0:36]
target2 = [36]
x_test = df2[feature_cols2]
y_test= df2[target2]


# In[115]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


range = list(range(1,8))
weight_c = ["uniform","distance"]
#metric = ["Euclidean"]
param_grid = dict(n_neighbors = range, weights = weight_c)
knn = KNeighborsClassifier()

grid = GridSearchCV(knn, param_grid,cv=10)
grid.fit(x_train,y_train)


# In[61]:


gs.get_params().keys()


# In[116]:


grid.best_score_


# In[117]:


grid.best_estimator_


# In[118]:


grid.best_params_


# In[67]:


clf  = KNN( K = 4)
clf.fit(x_train.values, y_train.values)


# In[84]:


trainpred = clf.predict(x_train.values)
from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y_train, trainpred))


# In[68]:


predictions = clf.predict(x_test.values)


# In[69]:


from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y_test, predictions))


# In[166]:


y_test=y_test.reshape(2000,)


# In[171]:


len(x_test)


# In[177]:


accuracy_list = []
error_rate = []
error_rate_mse = []
for i in range(1,8):

    knn = KNN(K=i)
    knn.fit(x_train.values,y_train.values)
    pred_i = knn.predict(x_test.values)
    #error_rate.append(np.mean(pred_i != y_test))
    #error_rate.append(np.sum(y_test != pred_i) / len(y_test))
    #print(np.unique(pred_i, return_counts=True))
    #print(np.unique(y_test, return_counts=True))
    error_rate_mse.append(np.square(np.subtract(y_test, pred_i)).mean())
    error_rate.append(1-(np.sum(y_test == pred_i)/len(y_test)))
    accuracy_list.append(np.sum(y_test == pred_i)/len(y_test))


# In[173]:


error_rate


# In[181]:


accuracy_list


# In[178]:


error_rate_mse


# In[184]:


from matplotlib import pyplot as plt
plt.figure(figsize=(11,7))
plt.plot(range(1,8),error_rate_mse,color='blue', linestyle='dashed', marker='o',
 markerfacecolor='red', markersize=10)
plt.title('Error MSE Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[186]:


from matplotlib import pyplot as plt
plt.figure(figsize=(11,7))
plt.plot(range(1,8),accuracy_list,color='blue', linestyle='dashed', marker='o',
 markerfacecolor='red', markersize=10)
plt.title('Accuracy List vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')


# In[ ]:
