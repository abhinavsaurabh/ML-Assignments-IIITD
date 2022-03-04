#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
df = pd.read_csv('sat.trn', sep='\s+',header=None)
feature_cols = list(df.columns)
feature_cols = feature_cols[0:36]
target = [36]
X_train = df[feature_cols]
y_train = df[target]


# In[39]:


df2 = pd.read_csv('sat.tst', sep='\s+',header=None)
feature_cols2 = list(df2.columns)
feature_cols2 = feature_cols2[0:36]
target2 = [36]
X_test = df2[feature_cols2]
y_test= df2[target2]


# In[53]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 5)
clf.fit(X_train, y_train)


# In[54]:


trainpred = clf.predict(X_train)
from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y_train, trainpred))


# In[55]:


predictions = clf.predict(X_test)
from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y_test, predictions))


# In[6]:


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


# In[50]:


clf1  = KNN( K = 5)
clf1.fit(X_train.values, y_train.values)


# In[51]:


trainpred = clf1.predict(X_train.values)
from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y_train, trainpred))


# In[52]:


predictions = clf1.predict(X_test.values)
from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y_test, predictions))


# In[ ]:
