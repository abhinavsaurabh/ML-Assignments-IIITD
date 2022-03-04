#!/usr/bin/env python
# coding: utf-8

# In[20]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from pandas import DataFrame


data1 = loadmat('dataset_2.mat')
data2 = data1['samples']
mlabel = data1['labels'][0]
data3 = np.column_stack((data2,mlabel))
df = pd.DataFrame(data3)


# In[21]:


df


# In[22]:


feature_cols = [0,1]
X = df[feature_cols] # Features
feature1 =[2]
y = df[feature1] # Target variable


# In[23]:


y


# In[24]:


def split(df):

# Shuffle your dataset 
    shuffle_df = df.sample(frac=1)

# Define a size for your train set 
    train_size = int(0.7 * len(df))

# Split your dataset 
    train_set = shuffle_df[:train_size]
    test_set = shuffle_df[train_size:]
    
    return train_set,test_set  


# In[25]:


train_set, test_set = split(df)


# In[26]:


feature_cols = [0,1]
X_train = train_set[feature_cols] # Features
X_test = test_set[feature_cols]

feature1 =[2]
Y_train = train_set[feature1] # Target variable
Y_test = test_set[feature1]


# In[27]:


list1=[]
for i in range(15):
    clf = DecisionTreeClassifier(max_depth=i+1)
    clf = clf.fit(X_train,Y_train)
    y_pred = clf.predict(X_test)
    yhat=np.array(y_pred)
    y=np.array(Y_test)
    size=yhat.size
    count=0
    for j in range(yhat.size):
        if yhat[j]==y[j]:
            count+=1;
    #print( acc)
    list1.append([i+1,count/yhat.size])
    #list1.append([i+1,metrics.accuracy_score(Y_test, y_pred)])
    #print("Accuracy:",i+1,metrics.accuracy_score(Y_test, y_pred))


# In[28]:


list1


# In[29]:


df2 = DataFrame (list1,columns=['Depth','Test Accuracy'])


# In[30]:


df2


# In[31]:


plt.plot(df2['Depth'], df2['Test Accuracy'])
plt.xlabel('Depth')
plt.ylabel('Test Accuracy')


# In[ ]:




