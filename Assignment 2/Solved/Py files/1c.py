#!/usr/bin/env python
# coding: utf-8

# In[2]:


from numpy.linalg import inv
def get_best_param(X, y):  
    X_transpose = X.T  
    best_params = inv(X_transpose.dot(X)).dot(X_transpose).dot(y)  
    
      
    return best_params 


# In[3]:


import pandas as pd
from sklearn.metrics import mean_squared_error
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


# In[4]:


def fold_split(Z):
    f = Z.shape[0]//5
    fold1=Z[:f]
    fold2=Z[f:(2*f)]
    fold3=Z[(2*f):(3*f)]
    fold4=Z[(3*f):(4*f)]
    fold5=Z[(4*f):]
    return fold1,fold2,fold3,fold4,fold5


# In[5]:


def mse(y, y_pred): 
    import numpy as np
    y, y_pred = np.array(y), np.array(y_pred)
    return np.square(np.subtract(y,y_pred)).mean()


# In[6]:


df = pd.read_csv( "Dataset.data",header=None,delimiter = r"\s+" )
label_encoder = preprocessing.LabelEncoder() 
df[0]= label_encoder.fit_transform(df[0])
feature_cols = [0,1,2,3,4,5,6,7]
target = [8]
X = df[feature_cols]
Y = df[target]
Xfold1,Xfold2,Xfold3,Xfold4,Xfold5 = fold_split(X)
Yfold1,Yfold2,Yfold3,Yfold4,Yfold5 = fold_split(Y)

dfX1 = pd.concat([Xfold1,Xfold2,Xfold3,Xfold4], ignore_index=True)
dfY1 = pd.concat([Yfold1,Yfold2,Yfold3,Yfold4], ignore_index=True)
params = get_best_param(dfX1, dfY1) 
Y_train1 = dfX1.dot(params)
Y_pred1 = Xfold5.dot(params)

dfX2 = pd.concat([Xfold2,Xfold3,Xfold4,Xfold5], ignore_index=True)
dfY2 = pd.concat([Yfold2,Yfold3,Yfold4,Yfold5], ignore_index=True)
params = get_best_param(dfX2, dfY2) 
Y_train2 = dfX2.dot(params)
Y_pred2 = Xfold1.dot(params)

dfX3 = pd.concat([Xfold3,Xfold4,Xfold5,Xfold1], ignore_index=True)
dfY3 = pd.concat([Yfold3,Yfold4,Yfold5,Yfold1], ignore_index=True)
params = get_best_param(dfX3, dfY3) 
Y_train3 = dfX3.dot(params)
Y_pred3 = Xfold2.dot(params)

dfX4 = pd.concat([Xfold4,Xfold5,Xfold1,Xfold2], ignore_index=True)
dfY4 = pd.concat([Yfold4,Yfold5,Yfold1,Yfold2], ignore_index=True)
params = get_best_param(dfX4, dfY4) 
Y_train4 = dfX4.dot(params)
Y_pred4 = Xfold3.dot(params)


dfX5 = pd.concat([Xfold5,Xfold1,Xfold2,Xfold3], ignore_index=True)
dfY5 = pd.concat([Yfold5,Yfold1,Yfold2,Yfold3], ignore_index=True)
params = get_best_param(dfX5, dfY5) 
Y_train5 = dfX5.dot(params)
Y_pred5 = Xfold4.dot(params)


list1=[]
    
list1.append((5,mse(dfY1,Y_train1),mse(Yfold5,Y_pred1)))
list1.append((1,mse(dfY2,Y_train2),mse(Yfold1,Y_pred2)))
list1.append((2,mse(dfY3,Y_train3),mse(Yfold2,Y_pred3)))
list1.append((3,mse(dfY4,Y_train4),mse(Yfold3,Y_pred4)))
list1.append((4,mse(dfY5,Y_train5),mse(Yfold4,Y_pred5)))
df2 = DataFrame(list1,columns=['Fold','Train MSE','Validation MSE'])
print(df2)


# In[7]:


list1


# In[ ]:




