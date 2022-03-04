#!/usr/bin/env python
# coding: utf-8

# In[13]:


class Regression(object):
    """docstring for Regression."""
    def __init__(self):
        super(Regression, self).__init__()
        #self.arg = arg


    """You can give any required inputs to the fit()"""
    def fit(self,X_train,y_train):

        """Here you can use the fit() from the LinearRegression of sklearn"""
        import numpy as np
        from sklearn.linear_model import LinearRegression
        clf=LinearRegression()
        clf.fit(X_train,y_train)
        self.coeff=clf.coef_
        self.intercept = clf.intercept_


    """ You can add as many methods according to your requirements, but training must be using fit(), and testing must be with predict()"""


    def predict(self,X_test):

        """ Write it from scratch usig oitcomes of fit()"""

        """Fill your code here. predict() should only take X_test and return predictions."""
        y_predicted =[]
        ini_ar = self.coeff
        #print(self.coeff)
        self.coeff = ini_ar.ravel() 
        print(X_test)
        for i in X_test:
            yi = 0
            #print(len(i))
            for k in range(len(i)):
                yi= yi + self.coeff[k]*i[k]
                #print(self.coeff[k],i[k])
            y_predicted.append(yi+self.intercept)    

        return y_predicted


# In[2]:


def fold_split(Z):
    f = Z.shape[0]//5
    fold1=Z[:f]
    fold2=Z[f:(2*f)]
    fold3=Z[(2*f):(3*f)]
    fold4=Z[(3*f):(4*f)]
    fold5=Z[(4*f):]
    return fold1,fold2,fold3,fold4,fold5


# In[3]:


def mse(y, y_pred): 
    import numpy as np
    y, y_pred = np.array(y), np.array(y_pred)
    return np.square(np.subtract(y,y_pred)).mean() 


# In[29]:


def main() :
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    import pandas as pd
    from pandas import DataFrame
    from sklearn import preprocessing 
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_csv( "Dataset.data",header=None,delimiter = r"\s+" )
    label_encoder = preprocessing.LabelEncoder() 
    df[0]= label_encoder.fit_transform(df[0])
    feature_cols = [0,1,2,3,4,5,6,7]
    target = [8]
    X = df[feature_cols]
    Y = df[target]
    Xfold1,Xfold2,Xfold3,Xfold4,Xfold5 = fold_split(X)
    Yfold1,Yfold2,Yfold3,Yfold4,Yfold5 = fold_split(Y)
    model = Regression()
    
    dfX1 = pd.concat([Xfold1,Xfold2,Xfold3,Xfold4], ignore_index=True)
    dfY1 = pd.concat([Yfold1,Yfold2,Yfold3,Yfold4], ignore_index=True)
    model.fit(dfX1.to_numpy(), dfY1.to_numpy())
    #print(dfX1)
    #print(dfX1.to_numpy())
    Y_train1 = model.predict(dfX1.to_numpy())
    Y_pred1 = model.predict(Xfold5.to_numpy())
    
    dfX2 = pd.concat([Xfold2,Xfold3,Xfold4,Xfold5], ignore_index=True)
    dfY2 = pd.concat([Yfold2,Yfold3,Yfold4,Yfold5], ignore_index=True)
    model.fit(dfX2.to_numpy(), dfY2.to_numpy()) 
    Y_train2 = model.predict(dfX2.to_numpy())
    Y_pred2 = model.predict(Xfold1.to_numpy())
    
    dfX3 = pd.concat([Xfold3,Xfold4,Xfold5,Xfold1], ignore_index=True)
    dfY3 = pd.concat([Yfold3,Yfold4,Yfold5,Yfold1], ignore_index=True)
    model.fit(dfX3.to_numpy(), dfY3.to_numpy()) 
    Y_train3 = model.predict(dfX3.to_numpy())
    Y_pred3 = model.predict(Xfold2.to_numpy())
    
    dfX4 = pd.concat([Xfold4,Xfold5,Xfold1,Xfold2], ignore_index=True)
    dfY4 = pd.concat([Yfold4,Yfold5,Yfold1,Yfold2], ignore_index=True)
    model.fit(dfX4.to_numpy(), dfY4.to_numpy()) 
    Y_train4 = model.predict(dfX4.to_numpy())
    Y_pred4 = model.predict(Xfold3.to_numpy())
    
    dfX5 = pd.concat([Xfold5,Xfold1,Xfold2,Xfold3], ignore_index=True)
    dfY5 = pd.concat([Yfold5,Yfold1,Yfold2,Yfold3], ignore_index=True)
    model.fit(dfX5.to_numpy(), dfY5.to_numpy()) 
    Y_train5 = model.predict(dfX5.to_numpy())
    Y_pred5 = model.predict(Xfold4.to_numpy())
    
    list1=[]
    list1.append((5,mse(dfY1,Y_train1),mse(Yfold5,Y_pred1)))
    list1.append((1,mse(dfY2,Y_train2),mse(Yfold1,Y_pred2)))
    list1.append((2,mse(dfY3,Y_train3),mse(Yfold2,Y_pred3)))
    list1.append((3,mse(dfY4,Y_train4),mse(Yfold3,Y_pred4)))
    list1.append((4,mse(dfY5,Y_train5),mse(Yfold4,Y_pred5)))
    df2 = DataFrame(list1,columns=['Fold','Train MSE','Validation MSE'])
    trainmean = np.mean(df2['Train MSE'])
    validmean = np.mean(df2['Validation MSE'])
    print("Training MSE Mean : %f"% trainmean)
    print("Validation MSE Mean:%f"% validmean)
    print(df2)
    
    


# In[30]:


if __name__ == "__main__" :    
    main()


# In[ ]:





# In[ ]:




