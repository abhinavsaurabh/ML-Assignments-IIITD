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
        ini_array1 = self.coeff
        #print(self.coeff)
        self.coeff = ini_array1.ravel() 
        print(X_test)
        for i in X_test:
            yi = 0
            #print(len(i))
            for k in range(len(i)):
                yi= yi + self.coeff[k]*i[k]
                #print(self.coeff[k],i[k])
            y_predicted.append(yi+self.intercept)    

        return y_predicted