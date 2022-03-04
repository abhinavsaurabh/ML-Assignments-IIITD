class LogRegression(object):
    """docstring for LogRegression."""
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    """You can give any required inputs to the fit()"""
    def fit(self, X, y):
        

        """Write it from scratch. Usage of sklearn is not allowed"""
        n_samp, n_feat = X.shape

        # init parameters
        self.weights = np.zeros(n_feat)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            
            linear_model = np.dot(X, self.weights) + self.bias
          
            y_predicted = self._sigmoid(linear_model)

         
            dw = (1 / n_samp) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samp) * np.sum(y_predicted - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    """ You can add as many methods according to your requirements, but training must be using fit(), and testing must be with predict()"""
    def predict(self, X):
               
        """Write it from scratch. Usage of sklearn is not allowed"""

        """Fill your code here. predict() should only take X_test and return predictions."""
        linear_model = np.dot(X, self.weights) + self.bias
        self.ypred = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in self.ypred]
        return np.array(y_predicted_cls)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def log_loss(self):
        for i in range(n):
            self.loss = np.append(self.loss, [-y[i]*np.log(self.ypred[i])-(1-y[i])*np.log(1-self.ypred[i])])
        print(self.loss)
        log_loss = np.mean(self.loss)
        return log_loss    
        
        

       