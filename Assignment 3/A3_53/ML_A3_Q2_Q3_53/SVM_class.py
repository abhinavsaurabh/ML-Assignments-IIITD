import numpy as np
from itertools import combinations 
from scipy.stats import mode 
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from collections import defaultdict

def linear(x, sv, gamma=0): #gamma is dummy just to same number of params as gaussian
    return [np.dot(vi, x) for vi in sv]

def rbf(x, sv, gamma=0.1):
    return [np.exp(-gamma* np.dot(vi - x, vi - x)) for vi in sv]

class SVM(object):
    def __init__(self):
        pass

    def fit(self, X, y, kernel, C=1, gamma=1):
        if kernel == 'linear':
            self.kernel = linear
        elif kernel == 'rbf':
            self.kernel = rbf
        
        self.gamma = gamma

        svm =  SVC(kernel = kernel,gamma=gamma,C=C)
        svm.fit(X, np.ravel(y))

        self.clf = svm
        self.dual_coef = svm.dual_coef_
        self.svs = svm.support_vectors_
        self.intercept = svm.intercept_
        self.decison_function = svm.decision_function
        return self.dual_coef, self.svs, self.intercept, self.decison_function

    def predict(self, X_test):

        y_predict = np.zeros((X_test.shape[0]))
        for i  in range(X_test.shape[0]):
            y_predict[i] = np.sum(self.dual_coef * self.kernel(X_test[i], self.svs, self.gamma))
        y_predicted = np.sign(y_predict + self.intercept)

        # Approach 2
        #y_predicted = np.sign(self.decison_function(X_test))

        y_predicted = np.where((y_predicted==-1),0,1)
        print(np.unique(y_predicted, return_counts=True))
        return y_predicted

    def plot_decisionBoundary(self,X,y):
        plt.scatter(X[:, 0], X[:, 1], c=np.ravel(y), s=30, cmap='autumn')

        # plot the decision function
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.clf.decision_function(xy).reshape(XX.shape)

        # plot decision boundary and margins
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
        
        # plot support vectors
        x_support = X[self.clf.support_]
        y_support = y[self.clf.support_]

        x1 = []
        x2 = []

        for ind in range(len(y_support)):
            if y_support[ind] == self.clf.classes_[0]:
                x1.append(x_support[ind])
            else:
                x2.append(x_support[ind])

        x1 = np.array(x1)
        x2 = np.array(x2)
        ax.scatter(x1[:, 0], x1[:, 1], s=100, linewidth=1, facecolors='#9400D3', edgecolors='k', label='SV C1')
        ax.scatter(x2[:, 0], x2[:, 1], s=100,linewidth=1, facecolors='#9400D3', edgecolors='k', marker='*', label='SV C2')
        ax.legend(loc='upper right', shadow=True, fontsize='x-large')

        plt.show()

    def accuracy(self,y_predicted,y):
        # print('$$$$$$$$ predictd',np.unique(y_predicted, return_counts=True))
        # print('$$$$$$$$ truth',np.unique(y, return_counts=True))
        # print(y.size, y.shape[0])
        # print(y_predicted.size, y_predicted.shape[0])
        return np.sum(y[:,0]==y_predicted)/y.size

    def classwiseAccuracy(self,y_predict,y):
        print(type(y_predict),type(y),y_predict.shape,y.shape)
        unique1, counts1 = np.unique(y_predict, return_counts=True)
        unique2, counts2 = np.unique(y, return_counts=True)

        d_pred = dict(zip(unique1, counts1))
        d_truth = dict(zip(unique2, counts2))
     
        d_truth_keys = [k for k in d_truth.keys()]
        d_pred_keys = [k for k in d_pred.keys()]
        values = set(d_truth_keys +  d_pred_keys)
        indices = dict((val, i) for (i, val) in enumerate(values))

        #create confusion matrix
        c_matrix = np.zeros((len(values),len(values)))
        for w, g in zip(y, y_predict):
            if (w in d_truth_keys and g in d_pred_keys):
                c_matrix[indices[w]][indices[g]] += 1

        cls_acc=[]
        cls_acc = c_matrix.diagonal()/c_matrix.sum(axis=1)        
        return cls_acc

class ovr(SVM):
    def __init__(self):
        super().__init__()
        self.classes = 0

    def fit(self, X, y, kernel, C=1, gamma=1):
        if kernel == 'linear':
            self.kernel = linear
        elif kernel == 'rbf':
            self.kernel = rbf

        self.gamma = gamma
        self.C= C

        print(f'OVR G = {self.gamma}, C = {C}, kernel = {kernel}')
    
        self.classes = np.unique(y)        
        
        self.ovr_dual_coef = []
        self.ovr_svs = []
        self.ovr_intercept = []
        self.ovr_decision_function = []
        
        for cl in self.classes:
            print(f'>>> Training for the class and not of this class: {cl}')
            y_new = np.where((y==cl),1,0) #question 2 also has classes as 0 and 1, -1 is not needed in training
            
            dc, sv, inter, df = super().fit(X, y_new, kernel=kernel, gamma=self.gamma, C=self.C)
            self.ovr_dual_coef.append(dc)
            self.ovr_svs.append(sv)
            self.ovr_intercept.append(inter)
            self.ovr_decision_function.append(df)           
            
    def predict(self, X_test):        
        # Calculate probabilties
        self.pred_test = np.zeros((len(self.classes),len(X_test)))
        for cl in range(len(self.classes)):
            pred = self.ovr_decision_function[cl](X_test)
            #pred = np.abs(pred)
            self.pred_test[cl,:] = pred
           
        y_predict_labels = np.argmax(self.pred_test, axis=0)
        #print('LOCALLLLL LABLES:',np.unique(y_predict_labels,return_counts=True))
        #probs = np.max(self.pred_test, axis=0)
        
        return np.reshape(y_predict_labels,(len(y_predict_labels),1))

    def accuracyy(self,y_predicted,y):
        # print('$$$$$$$$ predictd',np.unique(y_predicted, return_counts=True))
        # print('$$$$$$$$ truth',np.unique(y, return_counts=True))
        # print(y.size, y.shape[0])
        # print(y_predicted.size, y_predicted.shape[0])
        return np.sum(y==y_predicted)/y.size

    
class ovo(SVM):
    def __init__(self):
        super().__init__()
        self.classes = 0
    
    def fit(self, X, y, kernel, C=1, gamma=1):
        if kernel == 'linear':
            self.kernel = linear
        elif kernel == 'rbf':
            self.kernel = rbf

        self.gamma = gamma
        self.C= C

        print(f'OVO G = {self.gamma}, C = {C}, kernel = {kernel}')
    
        self.classes = np.unique(y)        
        
        self.ovo_dual_coef = []
        self.ovo_svs = []
        self.ovo_intercept = []
        self.ovo_decision_function = []
        comb = combinations(self.classes, 2)
        
        for cm in comb:
            print(f'>>> Training for the combination of classes : {cm[0]} and {cm[1]}')
            y_new = np.copy(y)
            y_new = np.where((y_new==cm[0]),-1,y_new)
            y_new = np.where((y_new==cm[1]),-2,y_new)
            y_new = np.where(((y_new==-1) | (y_new==-2)),y_new,-3)
            y_new = np.where((y_new==-1),0,y_new)
            y_new = np.where((y_new==-2),1,y_new)
            indices = np.where((y_new==0) | (y_new==1))
            
            X_new = X[indices[0]]
            y_new = y_new[indices[0]]

            dc, sv, inter, df = super().fit(X_new, y_new, kernel=kernel, gamma=self.gamma, C=self.C)
            self.ovo_dual_coef.append(dc)
            self.ovo_svs.append(sv)
            self.ovo_intercept.append(inter)
            self.ovo_decision_function.append(df)           
            
    def predict(self, X_test):        
        # Calculate probabilties
        comb = combinations(self.classes, 2)
        n_comb = len(list(comb))
        
        self.pred_test = np.zeros((n_comb,len(X_test)))

        comb = combinations(self.classes, 2)
        for i,cm in enumerate(comb):                     
            pred = np.sign(self.ovo_decision_function[i](X_test))
            self.pred_test[i,:] = np.where((pred==1), cm[1], cm[0])
           
        # Select max probability
        val, count = mode(self.pred_test, axis = 0) 
        y_predict_labels = val.ravel().tolist()
        print('LOCAL LABLES:',np.unique(y_predict_labels,return_counts=True)) 
        return np.reshape(y_predict_labels,(len(y_predict_labels),1))

    def accuracyy(self,y_predicted,y):
        # print('$$$$$$$$ predictd',np.unique(y_predicted, return_counts=True))
        # print('$$$$$$$$ truth',np.unique(y, return_counts=True))
        # print(y.size, y.shape[0])
        # print(y_predicted.size, y_predicted.shape[0])
        return np.sum(y==y_predicted)/y.size
