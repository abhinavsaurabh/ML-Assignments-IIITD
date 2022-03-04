import scipy.io
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations 
from scipy.stats import mode 
import joblib

def scatter_plot(X,y):
    mat_concat = np.append(X,y,axis=1)
    df = pd.DataFrame(mat_concat, columns=['X1','X2','Y'])
    df.plot.scatter(x='X1',y='X2',c='Y',colormap='viridis')
    plt.show()

def cross_fold(X,folds = 5):
    np.random.seed(24)
    perm = np.random.permutation(X.shape[0])
    #perm = np.arange(X.shape[0])

    fold_size = int(X.shape[0] // folds)

    indices = {}
    for i in range(folds): 
        test_start = i*fold_size
        test_end = test_start + fold_size - 1
    
        test_ind = perm[test_start : test_end]
        train_ind = np.delete(perm,range(test_start,test_end+1))
    
        indices[i] = (train_ind,test_ind)

    return indices

def local_train_test_split(X, y, test_size = 0.25):
        np.random.seed(42)
        perm = np.random.permutation(len(X))
        train_end = int((1-test_size) * len(X))
        X_trainset = X[perm[:train_end]]
        X_testset = X[perm[train_end:]]
        y_trainset = y[perm[:train_end]]
        y_testset = y[perm[train_end:]]
        return X_trainset, X_testset, y_trainset, y_testset
