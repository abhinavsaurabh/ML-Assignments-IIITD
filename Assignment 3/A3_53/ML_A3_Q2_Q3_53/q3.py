import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import utils
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import SVM_class

def callMultiClassSVM(X,y, C, kernel, gamma, type, model_name = 'default_q3', useSavedModel=False):
    folds = 5

    #Getting the training and validation indices
    indices = utils.cross_fold(X,folds)

    #Creating lists
    train_acc = []
    test_acc = []
    class_train_acc = []
    class_test_acc = []
    wts = []
    

    if useSavedModel == True:
        models = joblib.load(model_name)
    else:
        if type == 'OVO':
            models = [SVM_class.ovo() for i in range(folds)]
        elif type == 'OVR':
            models = [SVM_class.ovr() for i in range(folds)]


    for i in range(folds):
        print('>>> FOLD # and Type is ',i,type)
        svm = models[i]

        #Data Formation
        train_ind = indices[i][0]
        test_ind = indices[i][1]

        if useSavedModel==False:
            #Fit Model
            svm.fit(X = X[train_ind], y = y[train_ind], kernel=kernel, C=C, gamma=gamma)

        #Predict Train Data 
        print('TTRRAAIINN')
        y_predicted_train = svm.predict(X[train_ind])
        train_acc.append(svm.accuracyy(y_predicted_train,y[train_ind]))
        class_train_acc.append(svm.classwiseAccuracy(np.ravel(y_predicted_train),np.ravel(y[train_ind])))

        #Predict Test Data
        print('TEEESSSSTTT')
        y_predicted_test = svm.predict(X[test_ind])
        test_acc.append(svm.accuracyy(y_predicted_test,y[test_ind]))
        class_test_acc.append(svm.classwiseAccuracy(np.ravel(y_predicted_test),np.ravel(y[test_ind])))

    print(f'Training Accurcay per fold is: {train_acc}')
    print(f'Testing Accurcay per fold is: {test_acc}')
    print(f'Training Accurcay per class per fold is: {class_train_acc}')
    print(f'Testing Accurcay per class per fold is: {class_test_acc}')
    print(f'Mean Training Accuracy : {sum(train_acc)/len(train_acc):.4f}')
    print(f'Mean Testing Accuracy : {sum(test_acc)/len(test_acc):.4f}')
    print(f'Mean Training Accurcay per class per fold is: {np.mean(np.array(class_train_acc),axis=0)}')
    print(f'Mean Testing Accurcay per class per fold is: {np.mean(np.array(class_test_acc),axis=0)}')

    if useSavedModel==False:
        # Dump the model
        joblib.dump(models,model_name)

def callGridSearch(X,y,kernel,type,model_name = 'grid'):

    C=0.01
    gamma=1
    for i in range(3):
        if kernel=='linear':
            model_name = 'grid_'+'_'+str(type)+str(kernel)+'_'+str(C)
            print('*********************',model_name)
            callMultiClassSVM(X,y, C=C, gamma = gamma, kernel = kernel, type = type, model_name = model_name, useSavedModel=False)
            C*=10

        elif kernel=='rbf':
            gamma=1
            for j in range(3):
                model_name = 'grid_'+'_'+str(type)+'_'+str(kernel)+'_'+str(C)+'_'+str(gamma)
                print('*********************',model_name)
                callMultiClassSVM(X,y, C=C, gamma = gamma, kernel = kernel, type = type, model_name = model_name, useSavedModel=True)
                gamma*=10
            C*=10


def comparisionWithSklearn(X,y,type,kernel='linear',C=1,gamma=1):
    model = SVC(random_state=0,kernel = kernel,gamma=gamma,C=C)
    folds = 5

    #Getting the training and validation indices
    indices = utils.cross_fold(X,folds)
    if type == 'OVO':
        svm = OneVsOneClassifier(model)
    elif type == 'OVR':
        svm = OneVsRestClassifier(model) 

    #Creating lists for statistics
    train_acc = []
    test_acc = []
    class_train_acc = []
    class_test_acc = []
    wts = []

    for i in range(folds):
        print('>>> FOLD #',i)
        #Data Formation
        train_ind = indices[i][0]
        test_ind = indices[i][1]

        #Fit Model
        svm.fit(X = X[train_ind], y = np.ravel(y[train_ind]))

        #Predict Train Data 
        y_train_pred = svm.predict(X[train_ind])
        print('SKLEARN TRAIN',np.unique(y_train_pred,return_counts=True))
        train_acc.append(svm.score(X[train_ind],np.ravel(y[train_ind])))
        matrix = confusion_matrix(np.ravel(y[train_ind]), y_train_pred)
        class_train_acc.append(matrix.diagonal()/matrix.sum(axis=1))


        #Predict Test Data
        y_test_pred = svm.predict(X[test_ind])
        print('SKLEARN TEST',np.unique(y_test_pred,return_counts=True))
        test_acc.append(svm.score(X[test_ind],np.ravel(y[test_ind])))
        matrix = confusion_matrix(np.ravel(y[test_ind]), y_test_pred)
        class_test_acc.append(matrix.diagonal()/matrix.sum(axis=1))


    print(f'Training Accurcay per fold is: {train_acc}')
    print(f'Testing Accurcay per fold is: {test_acc}')
    print(f'Mean Training Accuracy : {sum(train_acc)/len(train_acc):.4f}')
    print(f'Mean Testing Accuracy : {sum(test_acc)/len(test_acc):.4f}')
    print(f'Training Accurcay per class per fold is: {class_train_acc}')
    print(f'Testing Accurcay per class per fold is: {class_test_acc}')
    print(f'Mean Training Accurcay per class per fold is: {np.mean(np.array(class_train_acc),axis=0)}')
    print(f'Mean Testing Accurcay per class per fold is: {np.mean(np.array(class_test_acc),axis=0)}')
    
if __name__== '__main__':
    mat2=scipy.io.loadmat('dataset_b.mat')
    X = mat2['samples']
    y = mat2['labels'].T

    ######### PART 1
    #utils.scatter_plot(X,y)

    ######### PART 2 OVO
    
    #callGridSearch(X,y,kernel='rbf',type='OVO',model_name = 'grid')
    callMultiClassSVM(X,y,type='OVO', kernel='rbf', C=1, gamma=1, model_name = 'q3_b', useSavedModel=False)
    
    ######### PART 3 OVR
    #callGridSearch(X,y,kernel='rbf',type='OVR',model_name = 'grid')
    callMultiClassSVM(X,y,type='OVR', kernel='rbf', C=1, gamma=1, model_name = 'q3_c', useSavedModel=False)

    ######### PART 4 Sklearn Comparision
    comparisionWithSklearn(X,y,type='OVO',kernel='rbf',C=1,gamma=1)
    comparisionWithSklearn(X,y,type='OVR',kernel='rbf',C=1,gamma=1)