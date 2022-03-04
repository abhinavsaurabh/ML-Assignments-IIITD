import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import utils
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import SVM_class

def callSVM(X,y, C, plot, gamma, kernel, model_name = 'default_q2',useSavedModel=False):

    #Creating object
    wts = []

    if useSavedModel == True:
        print('Loaded saved model')
        svm = joblib.load(model_name)
        print('Loading completed')
    else:      
        svm = SVM_class.SVM()
  
    #Data Formation
    X_train, X_test, y_train, y_test = utils.local_train_test_split(X, y, test_size = 0.20)

    print('>>>>>>>>>>>>>>>', kernel)
    if useSavedModel == False:
        #Fit Model
        print('>>>>>>> Fitting Model >>>>>>>>')
        svm.fit(X = X_train, y = y_train, kernel=kernel, C=C, gamma=gamma)
    
    # #Predict Train Data
    y_predicted_train = svm.predict(X_train)
    train_acc = (svm.accuracy(y_predicted_train,y_train))
    #print('CMatrix Train: ',confusion_matrix(y_train,y_predicted_train)) #FOR BUILDING CONFIDENCE

    # #Predict Test Data
    y_predicted_test = svm.predict(X_test)
    test_acc = (svm.accuracy(y_predicted_test,y_test))
    #print('CMatrix Test: ',confusion_matrix(y_test,y_predicted_test)) #FOR BUILDING CONFIDENCE

    #Plot the Accuracy and Loss graphs
    if plot == 1:
        svm.plot_decisionBoundary(X_train,y_train)
   
    print('Training Accurcay is: ',train_acc)
    print('Testing Accurcay is: ',test_acc)

    if useSavedModel == False:
        # Dump the model
        joblib.dump(svm,model_name)

    return train_acc, test_acc

def callGridSearch(X,y,kernel,model_name = 'grid'):

    trainAcc_list = []
    testAcc_list = []
    C=0.01
    gamma=1
    for i in range(3):
        if kernel=='linear':
            model_name = 'grid_'+str(kernel)+'_'+str(C)
            print('*********************',model_name)
            trainAcc, testAcc= callSVM(X,y, C=C, plot=1, gamma = gamma, kernel = kernel, model_name = model_name, useSavedModel=False)
            C*=10
            trainAcc_list.append(trainAcc)
            testAcc_list.append(testAcc)

        elif kernel=='rbf':
            gamma=1
            for j in range(3):
                model_name = 'grid_'+str(kernel)+'_'+str(C)+'_'+str(gamma)
                print('*********************',model_name)
                trainAcc, testAcc= callSVM(X,y, C=C, plot=1, gamma = gamma, kernel = kernel, model_name = model_name, useSavedModel=False)
                gamma*=10
                trainAcc_list.append(trainAcc)
                testAcc_list.append(testAcc)
            C*=10

    print(trainAcc_list)    
    print(testAcc_list)

def comparisionWithSklearn(X,y,kernel='linear',C=1,gamma=1):
    svm =  SVC(kernel = kernel,gamma=gamma,C=C)
  
    #Data Formation
    X_train, X_test, y_train, y_test = utils.local_train_test_split(X, y, test_size = 0.20)
    
    #Fit Model
    svm.fit(X = X_train, y = np.ravel(y_train))

    #Predict Train Data 
    train_acc = svm.score(X_train,np.ravel(y_train))
    y_predicted_train = svm.predict(X_train)
    print(np.unique(y_predicted_train, return_counts=True))
    print('SKLEARN CMatrix Train: ',confusion_matrix(y_train,y_predicted_train)) #FOR BUILDING CONFIDENCE

    #Predict Test Data
    test_acc = svm.score(X_test,np.ravel(y_test))
    y_predicted_test = svm.predict(X_test)
    print(np.unique(y_predicted_test, return_counts=True))
    print('Sklearn CMatrix Test: ',confusion_matrix(y_test,y_predicted_test)) #FOR BUILDING CONFIDENCE

    print(f'Sklearn Training Accurcay is: {train_acc}')
    print(f'Sklearn Testing Accurcay is: {test_acc}')

if __name__== '__main__':
    mat1=scipy.io.loadmat('dataset_a.mat')
    X = mat1['samples']
    y = mat1['labels'].T

    ######### PART 1
    #utils.scatter_plot(X,y)

    ######### PART 2
    #callGridSearch(X,y,kernel='linear')

    ######### PART 3
    #callGridSearch(X,y,kernel='rbf')
    callSVM(X,y, C=1, plot=0, gamma=1, kernel='rbf', model_name = 'default_q2',useSavedModel=False)

    ########## PART 4
    #comparisionWithSklearn(X,y,kernel='linear',C=1)
    comparisionWithSklearn(X,y,kernel='rbf',gamma=1,C=1)