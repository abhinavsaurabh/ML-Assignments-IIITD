#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle
filename = 'GridSearchCv/1-3 PCA SVM rbf C=10.sav'
SVM = pickle.load(open(filename, 'rb'))


# In[3]:


CIFAR = 'cifar-10-batches-py/'
import numpy as np


# In[4]:


def unpickle(file):
    import pickle
    with open(file,'rb') as fo:
        cifar_dict = pickle.load(fo,encoding='bytes')
    return cifar_dict


# In[5]:


dirs = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']

full_data = [0,1,2,3,4,5,6]

for i, direc in zip(full_data,dirs):
    full_data[i] = unpickle(CIFAR+direc)


# In[6]:


batch_meta = full_data[0]
data_batch1 = full_data[1]
data_batch2 = full_data[2]
data_batch3 = full_data[3]
data_batch4 = full_data[4]
data_batch5 = full_data[5]
test_batch = full_data[6]


# In[7]:


train_all_data = [data_batch1,data_batch2,data_batch3,data_batch4,data_batch5]
test_batch1 = [test_batch]
#train_all_data = [data_batch1]
#train_all_data1 = train_all_data[:1000]
training_image = np.vstack([d[b"data"] for d in train_all_data])
train_len = len(training_image)
training_image = training_image.reshape(train_len,3*32*32)
training_label = np.hstack([d[b"labels"] for d in train_all_data])
training_label = training_label.reshape(50000)

#temp = training_image[:100]
#temp_2  = training_label[:100]


# In[8]:


test_image = np.vstack([d[b"data"] for d in test_batch1])
test_len = len(test_image)
test_image = test_image.reshape(test_len,3*32*32)
test_label = np.hstack([d[b"labels"] for d in test_batch1])


# In[ ]:


from sklearn.decomposition import PCA

pca = PCA(.90)
X_train = pca.fit_transform(training_image)


# In[10]:



filename = 'PCA Xtrain Xtest/XtrainPCA50000.sav'
X_train = pickle.load(open(filename, 'rb'))


# In[11]:


filename = 'PCA Xtrain Xtest/XtestPCA10000.sav'
X_test = pickle.load(open(filename, 'rb'))


# In[15]:


SVM.support_vectors_.shape


# In[10]:


newSVM = SVM.support_vectors_


# In[11]:


svm_ind = SVM.support_


# In[13]:


x_m2 = X_train[SVM.support_]
y_m2 = training_label[SVM.support_]


# In[14]:


y_m2.shape


# In[17]:


from sklearn.svm import SVC
classifier = SVC(kernel ='rbf', random_state = 0,C=10)
 # training set in x, y axis
classifier.fit(x_m2, y_m2)


# In[25]:


filename = '1-4 PCA classifier.sav'
pickle.dump(classifier, open(filename, 'wb'))


# In[18]:


classifier.support_vectors_.shape


# In[19]:


trainpredict = classifier.predict(x_m2)


# In[20]:


testpredict = classifier.predict(X_test)


# In[21]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_m2, trainpredict))


# In[22]:


print(classification_report(test_label, testpredict))


# In[ ]:
