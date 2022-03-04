#!/usr/bin/env python
# coding: utf-8

# In[ ]:


CIFAR = '/content/drive/My Drive/Assignment/ML ASS 3/cifar-10-batches-py/'
import numpy as np


# In[ ]:


def unpickle(file):
    import pickle
    with open(file,'rb') as fo:
        cifar_dict = pickle.load(fo,encoding='bytes')
    return cifar_dict


# In[ ]:


dirs = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']

full_data = [0,1,2,3,4,5,6]

for i, direc in zip(full_data,dirs):
    full_data[i] = unpickle(CIFAR+direc)


# In[ ]:


batch_meta = full_data[0]
data_batch1 = full_data[1]
data_batch2 = full_data[2]
data_batch3 = full_data[3]
data_batch4 = full_data[4]
data_batch5 = full_data[5]
test_batch = full_data[6]


# In[ ]:


train_all_data = [data_batch1,data_batch2,data_batch3,data_batch4,data_batch5]
test_batch1 = [test_batch]
#train_all_data = [data_batch1]
#train_all_data1 = train_all_data[:1000]
training_image = np.vstack([d[b"data"] for d in train_all_data])
train_len = len(training_image)
training_image = training_image.reshape(train_len,3*32*32)
training_label = np.hstack([d[b"labels"] for d in train_all_data])
training_label = training_label.reshape(50000)


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


test_image = np.vstack([d[b"data"] for d in test_batch1])
test_len = len(test_image)
test_image = test_image.reshape(test_len,3*32*32)
test_label = np.hstack([d[b"labels"] for d in test_batch1])
test_label = test_label.reshape(10000)


# In[ ]:


train_image_new = training_image[:1000]
test_image_new = test_image[:1000]
train_new_label = training_label[:1000]
test_new_label = test_label[:1000]


# In[ ]:


#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()

#X_train = sc.fit_transform(training_image)
#X_test = sc.transform(test_image)


# In[14]:


from sklearn.decomposition import PCA

pca = PCA(.90)
X_train = pca.fit_transform(training_image)
X_test = pca.transform(test_image)


# In[13]:


#import pickle
#filename = 'XtestPCA10000.sav'
#X_test = pickle.load(open(filename, 'rb'))

#filename = 'XtrainPCA50000.sav'
#X_train = pickle.load(open(filename, 'rb'))


# In[15]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle
# defining parameter range
param_grid = {'C': [10],
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3,cv=5,n_jobs=-1)

# fitting the model for grid search
grid.fit(X_train, training_label)


# In[16]:


print(grid.best_params_)


# In[17]:


print(grid.best_estimator_)


# In[22]:


sv = grid.best_estimator_.fit(X_train,training_label)


# In[23]:


sv.support_vectors_.shape


# In[24]:


filename = '1-3 PCA SVM rbf C=10.sav'
pickle.dump(sv, open(filename, 'wb'))


# In[25]:


X_train.shape


# In[18]:


grid_trainpredict = grid.predict(X_train)


# In[19]:


grid_predictions = grid.predict(X_test)


# In[20]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(training_label, grid_trainpredict))


# In[21]:


print(classification_report(test_label, grid_predictions))
