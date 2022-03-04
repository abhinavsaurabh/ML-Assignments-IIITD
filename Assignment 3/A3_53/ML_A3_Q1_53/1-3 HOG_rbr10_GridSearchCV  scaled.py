#!/usr/bin/env python
# coding: utf-8

# In[2]:


CIFAR = '/content/drive/My Drive/ML ASS 3/cifar-10-batches-py/'
import numpy as np


# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[3]:


def unpickle(file):
    import pickle
    with open(file,'rb') as fo:
        cifar_dict = pickle.load(fo,encoding='bytes')
    return cifar_dict


# In[4]:


dirs = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']

full_data = [0,1,2,3,4,5,6]

for i, direc in zip(full_data,dirs):
    full_data[i] = unpickle(CIFAR+direc)


# In[5]:


batch_meta = full_data[0]
data_batch1 = full_data[1]
data_batch2 = full_data[2]
data_batch3 = full_data[3]
data_batch4 = full_data[4]
data_batch5 = full_data[5]
test_batch = full_data[6]


# In[6]:


train_all_data = [data_batch1,data_batch2,data_batch3,data_batch4,data_batch5]
test_batch1 = [test_batch]

training_image = np.vstack([d[b"data"] for d in train_all_data])
train_len = len(training_image)
training_image = training_image.reshape(train_len,3*32*32)
training_label = np.hstack([d[b"labels"] for d in train_all_data])


# In[7]:


test_image = np.vstack([d[b"data"] for d in test_batch1])
test_len = len(test_image)
test_image = test_image.reshape(test_len,3*32*32)
test_label = np.hstack([d[b"labels"] for d in test_batch1])


# In[8]:


##### RAJATA CODE for hog + color hist
import numpy as np
import matplotlib.colors as colors
from skimage.feature import hog
from skimage import color

X_train_hog = []

for img in training_image:

    img = np.reshape(img.T,(32,32,3))
    # color histogram
    array=np.asarray(img)
    arr=(array.astype(float))/255.0
    img_hsv = colors.rgb_to_hsv(arr[...,:3])
    img_color_hist = np.histogram(img_hsv[...,0],bins=10)
    hist_values = img_color_hist[0]

    #hog
    gray_image = color.rgb2gray(img)
    hog_feature = hog(gray_image,orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1))

    #joined
    X_train_hog.append(np.concatenate((hist_values,hog_feature),axis=None))


# In[9]:


X_test_hog = []

for img in test_image:

    img = np.reshape(img.T,(32,32,3))
    # color histogram
    array=np.asarray(img)
    arr=(array.astype(float))/255.0
    img_hsv = colors.rgb_to_hsv(arr[...,:3])
    img_color_hist = np.histogram(img_hsv[...,0],bins=10)
    hist_values = img_color_hist[0]

    #hog
    gray_image = color.rgb2gray(img)
    hog_feature = hog(gray_image,orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1))

    #joined
    X_test_hog.append(np.concatenate((hist_values,hog_feature),axis=None))


# In[10]:


import numpy
X_train =numpy.asarray(X_train_hog)
X_test = numpy.asarray(X_test_hog)


# In[12]:


from sklearn import preprocessing
X_train = preprocessing.scale(X_train)


# In[23]:


X_test = preprocessing.scale(X_test)


# In[26]:


X_train.shape


# In[14]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle
# defining parameter range
param_grid = {'C': [10],
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3,cv=5,n_jobs=-1)

# fitting the model for grid search
grid.fit(X_train, training_label)


# In[15]:


print(grid.best_params_)


# In[16]:


print(grid.best_estimator_)


# In[30]:


sv = grid.best_estimator_.fit(X_train,training_label)


# In[31]:


filename = '1-3 HOG SVM scaled rbf C=10.sav'
pickle.dump(sv, open(filename, 'wb'))


# In[17]:


grid_trainpredict = grid.predict(X_train)


# In[28]:


grid_predictions = grid.predict(X_test)


# In[18]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(training_label, grid_trainpredict))


# In[32]:


print(classification_report(test_label, grid_predictions))
