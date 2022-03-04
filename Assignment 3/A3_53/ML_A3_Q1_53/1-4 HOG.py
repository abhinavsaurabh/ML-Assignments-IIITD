#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
filename = 'GridSearchCv/1-3 HOG SVM scaled rbf C=10.sav'
SVM = pickle.load(open(filename, 'rb'))


# In[2]:


CIFAR = 'cifar-10-batches-py/'
import numpy as np


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
#train_all_data = [data_batch1]
#train_all_data1 = train_all_data[:1000]
training_image = np.vstack([d[b"data"] for d in train_all_data])
train_len = len(training_image)
training_image = training_image.reshape(train_len,3*32*32)
training_label = np.hstack([d[b"labels"] for d in train_all_data])
training_label = training_label.reshape(50000)

#temp = training_image[:100]
#temp_2  = training_label[:100]


# In[7]:


test_image = np.vstack([d[b"data"] for d in test_batch1])
test_len = len(test_image)
test_image = test_image.reshape(test_len,3*32*32)
test_label = np.hstack([d[b"labels"] for d in test_batch1])


# In[10]:


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


# In[12]:


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


# In[13]:


import numpy
X_train =numpy.asarray(X_train_hog)
X_test = numpy.asarray(X_test_hog)


# In[14]:


filename = 'XTrain_HOG 50000.sav'
pickle.dump(X_train, open(filename, 'wb'))


# In[15]:


filename = 'XTest_HOG 10000.sav'
pickle.dump(X_test, open(filename, 'wb'))


# In[8]:


filename = 'XTrain_HOG 50000.sav'
X_train = pickle.load(open(filename, 'rb'))


# In[9]:


filename = 'XTest_HOG 10000.sav'
X_test = pickle.load(open(filename, 'rb'))


# In[17]:


from sklearn import preprocessing
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)


# In[18]:


X_train.shape


# In[19]:


SVM.support_vectors_.shape


# In[20]:


x_m2 = X_train[SVM.support_]
y_m2 = training_label[SVM.support_]


# In[21]:


from sklearn.svm import SVC
classifier = SVC(kernel ='rbf', random_state = 0,C=10)
 # training set in x, y axis
classifier.fit(x_m2, y_m2)


# In[27]:


filename = '1-4 HOG classifier.sav'
pickle.dump(classifier, open(filename, 'wb'))


# In[22]:


trainpredict = classifier.predict(x_m2)


# In[23]:


testpredict = classifier.predict(X_test)


# In[24]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_m2, trainpredict))


# In[25]:


print(classification_report(test_label, testpredict))


# In[ ]:
