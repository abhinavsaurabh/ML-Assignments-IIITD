#!/usr/bin/env python
# coding: utf-8

# In[1]:


CIFAR = '/content/drive/My Drive/Assignment/ML ASS 3/cifar-10-batches-py/'
import numpy as np


# In[2]:


def unpickle(file):
    import pickle
    with open(file,'rb') as fo:
        cifar_dict = pickle.load(fo,encoding='bytes')
    return cifar_dict


# In[3]:


dirs = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']

full_data = [0,1,2,3,4,5,6]

for i, direc in zip(full_data,dirs):
    full_data[i] = unpickle(CIFAR+direc)


# In[4]:


batch_meta = full_data[0]
data_batch1 = full_data[1]
data_batch2 = full_data[2]
data_batch3 = full_data[3]
data_batch4 = full_data[4]
data_batch5 = full_data[5]
test_batch = full_data[6]


# In[5]:


train_all_data = [data_batch1,data_batch2,data_batch3,data_batch4,data_batch5]
test_batch1 = [test_batch]

training_image = np.vstack([d[b"data"] for d in train_all_data])
train_len = len(training_image)
training_image = training_image.reshape(train_len,3*32*32)
training_label = np.hstack([d[b"labels"] for d in train_all_data])


# In[13]:


training_image.shape


# In[6]:


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


# In[ ]:


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


# In[9]:


from sklearn.manifold import TSNE
data_X = X_train_hog
y = training_label
tsne = TSNE(n_components=2, random_state=0)
tsne_obj= tsne.fit_transform(data_X)
tsne_obj


# In[10]:


import pickle
filename = 'TSNE HOG.sav'
pickle.dump(tsne_obj, open(filename, 'wb'))


# In[11]:


import pandas as pd
tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
                        'Y':tsne_obj[:,1],
                        'Labels':y})
tsne_df.head()


# In[12]:


import seaborn as sns
scatter = sns.scatterplot(x="X", y="Y",
              hue="Labels",
              palette=['purple','red','orange','brown','blue',
                       'dodgerblue','green','lightgreen','darkcyan', 'black'],
              legend='full',
              data=tsne_df);
scatter.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
