#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat




data1 = loadmat('dataset_1.mat')
data2 = data1['samples']
data2 = data2.reshape(len(data2), 28,28,1)
mlabel = data1['labels'][0]
count= len(set(mlabel))


# In[2]:


mlabel.shape


# In[6]:


def plot_images_sample(X, Y):
    plt.figure(figsize=(10,10))
    #rand_indicies = np.random.randint(len(X), size=100)
    k=0
    c=0
    list1=[]
    for i in range(1000):
        if Y[i]==k:
            list1.append(i)
            c+=1
            if c==10:
                k+=1
                c=0
        
    for i in range(100):
        plt.subplot(10,10,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        #index = rand_indicies[i]
        index = list1[i]
        plt.imshow(np.squeeze(X[index]), cmap=plt.cm.binary)
        plt.xlabel(Y[index])
    plt.show()


plot_images_sample(data2, mlabel)


# In[5]:





# In[ ]:





# In[ ]:




