#!/usr/bin/env python
# coding: utf-8

# In[5]:


CIFAR = 'cifar-10-batches-py/'
import numpy as np


# In[6]:


def unpickle(file):
    import pickle
    with open(file,'rb') as fo:
        cifar_dict = pickle.load(fo,encoding='bytes')
    return cifar_dict


# In[7]:


dirs = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']

full_data = [0,1,2,3,4,5,6]

for i, direc in zip(full_data,dirs):
    full_data[i] = unpickle(CIFAR+direc)


# In[8]:


batch_meta = full_data[0]
data_batch1 = full_data[1]
data_batch2 = full_data[2]
data_batch3 = full_data[3]
data_batch4 = full_data[4]
data_batch5 = full_data[5]
test_batch = full_data[6]


# In[ ]:





# In[9]:


train_all_data = [data_batch1,data_batch2,data_batch3,data_batch4,data_batch5]
test_batch1 = [test_batch]

training_image = np.vstack([d[b"data"] for d in train_all_data])
train_len = len(training_image)
training_image = training_image.reshape(train_len,3*32*32)
training_label = np.hstack([d[b"labels"] for d in train_all_data])


# In[ ]:


training_label


# In[10]:


test_image = np.vstack([d[b"data"] for d in test_batch1])
test_len = len(test_image)
test_image = test_image.reshape(test_len,3*32*32)
test_label = np.hstack([d[b"labels"] for d in test_batch1])


# In[ ]:


from sklearn.decomposition import PCA

pca = PCA(.90)
X_train = pca.fit_transform(training_image)
X_test = pca.transform(test_image)


# In[ ]:


from sklearn.manifold import TSNE
data_X = X_train
y = training_label
tsne = TSNE(n_components=2, random_state=0)
tsne_obj= tsne.fit_transform(data_X)
tsne_obj


# In[ ]:


import pickle
filename = 'TSNE_obj PCA.sav'
pickle.dump(tsne_obj, open(filename, 'wb'))


# In[1]:


import pickle
filename = 'TSNE/TSNE_obj PCA.sav'
tsne_obj = pickle.load(open(filename, 'rb'))


# In[12]:


y = training_label


# In[13]:


import pandas as pd
tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
                        'Y':tsne_obj[:,1],
                        'Labels':y})
tsne_df.head()


# In[15]:


import seaborn as sns
scatter = sns.scatterplot(x="X", y="Y",
              hue="Labels",
              palette=['purple','red','orange','brown','blue',
                       'dodgerblue','green','lightgreen','darkcyan', 'black'],
              legend='full',
              data=tsne_df);
scatter.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)


# In[ ]:
