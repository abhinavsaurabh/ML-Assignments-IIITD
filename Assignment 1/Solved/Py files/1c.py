#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.io import loadmat
from sklearn.manifold import TSNE

data1 = loadmat('/content/drive/My Drive/ML(PG)/ML(PG)_assignment_1/dataset_1.mat')


# In[3]:


data2 = data1['samples']
#data2 = data2.reshape(len(data2), 28,28,1)
data2 = data2.reshape(len(data2), 28*28)
mlabel = data1['labels'][0]
count= len(set(mlabel))


# In[4]:


data2.shape
data2


# In[5]:


data1


# In[6]:


data_X = data2
y = mlabel
tsne = TSNE(n_components=2, random_state=0)
tsne_obj= tsne.fit_transform(data_X)
tsne_obj


# In[8]:


tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
                        'Y':tsne_obj[:,1],
                        'Labels':y})
tsne_df.head()


# In[10]:


scatter = sns.scatterplot(x="X", y="Y",
              hue="Labels",
              palette=['purple','red','orange','brown','blue',
                       'dodgerblue','green','lightgreen','darkcyan', 'black'],
              legend='full',
              data=tsne_df);
scatter.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)


# In[ ]:




