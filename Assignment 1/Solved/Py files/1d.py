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


data_X = data2
y = mlabel  
tsne = TSNE(n_components=3, random_state=0)
tsne_obj= tsne.fit_transform(data_X)
tsne_obj


# In[ ]:


tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
                        'Y':tsne_obj[:,1],
                        'Labels':y})
tsne_df.head()


# In[17]:


import pandas as pd
df_subset = pd.DataFrame()
df_subset['X'] = tsne_obj[:,0]
df_subset['Y'] = tsne_obj[:,1]
df_subset['Z'] = tsne_obj[:,2]
df_subset['Labels'] = y
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
scatter = ax.scatter(
    xs=df_subset['X'], 
    ys=df_subset['Y'], 
    zs=df_subset['Z'], 
    c=df_subset['Labels'], 
    cmap='tab10'
)
legend1 = ax.legend(*scatter.legend_elements(),loc="upper left", title="Label")
ax.add_artist(legend1)
ax.set_xlabel('Tsne-one')
ax.set_ylabel('Tsne-two')
ax.set_zlabel('Tsne-three')
plt.show()

