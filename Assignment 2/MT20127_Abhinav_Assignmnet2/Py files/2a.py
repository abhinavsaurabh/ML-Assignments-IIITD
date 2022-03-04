#!/usr/bin/env python
# coding: utf-8

# In[15]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import pandas as pd
import seaborn as sns


# In[3]:


data1 = loadmat('dataset_1.mat')


# In[8]:


data1


# In[11]:


data2 = data1['samples']
mlabel = data1['labels'][0]
data3 = np.column_stack((data2,mlabel))
df = pd.DataFrame(data3)


# In[13]:


df.columns = [0, 1,'Labels']


# In[16]:


scatter = sns.scatterplot(data=df, x=0, y=1, hue='Labels',)
scatter.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)


# In[ ]:




