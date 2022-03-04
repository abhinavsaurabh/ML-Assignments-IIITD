#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.io import loadmat

data1 = loadmat('dataset_2.mat')


# In[2]:


data1


# In[13]:


data2 = data1['samples']
#data2 = data2.reshape(len(data2), 28,28,1)
mlabel = data1['labels'][0]
#df1 = df = pd.DataFrame({mlabel,data2}) 
data3 = np.column_stack((data2,mlabel))
df = pd.DataFrame(data3)
df.columns = [0, 1,'Labels']

scatter = sns.scatterplot(data=df, x=0, y=1, hue='Labels',)
scatter.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
#fig=plt.figure()
#ax=fig.add_axes([0,0,1,1])
#ax.scatter(mlabel, data2, color='r')
#ax.scatter(grades_range, boys_grades, color='b')


# In[4]:


data2


# In[5]:


mlabel


# In[39]:


df


# In[ ]:




