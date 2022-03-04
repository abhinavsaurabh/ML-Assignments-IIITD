#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
df = pd.read_csv('sat.trn', sep='\s+',header=None)


# In[58]:


df


# In[83]:



feature_cols = list(df.columns) 
feature_cols = feature_cols[0:36]
target = ['class']
X_train = df[feature_cols]
y_train = df[36]


# In[84]:


df2 = pd.read_csv('sat.tst', sep='\s+',header=None)
feature_cols = list(df2.columns) 
feature_cols = feature_cols[0:36]
target = ['class']
X_test = df2[feature_cols]
y_test= df2[36]


# In[85]:


X_test


# In[86]:


from sklearn.manifold import TSNE
data_X = X_train
y = y_train
tsne = TSNE(n_components=2, random_state=0)
tsne_obj= tsne.fit_transform(data_X)
tsne_obj


# In[87]:


import pandas as pd
tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
                        'Y':tsne_obj[:,1],
                        'Labels':y})
tsne_df.head()


# In[88]:


import seaborn as sns
scatter = sns.scatterplot(x="X", y="Y",
              hue="Labels",
              palette=['purple','red','orange','brown','blue','green'],
                       #'dodgerblue','green','lightgreen','darkcyan', 'black'],
              legend='full',
              data=tsne_df);
scatter.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)


# In[ ]:




