#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# In[4]:


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


# In[ ]:


test_images.shape


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_images,train_labels,test_size=0.3,random_state = 0)


# In[ ]:


y_train.shape


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


y_test.shape


# In[ ]:





# In[6]:


model = models.Sequential()


model.add(layers.ZeroPadding2D(padding=(1,1), input_shape=(32, 32, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))


# In[7]:


model.summary()


# In[8]:


from keras.optimizers import SGD
#momentum=0.9, decay=1e-6,
sgd = SGD(lr=0.001, momentum=0.08)


# In[9]:


model.compile(optimizer=sgd,
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

from keras.utils import np_utils
y_train1 = np_utils.to_categorical(y_train)
y_test1 = np_utils.to_categorical(y_test)
history = model.fit(X_train, y_train1, epochs=200,validation_data=(X_test, y_test1))


# In[16]:


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(X_test, y_test1, verbose=2)
model.save('/content/drive/MyDrive/ML(PG)/ML-6/ML-6 1B')


# In[15]:


from google.colab import drive
drive.mount('/content/drive')


# In[17]:


print(test_acc)


# In[19]:


print(test_acc)
