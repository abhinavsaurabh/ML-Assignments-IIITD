import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from sklearn.manifold import TSNE
import pandas as pd
import h5py
import joblib

myarray = np.fromfile('MNIST_Subset.h5', dtype=float)
print(myarray.shape)
f1 = h5py.File('MNIST_Subset.h5', 'r')
list(f1.keys())
X1 = f1['X']
y1=f1['Y']
df1= np.array(X1.value)
dfy1= np.array(y1.value)
print (df1.shape)
print (dfy1.shape)
f1.close() ####### If something broke, please check it f1.close should be commented

X_train, X_test, y_train, y_test = train_test_split(df1, dfy1, test_size=0.2, random_state=42)


d1, d2, d3 = X_train.shape
X_train = X_train.reshape((d1,d2*d3))
d1, d2, d3 = X_test.shape
X_test = X_test.reshape((d1,d2*d3))

mlp = MLPClassifier(hidden_layer_sizes=(100,50,50), activation = 'logistic')

mlp.fit(X_train,y_train)

y_pred_train = mlp.predict(X_train)
y_pred_test = mlp.predict(X_test)

print('Training and Testing accurcay resp. ',mlp.score(X_train,y_train),mlp.score(X_test,y_test))
print('Training and Testing loss resp.',np.square(np.subtract(y_train, y_pred_train)).mean(),np.square(np.subtract(y_test, y_pred_test)).mean())


def plot_db(X,y,clf,name):
    h = 0.2 # step size in the mesh
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    #print('INSIDE FUNCTION',xx.shape,yy.shape)
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    Z = Z.reshape(xx.shape)
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(name, fontsize=16)
    plt.xlabel('X0', fontsize=18)
    plt.ylabel('X1', fontsize=16)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='viridis', alpha=0.6, edgecolors='black', s=25)
    plt.show()
    

#TSNE EMBED
#X_train_embedded = TSNE(n_components=2).fit_transform(X_train)
#joblib.dump(X_train_embedded,'X_train_embedded')
X_train_embedded = joblib.load('X_train_embedded')

tsne_data = np.vstack((X_train_embedded.T,y_train)).T
tsne_df = pd.DataFrame(data=tsne_data,columns=("A","B","label"))
tsne_df.plot.scatter(x='A',y='B',c='label',colormap='viridis')
plt.show()

X_train_tsne = tsne_df.loc[:, tsne_df.columns != 'label']
y_train = tsne_df['label']

print(X_train_tsne.shape,y_train.shape)


def mlpp(alpha,XX,yy,name):
    #print(XX.shape,yy.shape)
    mlp_local = MLPClassifier(hidden_layer_sizes=(100,50,50), activation = 'logistic',alpha=alpha,random_state=42)
    #print(mlp_local)
    mlp_local.fit(XX,yy)
    y_preedd = mlp_local.predict(XX)
    #print(np.unique(y_preedd, return_counts=True)) ## DATASET HAS 2 classes 7 and 9, binary classification
    plot_db(XX,yy,mlp_local,name)


#alphas = np.logspace(-5, 0, 3)
alphas = [0.0000001,0.0001,1,100]
names = ['alpha ' + str(i) for i in alphas]

for i in range(len(names)):
    mlpp(alphas[i],X_train_tsne,y_train,name=names[i])