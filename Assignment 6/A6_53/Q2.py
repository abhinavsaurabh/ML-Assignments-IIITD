#Import package
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, Bidirectional
from sklearn.model_selection import train_test_split

#Load Data
df = pd.read_csv('ML_A6_Q2_data.txt',delimiter = '\t', engine='python', quoting = 3,header=None)
X=df[0]
y=df[1]


#Split
from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X, y , test_size = 0.20, random_state=42)

#Tokenizer
oov_token = "<OOV>"
max_length = 50 #Based on glove dimension
tokenizer = Tokenizer(oov_token=oov_token)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
vocab_size = len(word_index)+1
print('Unique tokens ',len(word_index))

#Text to sequence and Padding
padding_type='post'
trunction_type="post"
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_train_padded = pad_sequences(X_train_sequences,maxlen=max_length, padding=padding_type, 
                       truncating=trunction_type)
X_test_sequences = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_sequences,maxlen=max_length, 
                               padding=padding_type, truncating=trunction_type)

#Embedding
embeddings_index = {}
f = open('glove.6B.50d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


embedding_matrix = np.zeros((vocab_size, max_length))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector       

#Embedding Layer
embedding_layer = Embedding(input_dim=len(word_index) + 1, #vocab
                            output_dim=max_length, #each word dimension
                            weights=[embedding_matrix],
                            input_length=max_length, #sentence max length
                            trainable=False)
#Build Model
model = Sequential([
    embedding_layer,
    #tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(16,dropout=0.3,return_sequences=True)),
    #tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(16,dropout=0.3)),
    tf.keras.layers.SimpleRNN(16,dropout=0.3,activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    #tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
])

#Compile Model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

#Train Model
num_epochs = 500
batch_size=128
history = model.fit(X_train_padded, y_train, epochs=num_epochs, batch_size=batch_size,  validation_data=(X_test_padded, y_test))

#PLOTS
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#Results
print("Testing loss and accuracy", model.evaluate(X_test_padded, y_test, batch_size=128))
print("Train loss and accuracy: ",model.evaluate(X_train_padded, y_train, batch_size=128))