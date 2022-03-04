import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

### Read Data
data=pd.read_csv('yelp_labelled.txt',delimiter='\t',header=None)
X = data[0].tolist() 
y = data[1].tolist() 

### Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

### Preprocess Data
def preprocess(X):
    processed_X = []
    
    PUNCT_TO_REMOVE = string.punctuation
    cachedStopWords = stopwords.words("english")
       
    for sent in X:
        sent = "".join([char for char in sent if char not in PUNCT_TO_REMOVE])
        sent = sent.lower()
        sent = " ".join([w for w in sent.split(' ') if w not in cachedStopWords])
        
        processed_X.append(sent)
        
    return processed_X

X_train_processed = preprocess(X_train)
X_test_processed = preprocess(X_test)

### Build Vocab
wordngram = defaultdict(int)
for sent in X_train_processed:
    for word in sent.split():
        wordngram[word] += 1

### Generating Feature Matrix
def build_feature_matrix(X):   
    feature_matrix = []
    
    for sent in X:
        word_vec = [0] * len(wordngram)       
        for word in sent.split():
            if word in wordngram:               
                word_vec[list(wordngram).index(word)] += 1
        feature_matrix.append(word_vec)
    
    return feature_matrix

X_train_fm = build_feature_matrix(X_train_processed)
X_test_fm = build_feature_matrix(X_test_processed)

### Training and Testing Naive Bayse Multinominal classifier
clf = MultinomialNB(alpha=1)
clf.fit(X_train_fm, y_train)

print('Training and Validation accurcay is: ',clf.score(X_train_fm,y_train), clf.score(X_test_fm,y_test))

### Finding misclassified validation samples
y_pred = clf.predict(X_test_fm)

y = 0
mismatch_index = []
for x in y_test: 
    if x != y_pred[y]: 
        mismatch_index.append(y) 
    y = y + 1

# for ind in mismatch_index:
#     print(X_test_processed[ind])

### Code for Post analysis
res = [i +' '+str(j) for i, j in zip(X_train_processed, y_train)]

X_train_pos = []
X_train_neg = []
for ind,y in enumerate(y_train):
    if y==0:
        X_train_neg.append(X[ind])
    else:
        X_train_pos.append(X[ind])

X_pos_word = []
for sent in X_train_pos:
    for w in sent.split():
        X_pos_word.append(w)
    
X_neg_word = []
for sent in X_train_neg:
    for w in sent.split():
        X_neg_word.append(w)
    
#print(len(X_pos_word),len(X_neg_word))
#print(clf.class_log_prior_)
# sentence "cant go wrong food"
#print(clf.feature_log_prob_[0][245],clf.feature_log_prob_[0][15],clf.feature_log_prob_[0][105],clf.feature_log_prob_[0][23])
#print(clf.feature_log_prob_[1][245],clf.feature_log_prob_[1][15],clf.feature_log_prob_[1][105],clf.feature_log_prob_[1][23])