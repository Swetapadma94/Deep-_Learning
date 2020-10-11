# -*- coding: utf-8 -*-
"""fake-news-classification-Bidirectonal-LSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fzJPPme4N341WEUN91JnS_SxkkFD6_eJ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

df=pd.read_csv('/content/fake.csv')

df.head()

df=df.dropna()

X=df.drop('label',axis=1)

y=df.label

y.value_counts()

X.shape

y.shape

import tensorflow as tf

tf.__version__

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout

voc_size=5000

"""ONE-HOT Representation"""

message=X.copy()

message.head()

message.title[1]

message.reset_index(inplace=True)

import nltk
import re
from nltk.corpus import stopwords



nltk.download('stopwords')

### Dataset Preprocessing
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(message)):
    print(i)
    review = re.sub('[^a-zA-Z]', ' ', message['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

corpus

onehot_repr=[one_hot(words,voc_size)for words in corpus] 
onehot_repr

sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)

embedded_docs[0]

## Creating model
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

## Creating model
embedding_vector_features=40
model1=Sequential()
model1.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model1.add(Bidirectional(LSTM(100)))
model1.add(Dropout(0.3))
model1.add(Dense(1,activation='sigmoid'))
model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model1.summary())

len(embedded_docs),y.shape

import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)

X_final.shape,y_final.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)

### Finally Training
model1.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)

y_pred1=model1.predict_classes(X_test)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred1)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred1)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred1))

