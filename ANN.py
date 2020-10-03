# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 21:30:51 2020

@author: NEW
"""


import pandas as pd
import numpy as np
import seaborn as sns

data=pd.read_csv("E:\Krish naik\Deep_Learning\Churn_Modelling.csv") 

X=data.iloc[:,3:13]
y=data.iloc[:,13]

# Encoding
geography=pd.get_dummies(X.Geography,drop_first=True)
gender=pd.get_dummies(X.Gender,drop_first=True)

X=pd.concat([X,geography,gender],axis=1)
X=X.drop(["Gender","Geography"],axis=1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# Feature-Scaling

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()
classifier.add(Dense(output_dim=6,init='he_uniform',activation='relu',input_dim=11))
classifier.add(Dense(units=6, kernel_initializer="he_uniform",activation='relu',input_dim=11)) 


# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'he_uniform',activation='relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'glorot_uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
model_history=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10, nb_epoch = 100)

y_pred=classifier.predict(X_test)
y_pred=y_pred>0.5


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()