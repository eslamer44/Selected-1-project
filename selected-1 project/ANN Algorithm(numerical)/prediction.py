# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 15:25:28 2021

@author: LENOVO
"""
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from keras import models

dataset = np.loadtxt('preprocessed_online_shoping.csv', delimiter=',', skiprows=1)

#split the data for testing and training
data = dataset[:,1:12]
labels = dataset[:,12]
X_train, X_test, Y_train, Y_test = train_test_split(data, labels,test_size=0.2)

c=models.load_model('my_model.h5')

c.load_weights('weights.h5')


y_pred = c.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, y_pred)
print(cm)
print(accuracy_score(Y_test, y_pred))


# print(c.predict(np.array([[0.0,0.0,0.0,0.0,7.0,0.0,18.855502398000006,0.0,1,1,0]])))

# print(c.predict(np.array([[3.0,173.9444444,0.0,0.0,14.0,0.0,18.855502398000006,0.0,0,0,1]])))

# print(c.predict(np.array([[0.0,0.0,0.0,0.0,1.0,0.058196969500000056,0.0,0.0,1,0,1]])))

# print(c.predict(np.array([[0.0,0.0,0.0,0.0,10.0,0.058196969500000056,18.855502398000006,0.0,0,0,1]])))