# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 23:02:13 2021

@author: LENOVO
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


dataset = np.loadtxt('preprocessed_online_shoping.csv', delimiter=',', skiprows=1)

#split the data for testing and training
data = dataset[:,1:12]
labels = dataset[:,12]
X_train, X_test, Y_train, Y_test = train_test_split(data, labels,test_size=0.2)

# # # Build the model

model = Sequential()
model.add(Dense(11, activation='linear'))

model.add(Dense(28, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mse', optimizer='adam')

checkpoint = keras.callbacks.ModelCheckpoint(filepath="weights.h5", verbose=1, save_best_only=True)
history = model.fit(X_train, Y_train, epochs=100, batch_size=400, verbose=1, validation_split=0.2, callbacks=[checkpoint])
model.summary()

print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()



print(model.evaluate(X_test, Y_test))
print(model.metrics_names)

# model.save('my_model.h5',save_format='h5')




# from sklearn.metrics import confusion_matrix, accuracy_score
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# accuracy_score(y_test, y_pred)




