# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 19:42:00 2021

@author: LENOVO
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import models
import sklearn.metrics as metrics
import pandas as pd

dataset = np.loadtxt('preprocessed_online_shoping.csv', delimiter=',', skiprows=1)
#split the data for testing and training
data = dataset[:,1:12]
labels = dataset[:,12]
X_train, X_test, Y_train, Y_test = train_test_split(data, labels,test_size=0.2)

model=models.load_model('my_model.h5')
model.load_weights('weights.h5')


#define metrics
y_pred_proba = model.predict([X_test])
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba)
auc = metrics.roc_auc_score(Y_test, y_pred_proba)
#create ROC curve
plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


