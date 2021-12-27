# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 13:56:11 2021

@author: Mariam
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve
import sklearn.metrics as metrics
from sklearn.model_selection import learning_curve
#read data 
dataset=pd.read_csv("C:/Users/mariam/Downloads/preprocessed_online_shoping_intention.csv")
dataset.drop(columns=dataset.columns[0], 
        axis=1, 
        inplace=True)
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
#Splitting data
X_train,X_test,Y_train,Y_test= train_test_split(X,Y, test_size=0.25,random_state=0)
#feature scaling
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
#trainig model
classifier = SVC(C=1.0,kernel="linear",random_state=0)
classifier.fit(X_train,Y_train)
#predicting test set 
y_pred=classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1))
#confusion matrix and accurracy 
cs=confusion_matrix(Y_test,y_pred)
print(cs)
acc=accuracy_score(Y_test, y_pred)
print(acc*100)
#Displaying ROC

fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred)
auc = metrics.roc_auc_score(Y_test, y_pred)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()
#displaing loss curve
train_sizes, train_scores, test_scores, fit_times = learning_curve(classifier, X, Y, cv=30,return_times=True)
plt.plot(train_sizes,np.mean(train_scores,axis=1))
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

