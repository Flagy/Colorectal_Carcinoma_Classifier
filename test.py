# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:24:38 2018

@author: super
"""

import keras
from keras import Model
from keras.models import model_from_json, Sequential
from keras import optimizers
from keras.applications.mobilenet import decode_predictions
from image_loading import LoadingData
import json
from numpy import array
from keras import backend as K
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp




def giveLabel(y_final,y_tocompare,y_real,y_started):
    correct_H=0
    uncorrect_H=0
    correct_S=0
    uncorrect_S=0
    correct_AC=0
    uncorrect_AC=0
    correct_T=0
    uncorrect_T=0
    correct_V=0
    uncorrect_V=0
    #i dati sul databse 5 sono già presenti, dopo aver allenato sulle 5 classi, è stata fatta l'evaluation
#    y_real=np.load("y_real.npy")
#    y_tocompare=np.load("y_tocompare.npy")
#    y_final=np.load("y_final.npy")
#    y_started=np.load("y_started.npy")
    for i in range(len(y_real)):
        print("for the "+str(i)+"Image the real label is:")
        if y_real[i][0]==0:
            print("AC")
            if len(y_final[i][0][0])==3 and np.argmax(y_final[i][0][0])==0:
                correct_AC=correct_AC+1
            else:
                uncorrect_AC=uncorrect_AC+1

        if y_real[i][0]==1:
            print("H")
            if len(y_final[i][0][0])==2 and np.argmax(y_final[i][0][0])==0:
                correct_H=correct_H+1
            else:
                uncorrect_H=uncorrect_H+1
        if y_real[i][0]==2:
            print("S")
            if len(y_final[i][0][0])==2 and np.argmax(y_final[i][0][0])==1:
                correct_S=correct_S+1
            else:
                uncorrect_S=uncorrect_S+1
        if y_real[i][0]==3:
            print("T")
            if len(y_final[i][0][0])==3 and np.argmax(y_final[i][0][0])==1:
                correct_T=correct_T+1
            else:
                uncorrect_T=uncorrect_T+1
        if y_real[i][0]==4:
            print("V")
            if len(y_final[i][0][0])==3 and np.argmax(y_final[i][0][0])==2:
                correct_V=correct_V+1
            else:
                uncorrect_V=uncorrect_V+1
    









#Open the root model
json_file = open("vgg16_HAC.json","r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("vgg16_HAC_w.h5")

sgd = optimizers.SGD(lr=0.0001, nesterov=True)
model.compile(optimizer=sgd,loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])



#open branch model
json_file = open("HS.json","r")
model_json = json_file.read()
json_file.close()
modelbranch1 = model_from_json(model_json)
modelbranch1.load_weights("HS_w.h5")
sgd = optimizers.SGD(lr=0.0001, nesterov=True)
modelbranch1.compile(optimizer=sgd,loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])



json_file = open("ACTV.json","r")
model_json = json_file.read()
json_file.close()
modelbranch2 = model_from_json(model_json)
modelbranch2.load_weights("ACTV_w.h5")
sgd = optimizers.SGD(lr=0.0001, nesterov=True)
modelbranch2.compile(optimizer=sgd,loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])



#loading 5 classes model
json_file = open("vgg16.json","r")
model_json = json_file.read()
json_file.close()
modeltocompare = model_from_json(model_json)
modeltocompare.load_weights("vgg16.h5")
sgd = optimizers.SGD(lr=0.0001, nesterov=True)
modeltocompare.compile(optimizer=sgd,loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])





#carichiamo i dati di test, basta caricare soltanto i dati di database 5
data=LoadingData("json",5)
x_test=data.getXTest()
y_test=data.getYTest()
#y_start è la variabile temporanea che per ogni predict del modello root classifica l'immagine in H o AC
y_final=[]#vettore che da root sottoclassifica specificamente l'immagine
y_tocompare=[]#OUTPUT della classificazione per la rete a 5 classi
y_real=[]#classe reale del classificato
y_started=[]
#Eseguo test
for i in range(0,x_test.shape[0]):
    print(i)
    y_final.append([])
    y_tocompare.append([])
    y_real.append([])
    y_started.append([])
    x_temp=x_test[i,:,:,:]
    x_temp=np.expand_dims(x_temp, axis=0)
    y_start=model.predict(x_temp)
    y_started[i].append(y_start)
    if  y_start[0][0]>y_start[0][1]: #somiglianza con AC
        y_final[i].append(modelbranch2.predict(x_temp))
    else:
       y_final[i].append(modelbranch1.predict(x_temp))
    y_tocompare[i].append(modeltocompare.predict(x_temp))
    y_real[i].append(y_test[i])
giveLabel(y_final,y_tocompare,y_real,y_started)

#np.save("y_final.npy",y_final)
#np.save("y_tocompare.npy",y_tocompare)
#np.save("y_real.npy",y_real)
#np.save("y_started.npy",y_started)









#Valuto i risultati


# Compute ROC curve and ROC area for each class
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#for i in range(n_classes):
#    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#    roc_auc[i] = auc(fpr[i], tpr[i])
#
## Compute micro-average ROC curve and ROC area
#fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
