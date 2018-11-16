# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:24:38 2018

@author: super
"""
import keras
from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Dropout, Flatten, Dense,GlobalAveragePooling2D
from keras import applications
from keras.models import model_from_json
from keras import optimizers
from keras.applications.mobilenet import decode_predictions
import json
from numpy import array
from keras import backend as K
import numpy as np
import tensorflow as tf
import os

os.chdir('../')



class Testing():
    def __init__(self):

########################################ROOT MODEL################################
        json_file = open("Nets/Softmax/ACHS.json","r")
        model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(model_json)
        self.model.load_weights("Nets/Softmax/ACHS_w.h5"")
        self.model_graph = tf.get_default_graph()
        sgd = optimizers.SGD(lr=0.0001, nesterov=True)
        self.model.compile(optimizer=sgd,loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])

##########################ACTV MOdel#########################################################

        json_file = open("Nets/Softmax/ACTV.json","r")
        model_json = json_file.read()
        json_file.close()
        self.modelbranch2 = model_from_json(model_json)
        self.modelbranch2.load_weights("../Nets/Softmax/ACTV_w.h5")
        self.modelbranch2_graph = tf.get_default_graph()
        sgd = optimizers.SGD(lr=0.0001, nesterov=True)
        self.modelbranch2.compile(optimizer=sgd,loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])

        #############################5_classes model###########################################
        json_file = open("Nets/Softmax/vgg16.json","r")
        model_json = json_file.read()
        json_file.close()
        self.modeltocompare = model_from_json(model_json)
        self.modeltocompare.load_weights("Nets/Softmax/vgg16_w.h55")
        self.modeltocompare_graph = tf.get_default_graph()
        sgd = optimizers.SGD(lr=0.0001, nesterov=True)
        self.modeltocompare.compile(optimizer=sgd,loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])




    def giveLabel(self,y_final,y_tocompare,y_started,flag):
        dic={}
        maxpos=np.argmax(y_final[0][0])
        if flag==0 and maxpos==0:
            return(dic={"percentage":y_final[0][0][maxpos],"class":"AC"})


        if flag==1 and maxpos==1:
            return({"percentage":y_final[0][0][maxpos],"class":"H"})

        if flag==1 and maxpos==2:
            return({"percentage":y_final[0][0][maxpos],"class":"S"})
        if flag==0 and maxpos==1:
            return({"percentage":y_final[0][0][maxpos],"class":"T"})
        if flag==0 and maxpos==2:
            return({"percentage":y_final[0][0][maxpos],"class":"V"})




    def predict(self,img,param):
        label=["AC","H","S","T","V"]
        print(img.shape)
        #y_start è la variabile temporanea che per ogni predict del self.modello root classifica l'immagine in H o AC
        flag=[]
        y_final=[]#vettore che da root sottoclassifica specificamente l'immagine
        y_tocompare=[]#OUTPUT della classificazione per la rete a 5 classi
        y_started=[]
        #Eseguo test
        img = np.expand_dims(img, axis=0)
        with self.model_graph.as_default():
            y_start=self.model.predict(img)
        maxposition=np.argmax(y_start)#trovo la posizione del max in y_start
        if  maxposition==0: #somiglianza con AC
            with self.modelbranch2_graph.as_default():
                y_final.append(self.modelbranch2.predict(img))#ACTV prediction
            flag=0
        else:
            y_finel.append(y_start)
            flag=1
        with self.modeltocompare_graph.as_default():
            y_tocompare.append(self.modeltocompare.predict(img))#5_classes model prediction
        #Param 1 give the prediction for the 5_classes model
        if param=="1":
            return({"percentage":y_tocompare[0][0][np.argmax(y_tocompare[0][0])],"class":label[np.argmax(y_tocompare[0][0])]})
        #Other params perform prediction on the tree CNN model
        else:
            return self.giveLabel(y_final,y_tocompare,y_started,flag)
