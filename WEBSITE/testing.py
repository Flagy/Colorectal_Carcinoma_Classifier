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




class Testing():
    def __init__(self):

        #Open the root self.model
        json_file = open("network/vgg16_HAC.json","r")
        model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(model_json)
        self.model.load_weights("network/vgg16_HAC_w.h5")
        self.model_graph = tf.get_default_graph()

        sgd = optimizers.SGD(lr=0.0001, nesterov=True)
        self.model.compile(optimizer=sgd,loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])

        json_file = open("network/HS.json","r")
        model_json = json_file.read()
        json_file.close()
        self.modelbranch1 = model_from_json(model_json)
        self.modelbranch1.load_weights("network/HS_w.h5")
        self.modelbranch1_graph = tf.get_default_graph()
        sgd = optimizers.SGD(lr=0.0001, nesterov=True)
        self.modelbranch1.compile(optimizer=sgd,loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])

        json_file = open("network/ACTV.json","r")
        model_json = json_file.read()
        json_file.close()
        self.modelbranch2 = model_from_json(model_json)
        self.modelbranch2.load_weights("network/ACTV_w.h5")
        self.modelbranch2_graph = tf.get_default_graph()
        sgd = optimizers.SGD(lr=0.0001, nesterov=True)
        self.modelbranch2.compile(optimizer=sgd,loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])
        json_file = open("network/vgg16.json","r")
        model_json = json_file.read()
        json_file.close()
        self.modeltocompare = model_from_json(model_json)
        self.modeltocompare.load_weights("network/vgg16.h5")
        self.modeltocompare_graph = tf.get_default_graph()
        sgd = optimizers.SGD(lr=0.0001, nesterov=True)
        self.modeltocompare.compile(optimizer=sgd,loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])




    def giveLabel(self,y_final,y_tocompare,y_started):
        dic={}
        print(y_final[0][0])
        quota=len(y_final[0][0])
        maxpos=np.argmax(y_final[0][0])
        poscompare=np.argmax(y_tocompare[0][0])
        if quota==3 and maxpos==0:
            dic={"percentage":y_final[0][0][maxpos],"class":"AC"}
            print(dic)
            return(dic)

        if quota==2 and maxpos==0:
            return({"percentage":y_final[0][0][maxpos],"class":"H"})

        if quota==2 and maxpos==1:
            return({"percentage":y_final[0][0][maxpos],"class":"S"})
        if quota==3 and maxpos==1:
            return({"percentage":y_final[0][0][maxpos],"class":"T"})
        if quota==3 and maxpos==2:
            return({"percentage":y_final[0][0][maxpos],"class":"V"})




    def predict(self,img,param):
        label=["AC","H","S","T","V"]
        print(img.shape)
        #y_start è la variabile temporanea che per ogni predict del self.modello root classifica l'immagine in H o AC
        y_final=[]#vettore che da root sottoclassifica specificamente l'immagine
        y_tocompare=[]#OUTPUT della classificazione per la rete a 5 classi
        y_started=[]
        #Eseguo test
        img = np.expand_dims(img, axis=0)
        with self.model_graph.as_default():
            y_start=self.model.predict(img)
            print(y_start[0][1])
            y_started.append(y_start)
        if  y_start[0][0]>y_start[0][1]: #somiglianza con AC
            with self.modelbranch2_graph.as_default():
                y_final.append(self.modelbranch2.predict(img))
        else:
            with self.modelbranch1_graph.as_default():
                y_final.append(self.modelbranch1.predict(img))
        with self.modeltocompare_graph.as_default():
            y_tocompare.append(self.modeltocompare.predict(img))
        print(y_final[0][0])
        if param=="1":
            return({"percentage":y_tocompare[0][0][np.argmax(y_tocompare[0][0])],"class":label[np.argmax(y_tocompare[0][0])]})
        else:
            return self.giveLabel(y_final,y_tocompare,y_started)
