# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:24:38 2018

@author: super
"""

from keras.models import model_from_json
from keras import optimizers
from image_loading import LoadingData
import numpy as np
import os

os.chdir('../')

def giveLabel(y_final,y_real,y_vgg16):
    correct_AC=0
    correct_H=0
    correct_S=0
    correct_T=0
    correct_V=0
    correct_AC_5=0
    correct_H_5=0
    correct_S_5=0
    correct_T_5=0
    correct_V_5=0	
    #i dati sul databse 5 sono già presenti, dopo aver allenato sulle 5 classi, è stata fatta l'evaluation

    for i in range(len(y_real)):
        if y_real[i]==0:
            if y_final[i]==0:
                correct_AC=correct_AC+1
            if y_vgg16[i]==0:
                correct_AC_5=correct_AC_5+1
        elif y_real[i]==1:
            if y_final[i]==1:
                correct_H=correct_H+1
            if y_vgg16[i]==1:
                correct_H_5=correct_H_5+1
        elif y_real[i]==2:
            if y_final[i]==2:
                correct_S=correct_S+1
            if y_vgg16[i]==2:
                correct_S_5=correct_S_5+1
        elif y_real[i]==3:
            if y_final[i]==3:
                correct_T=correct_T+1
            if y_vgg16[i]==3:
                correct_T_5=correct_T_5+1
        elif y_real[i]==4:
            if y_final[i]==4:
                correct_V=correct_V+1
            if y_vgg16[i]==4:
                correct_V_5=correct_V_5+1
        else:
            raise AssertionError ("Something wrong in y_test")
    print("Prestazioni TREE-CNN:")
    print("AC:", str(correct_AC)+"/"+str(sum(y_real==0)))		
    print("H: ", str(correct_H)+"/"+str(sum(y_real==1)))
    print("S: ", str(correct_S)+"/"+str(sum(y_real==2)))
    print("T: ", str(correct_T)+"/"+str(sum(y_real==3)))
    print("V: ", str(correct_V)+"/"+str(sum(y_real==4)))
    print("Total Accuracy: ",str((correct_AC+correct_H+correct_S+correct_T+correct_V)/len(y_real)),'\n')
    print("Prestazioni VGG16:")
    print("AC:", str(correct_AC_5)+"/"+str(sum(y_real==0)))		
    print("H: ", str(correct_H_5)+"/"+str(sum(y_real==1)))
    print("S: ", str(correct_S_5)+"/"+str(sum(y_real==2)))
    print("T: ", str(correct_T_5)+"/"+str(sum(y_real==3)))
    print("V: ", str(correct_V_5)+"/"+str(sum(y_real==4)))
    print("Total Accuracy: ",str((correct_AC_5+correct_H_5+correct_S_5+correct_T_5+correct_V_5)/len(y_real)))    

#Open the root model
json_file = open("Nets/Softmax/ACHS.json","r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("Nets/Softmax/ACHS_w.h5")
sgd = optimizers.SGD(lr=0.0001, nesterov=True)
model.compile(optimizer=sgd,loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])

json_file = open("Nets/Softmax/ACTV.json","r")
model_json = json_file.read()
json_file.close()
modelbranch2 = model_from_json(model_json)
modelbranch2.load_weights("Nets/Softmax/ACTV_w.h5")
sgd = optimizers.SGD(lr=0.0001, nesterov=True)
modelbranch2.compile(optimizer=sgd,loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])

#loading 5 classes model
json_file = open("Nets/Softmax/vgg16.json","r")
model_json = json_file.read()
json_file.close()
modeltocompare = model_from_json(model_json)
modeltocompare.load_weights("Nets/Softmax/vgg16_w.h5")
sgd = optimizers.SGD(lr=0.0001, nesterov=True)
modeltocompare.compile(optimizer=sgd,loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])

#carichiamo i dati di test, basta caricare soltanto i dati di database 5
data=LoadingData("json",'5classes')
x_test=data.x_test
y_test=data.y_test
#y_start è la variabile temporanea che per ogni predict del modello root classifica l'immagine in H o AC
y_final=[]#vettore che da root sottoclassifica specificamente l'immagine
y_vgg16=[]
#Eseguo test
for i in range(0,x_test.shape[0]):
    if i%300 == 0:
        print(i)    
    x_temp=np.expand_dims(x_test[i,:,:,:], axis=0)
    y_vgg16.append(np.argmax(modeltocompare.predict(x_temp)))
    y_start=model.predict(x_temp)
    maxposition=np.argmax(y_start)
    if maxposition==0: #somiglianza con AC
        if np.argmax(modelbranch2.predict(x_temp))==0:
            y_final.append(0)# predict for ACTV
        else:
            y_final.append(np.argmax(modelbranch2.predict(x_temp))+2)
    else:
        y_final.append(np.argmax(y_start))  
giveLabel(y_final,y_test,y_vgg16)


