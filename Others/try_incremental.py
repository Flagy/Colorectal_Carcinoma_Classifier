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

#Open the model
json_file = open("vgg16_HAC.json","r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("vgg16_HAC.h5")

sgd = optimizers.SGD(lr=0.0001, nesterov=True)
model.compile(optimizer=sgd,loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])

data=LoadingData("json","HAC")

O = []
choice = 0 # 0 = AD, 1 = H
y = data.getYTrain() == choice
pos = np.where(y)[0]
flag = 0
H_HAD = []
AD_HAD = []
for i in pos:
    a = data.getXTrain()[i]
    check = data.getYTrain()[i]
    if(check != choice):
        assert("Something wrong!")
    a = np.expand_dims(a, axis=0)
    AD_HAD.append(model.predict(a))
    '''outputs=K.function([model.layers[0].input],[model.layers[20].output])
    O.append(outputs([a])[0])'''
    flag +=1
    if flag == 300:
        break
    print(i)
    print(flag)
if choice == 1:
    np.save('predict_H_HAD_avg.npy',np.mean(H_HAD,0))
else:
    np.save('predict_AD_HAD_avg.npy',np.mean(AD_HAD,0))
    
predict_AD_HAD_avg = np.load('predict_AD_HAD_avg.npy')
predict_H_HAD_avg = np.load('predict_H_HAD_avg.npy')
'''layer_output = K.function([model.layers[21].input],
                                  [model.output])'''
np.save('output_pred_AD_HAD.npy',predict_H_HAD_avg)
np.save('output_pred_H_HAD.npy',predict_AD_HAD_avg)

import numpy as np
import os
os.chdir("C:/Users/super/OneDrive/Documenti/Project_bioinfo")
output_AC_HAC = np.load('output_AC_HAC.npy')
output_H_HAC = np.load('output_H_HAC.npy')
output_AD = np.load('output_AD.npy')
output_H = np.load('output_H.npy')
p_output_AD_HAD = np.load('output_pred_AD_HAD.npy')
p_output_H_HAD = np.load('output_pred_H_HAD.npy')
