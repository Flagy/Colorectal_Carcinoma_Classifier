# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:24:38 2018

@author: super

"""
import os



from keras.models import model_from_json
from keras import optimizers
from image_loading import LoadingData
from keras import backend as K
import numpy as np

act_func = 'Sigmoid' #'Softmax'
#Set working directory
os.chdir("C:/Users/super/OneDrive/Documenti/Project_bioinfo")
#Open the model, the model corresponds to the root node.
json_file = open("Nets/"+act_func+"/HAC.json","r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("Nets/"+act_func+"/HAC_w.h5")

sgd = optimizers.SGD(lr=0.0001, nesterov=True)
model.compile(optimizer=sgd,loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])

#Initialization parameters
dataset = {"HAC":(0,1),"STV":(0,1,2)}# 0 = AC/AD, 1 = H ; 0 = S, 1 = T, 2 = V.

L = []

for param in dataset.keys():
    print(param)
    data=LoadingData("json",param)
    for choice in dataset[param]:
        print("choice is ",choice)
        O = []
        y = data.y_train == choice
        pos = np.where(y)[0]
        flag = 0
        for i in pos:
            #print(i)
            a = data.x_train[i]
            check = data.y_train[i]
            assert(check == choice),"Something wrong!"
            a = np.expand_dims(a, axis=0)
            outputs = K.function([model.layers[0].input],[model.layers[-1].output])
            O.append(outputs([a])[0])
            flag +=1
            if flag == 300:
                break
        L.append(np.mean(O,0))
        print("added ",str(L[-1]))
        
L_dict = {'AC':L[0],'H':L[1],'S':L[2],'T':L[3],'V':L[4]}
np.save('Likelihood_matrix_'+act_func+'.npy',L)
