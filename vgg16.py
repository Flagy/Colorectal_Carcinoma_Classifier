#!/usr/bin/env python
import keras
from keras import optimizers
from keras.models import Model
from image_loading import LoadingData
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import sys
import telepot
from Token import TOKEN
#import os

class VGG16():

    def __init__(self,model_name, n_classes=2):
        self.model_name=model_name
        self.n_classes=n_classes
        self.TOKEN = TOKEN
        self.TG_bot = telepot.Bot(self.TOKEN)
        self.chat_id= "-287388612"


    def callbacks(self):
        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.001, patience=5, verbose=1, mode='auto')

        checkpointer = keras.callbacks.ModelCheckpoint(monitor='val_acc', filepath='bestweightsmn.hdf5', verbose=1,
                                                   save_best_only=True, save_weights_only=False, mode='auto', period=1)

        reducelr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, verbose=1, mode='auto',
                                                 min_delta=0.001, cooldown=3, min_lr=0)
        epoch_print_callback = keras.callbacks.LambdaCallback(
            on_epoch_end = lambda epoch, logs: self.TG_bot.sendMessage(self.chat_id, "Epoch " +  str(epoch) + "val_acc: " + str(logs['val_acc'])),
            on_train_end = lambda logs: self.TG_bot.sendMessage(self.chat_id, "Training is over!"))

        callbacks_list = [checkpointer, earlyStopping, reducelr, epoch_print_callback]
        return callbacks_list


    def modelLoading(self,weights = 'imagenet',img_shape = (224, 224, 3)):
        base_model = keras.applications.vgg16.VGG16(include_top=False, weights=weights, input_tensor=None, input_shape=img_shape)
        print ("Model Loaded")
        return base_model


    def easyCompile(self,layer,totmodel,param="fulltraining"):
        print(param)
        if param=="fulltraining":
            self.TG_bot.sendMessage(self.chat_id,"Full training mode")
            for i in layer:
                i.trainable=True
        elif param =='ft':
            self.TG_bot.sendMessage(self.chat_id,"Fast training mode")
            for i in layer:
                i.trainable=False
        else:
            print('param has to be fulltraining or ft')
            sys.exit(0)
        loss='sparse_categorical_crossentropy'
        totmodel.compile(loss=loss,optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),metrics=['accuracy'])


    def dataaug(self):
        datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
        return datagen


    def savemodel(self,model,activ_func):
        # serialize model to JSON
        model_json = model.to_json()
        with open(self.model_name+activ_func+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(self.model_name+activ_func+"_w.h5")
        print("Saved model to disk")


    def plotAccLoss(self,history):
        # visualizzazione dei dati
        # summarize history for accuracy
        fig = plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title(self.model_name+' '+self.activ_func+' accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        #plt.show()
        fig.savefig(self.model_name+'_acc.png')
        plt.close()
        f = open(self.model_name+'_acc.png',"rb")
        self.TG_bot.sendPhoto(self.chat_id,f,caption="Accuracy of the model")
        f.close()
        # summarize history for loss
        fig = plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(self.model_name+' '+self.activ_func+' model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        #plt.show()
        fig.savefig(self.model_name+'_loss.png')
        plt.close()
        f = open(self.model_name+'_loss.png',"rb")
        self.TG_bot.sendPhoto(self.chat_id,f,caption="Loss of the model")
        f.close()


    def creation(self,base_model,activ_func):
        print("Model creation: ",self.model_name,self.n_classes,activ_func)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # Add fully-connected layer
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.2)(x)
        if activ_func == 'both':
            x = Dense(self.n_classes, activation="sigmoid")(x)
            x = Dense(self.n_classes, activation="softmax")(x)
        elif activ_func == 'sigmoid':
            x = Dense(self.n_classes, activation="sigmoid")(x)
        elif activ_func == 'softmax':
            x = Dense(self.n_classes, activation="softmax")(x)
        else:
            raise AssertionError('Wrong activ_func\t'+str(activ_func))
        return x


    def start(self,param='fulltraining',activ_func='both'):
        #caricamento dei dati
        self.activ_func = activ_func
        load=LoadingData("json",self.model_name)
        print(load.n_classes)
        self.n_classes=load.n_classes
        x_train=load.x_train
        print(np.shape(x_train))
        y_train=load.y_train
        x_test=load.x_test
        y_test=load.y_test
        #caricamento del modello
        model_base=self.modelLoading()
        #caricamento del modello output
        predictions=self.creation(model_base,activ_func)
        model = Model(inputs=model_base.input, outputs=predictions)
        self.easyCompile(model_base.layers,model,param)
        self.TG_bot.sendMessage(self.chat_id,"Data augmentation is disabled")
        history=model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=1, callbacks=self.callbacks(), validation_split=0.2, shuffle=True)
        #datavisualization
        #score = model.evaluate(x_test, y_test, batch_size = 32)
        self.savemodel(model,activ_func)
        self.plotAccLoss(history)
        self.TG_bot.sendMessage("Model evaluation: ",str(model.evaluate(x_test,y_test,batch_size=16)))
