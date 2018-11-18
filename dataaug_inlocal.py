import numpy as np
import keras
import random
from keras.preprocessing.image import ImageDataGenerator
x_T_new=[]
###############################LOADING DATASET#################################
x_T=np.load("V_train.npy")
print("loading")
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
print(x_T.shape)
for i in range(0,(10000-x_T.shape[0])):
    r=random.randint(0,x_T.shape[0]-1)
    x_temp=x_T[r][:][:][:]
    #x_temp = np.expand_dims(x_temp, axis=0) # 1 x input_shape
    x_trans = train_datagen.random_transform(x_temp)
    print (i)
    x_T_new.append(x_trans)

np.save("x_V_new.npy",x_T_new)

x_T_new=[]
x_T=np.load("H_train.npy")
print("loading")
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
print(x_T.shape)
for i in range(0,(10000-x_T.shape[0])):
    r=random.randint(0,x_T.shape[0]-1)
    x_temp=x_T[r][:][:][:]
    #x_temp = np.expand_dims(x_temp, axis=0) # 1 x input_shape
    x_trans = train_datagen.random_transform(x_temp)
    print (i)
    x_T_new.append(x_trans)

np.save("x_H_new.npy",x_T_new)


x_T_new=[]
x_T=np.load("AC_train.npy")
print("loading")
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
print(x_T.shape)
for i in range(0,(10000-x_T.shape[0])):
    r=random.randint(0,x_T.shape[0]-1)
    x_temp=x_T[r][:][:][:]
    #x_temp = np.expand_dims(x_temp, axis=0) # 1 x input_shape
    x_trans = train_datagen.random_transform(x_temp)
    print (i)
    x_T_new.append(x_trans)

np.save("x_AC_new.npy",x_T_new)


x_T_new=[]
x_T=np.load("S_train.npy")
print("loading")
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
print(x_T.shape)
for i in range(0,(10000-x_T.shape[0])):
    r=random.randint(0,x_T.shape[0]-1)
    x_temp=x_T[r][:][:][:]
    #x_temp = np.expand_dims(x_temp, axis=0) # 1 x input_shape
    x_trans = train_datagen.random_transform(x_temp)
    print (i)
    x_T_new.append(x_trans)

np.save("x_S_new.npy",x_T_new)

x_T_new=[]
x_T=np.load("T_train.npy")
print("loading")
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
print(x_T.shape)
for i in range(0,(10000-x_T.shape[0])):
    r=random.randint(0,x_T.shape[0]-1)
    x_temp=x_T[r][:][:][:]
    #x_temp = np.expand_dims(x_temp, axis=0) # 1 x input_shape
    x_trans = train_datagen.random_transform(x_temp)
    print (i)
    x_T_new.append(x_trans)

np.save("x_T_new.npy",x_T_new)
