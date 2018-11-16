#!/usr/bin/env python
import json
import tkinter as Tk
import numpy as np
from tkinter.filedialog import askopenfilename

class LoadingData():
	def __init__(self,type, param):
		self.type=type
		self.param=param
		self.x_test=[]
		self.y_test=[]
		self.x_train=[]
		self.y_train=[]
		print('calling on choice')
		self.onChoice()


	def json_reading(self,file):
		with open(file) as f:
			conf = json.loads(f.read())
			conf = conf['CNN']
			f.close()
			return conf

	def onChoice(self):
		print('in onchoice')
		if self.type=="json":
			conf= self.json_reading("loading_conf.json")
			try:
				conf = conf[self.param]
			except:
				print('Wrong Param!')
			self.n_classes = conf['n_classes']
			self.datahere(conf)

		if self.type=="user_choice":
			path_list=[]
			file_num=4
			params_list=["x_train","y_train","x_test","y_test"]
			for i in range(file_num):
				Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
				filename = askopenfilename(title = "choose"+ params_list[i],filetypes = (("npy files","*.npy"),("all files","*.*")))
				path_list.append(filename)
			self.decodingfilename(path_list)


	def datahere(self,conf):
		print('datahere')
		try:
			self.x_train = np.load(conf['paths']['x_train'])
			self.x_train = np.array(self.x_train)
			self.y_train = np.load(conf['paths']['y_train'])
			self.x_test = np.load(conf['paths']['x_test'])
			self.x_test = np.array(self.x_test)
			self.y_test = np.load(conf['paths']['y_test'])
		except:
			raise AssertionError ("Problems with importation of data")
		print("All Data Loaded")


	def decodingfilename(self,path_list):
		try:
			self.x_train=np.load(path_list[0])
			self.y_train=np.load(path_list[1])
			self.x_test=np.load(path_list[2])
			self.y_test=np.load(path_list[3])
		except ValueError:
			print ("An error occurred in the data loading")


	def loading4predict(self,image_number):
		try:
			self.image_number= int(image_number)
			print (type(self.image_number))
			return (self.x_test[self.image_number,:,:,:], self.y_test[self.image_number])
		except IndexError:
			#AGGIUNGERE UN CONTROLLO SULLA DIMENSIONE!
			return
