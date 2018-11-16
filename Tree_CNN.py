from keras.models import model_from_json
from keras import optimizers
import numpy as np
import os

class TREE_CNN(object):

	def charge_model(self,param):
	    filename = param + ".json"
	    json_file = open(filename,"r")
	    model_json = json_file.read()
	    json_file.close()
	    model = model_from_json(model_json)
	    filename = param + "_w.h5"
	    model.load_weights(filename)
	    sgd = optimizers.SGD(lr=0.0001, nesterov=True)
		model.compile(optimizer=sgd,loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])
	    return model

	def __init__(self,param,vect):
		self.model = charge_model('HAC')
		pred = model.predict(vect)
		
	    if classification == 'AC':
	      	"""We are in the branch node AC which is composed by ACTV classes, so the ACTV classificator
	       	will be needed."""
	       	self.model = charge_model('ACTV')
	       	self.model.predict(vect)

	    print('\n',file = f)
	    f.close()
